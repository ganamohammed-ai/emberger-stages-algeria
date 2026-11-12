#!/usr/bin/env python3
"""
Does TerraClimate reproduce the CHANGE in Q2 seen at stations?  (validation of the change, not only of the level)

For every GHCN-Daily station of northern Algeria (lat >= 33) in ghcn_daily_cache with enough data, Q2 is computed for
several windows from the station record and from the TerraClimate cell nearest to the station, using exactly the same
valid years / months as the station (matched sampling).  The change between windows is then compared.

Run by double-clicking RUN_Station_Change_Validation.bat.  Needs: ghcn_daily_cache\\*.dly, stations_north_GHCN.csv,
TerraClimate files 1958-2025 (as the other scripts), xarray.  Outputs in results_grid\\station_change\\ :
  station_windows.csv        per station and window: coverage, station P/M/m/Q2 and TerraClimate (matched) P/M/m/Q2
  station_changes.csv        per station and window pair: change of Q2 (%), P (%), M, m at the station and in TerraClimate
  station_change_summary.txt summary per pair (n, mean change, TerraClimate minus station, correlation, sign agreement)
  fig_station_change.png     scatter of TerraClimate change against station change
Windows: B1 1961-1980, B2 1981-2000, B3 2001-2020, W0 1961-1990, W1 1991-2020, E1 1961-1984, E2 1985-2004.
Rules: a window needs >= 60% of its years with valid annual P (every month >= 20 valid days) and, for each calendar month,
>= 50% of the years with a valid monthly mean TMAX and TMIN (>= 15 valid days).  Station flags other than blank are dropped.
"""
import os, sys, csv, math, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Q2_Stage_Shift as S
import Algeria_Grid_Q2_Shift as G

CACHE = os.path.join(HERE, "ghcn_daily_cache")
OUT = os.path.join(G.OUT, "station_change")
WINDOWS = {"B1": (1961, 1980), "B2": (1981, 2000), "B3": (2001, 2020), "W0": (1961, 1990), "W1": (1991, 2020),
           "E1": (1961, 1984), "E2": (1985, 2004)}
PAIRS = [("B1", "B2"), ("B2", "B3"), ("W0", "W1"), ("E1", "E2")]
VARS = [("dQ2", "Q2", True), ("dP", "P", True), ("dM", "M", False), ("dm", "m", False)]   # (name, column, relative change in %)
MIN_FRAC_P, MIN_FRAC_M = 0.6, 0.5
STAGE_LIMITS = [lim for lim, _ in S.STAGES[:-1]]


# ------------------------------------------------------------------ pure functions
def station_series(raw):
    """raw from S.read_dly -> (P_by_year, TX_by_ym, TN_by_ym), only valid entries."""
    P, TX, TN = {}, {}, {}
    years = sorted({y for (y, m) in raw})
    for y in years:
        tot, ok = 0.0, True
        for m in range(1, 13):
            p = raw.get((y, m), {}).get("PRCP", [])
            if len(p) < S.MIN_DAYS_PRCP:
                ok = False
                break
            tot += sum(p) / len(p) * S.MONTH_DAYS[m - 1]
        if ok:
            P[y] = tot
        for m in range(1, 13):
            d = raw.get((y, m))
            if not d:
                continue
            if len(d["TMAX"]) >= S.MIN_DAYS_TEMP:
                TX[(y, m)] = float(np.mean(d["TMAX"]))
            if len(d["TMIN"]) >= S.MIN_DAYS_TEMP:
                TN[(y, m)] = float(np.mean(d["TMIN"]))
    return P, TX, TN


def window_q2(P, TX, TN, y0, y1):
    """Climatological P, M, m and Q2 of a window from dicts of valid values.  Each of P, M, m is NaN when its own coverage is
    too low; Q2 is NaN unless all three exist.  Returns None only if nothing at all is valid."""
    n = y1 - y0 + 1
    py = [P[y] for y in range(y0, y1 + 1) if y in P]
    Pm = float(np.mean(py)) if len(py) >= MIN_FRAC_P * n else np.nan
    mx, mn = [], []
    for m in range(1, 13):
        a = [TX[(y, m)] for y in range(y0, y1 + 1) if (y, m) in TX]
        b = [TN[(y, m)] for y in range(y0, y1 + 1) if (y, m) in TN]
        mx.append(np.mean(a) if len(a) >= MIN_FRAC_M * n else np.nan)
        mn.append(np.mean(b) if len(b) >= MIN_FRAC_M * n else np.nan)
    M = float(max(mx)) if not np.isnan(mx).any() else np.nan
    m_ = float(min(mn)) if not np.isnan(mn).any() else np.nan
    if np.isnan(Pm) and np.isnan(M) and np.isnan(m_):
        return None
    q2 = np.nan
    if not (np.isnan(Pm) or np.isnan(M) or np.isnan(m_)):
        den = (M + 273.15) ** 2 - (m_ + 273.15) ** 2
        if den > 0:
            q2 = 2000.0 * Pm / den
    return dict(n_yearsP=len(py), P=Pm, M=M, m=m_, Q2=q2)


def stage_of(q2):
    return int(np.digitize(q2, STAGE_LIMITS))


def pct(a, b):
    return 100.0 * (b - a) / a if a else np.nan


def summarize_pair(df, nm="dQ2"):
    """df: rows with <nm>_st and <nm>_tc.  Returns dict of summary numbers for that variable."""
    d = df.dropna(subset=[f"{nm}_st", f"{nm}_tc"])
    n = len(d)
    if n < 3:
        return dict(n=n)
    x, y = d[f"{nm}_st"], d[f"{nm}_tc"]
    diff = y - x
    return dict(n=n, mean_st=x.mean(), mean_tc=y.mean(), mean_diff_tc_minus_st=diff.mean(),
                se_diff=diff.std(ddof=1) / math.sqrt(n), r=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan,
                sign_agree_pct=100.0 * float(np.mean(np.sign(x) == np.sign(y))), MAE=float(np.abs(diff).mean()))


# ------------------------------------------------------------------ TerraClimate at the stations
def tc_point_series(stations, years):
    """Returns dict var -> {station_id: {(y, m): value}} for ppt, tmax, tmin (nearest cell, monthly)."""
    out = {v: {s["id"]: {} for s in stations} for v in ("ppt", "tmax", "tmin")}
    for y in years:
        for var in ("ppt", "tmax", "tmin"):
            arr, la, lo = G.load_tc_year(var, y)
            if arr is None:
                continue
            la, lo = np.asarray(la, float), np.asarray(lo, float)
            for s in stations:
                i = int(np.argmin(np.abs(la - s["lat"])))
                j = int(np.argmin(np.abs(lo - s["lon"])))
                for m in range(arr.shape[0]):
                    v = float(arr[m, i, j])
                    if np.isfinite(v):
                        out[var][s["id"]][(y, m + 1)] = v
        print(f"  TerraClimate {y} read", flush=True)
    return out


def tc_matched(tcP, tcX, tcN, P, TX, TN):
    """TerraClimate dicts restricted to the station's valid years / year-months (matched sampling)."""
    Pm = {}
    for y in P:
        mons = [tcP.get((y, m)) for m in range(1, 13)]
        if all(v is not None for v in mons):
            Pm[y] = float(sum(mons))
    return Pm, {k: tcX[k] for k in TX if k in tcX}, {k: tcN[k] for k in TN if k in tcN}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    meta = list(csv.DictReader(open(os.path.join(HERE, "stations_north_GHCN.csv"), encoding="utf-8")))
    sts = []
    for s in meta:
        p = os.path.join(CACHE, s["id"] + ".dly")
        if float(s["lat"]) >= S.LAT_MIN_STUDY and os.path.exists(p):
            sts.append(dict(id=s["id"], name=s["name"], lat=float(s["lat"]), lon=float(s["lon"]), elev=s["elev"], path=p))
    print(f"{len(sts)} stations with a .dly file (lat >= {S.LAT_MIN_STUDY})", flush=True)
    series = {}
    for s in sts:
        series[s["id"]] = station_series(S.read_dly(s["path"]))
    print("Reading TerraClimate at the stations (several minutes) ...", flush=True)
    tc = tc_point_series(sts, range(1958, 2026))

    rows = []
    for s in sts:
        P, TX, TN = series[s["id"]]
        Pm, Xm, Nm = tc_matched(tc["ppt"][s["id"]], tc["tmax"][s["id"]], tc["tmin"][s["id"]], P, TX, TN)
        for w, (y0, y1) in WINDOWS.items():
            a = window_q2(P, TX, TN, y0, y1)
            b = window_q2(Pm, Xm, Nm, y0, y1)
            if a is None or b is None:
                continue
            rows.append(dict(id=s["id"], name=s["name"], lat=s["lat"], lon=s["lon"], elev=s["elev"], window=w,
                             n_yearsP=a["n_yearsP"], P_st=a["P"], M_st=a["M"], m_st=a["m"], Q2_st=a["Q2"],
                             P_tc=b["P"], M_tc=b["M"], m_tc=b["m"], Q2_tc=b["Q2"]))
    if not rows:
        print("No station has enough valid data in any window (or no TerraClimate files were found).")
        sys.exit(1)
    W = pd.DataFrame(rows)
    W.round(2).to_csv(os.path.join(OUT, "station_windows.csv"), index=False, encoding="utf-8-sig")

    ch = []
    for w0, w1 in PAIRS:
        a_ = W[W["window"] == w0].set_index("id")
        b_ = W[W["window"] == w1].set_index("id")
        for sid in a_.index.intersection(b_.index):
            r0, r1 = a_.loc[sid], b_.loc[sid]
            row = dict(pair=f"{w0}->{w1}", id=sid, name=r0["name"])
            for nm, col, rel in VARS:
                for suf in ("st", "tc"):
                    x0, x1 = r0[f"{col}_{suf}"], r1[f"{col}_{suf}"]
                    row[f"{nm}_{suf}"] = pct(x0, x1) if rel else x1 - x0
            for suf in ("st", "tc"):
                q0, q1 = r0[f"Q2_{suf}"], r1[f"Q2_{suf}"]
                row[f"stage_change_{suf}"] = stage_of(q1) - stage_of(q0) if np.isfinite(q0) and np.isfinite(q1) else np.nan
            ch.append(row)
    C = pd.DataFrame(ch)
    C.round(2).to_csv(os.path.join(OUT, "station_changes.csv"), index=False, encoding="utf-8-sig")

    L = ["Change of Q2, P, M and m at stations versus TerraClimate (matched years), GHCN-Daily stations north of 33 N", ""]
    cov = {w: {k: int(W[(W['window'] == w)][c].notna().sum()) for k, c in (("P", "P_st"), ("M", "M_st"), ("m", "m_st"))}
           for w in WINDOWS}
    L.append(f"stations with a usable value, by window: {cov}")
    L.append("")
    fig, axs = plt.subplots(len(VARS), len(PAIRS), figsize=(4.0 * len(PAIRS), 3.6 * len(VARS)), squeeze=False)
    units = {"dQ2": "%", "dP": "%", "dM": "degC", "dm": "degC"}
    for r_i, (nm, col, rel) in enumerate(VARS):
        for c_i, (w0, w1) in enumerate(PAIRS):
            ax = axs[r_i][c_i]
            tag = f"{w0}->{w1}"
            d = C[C["pair"] == tag].dropna(subset=[f"{nm}_st", f"{nm}_tc"]) if len(C) else C
            sm = summarize_pair(d, nm) if len(d) else dict(n=0)
            L.append(f"{nm} {tag} ({WINDOWS[w0][0]}-{WINDOWS[w0][1]} -> {WINDOWS[w1][0]}-{WINDOWS[w1][1]}): " +
                     ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in sm.items()))
            if nm == "dQ2" and len(d):
                dd = d.dropna(subset=["stage_change_st", "stage_change_tc"])
                if len(dd):
                    L.append(f"    stage change, same number of stages gained/lost in {100*float(np.mean(dd['stage_change_st'] == dd['stage_change_tc'])):.0f}% "
                             f"of {len(dd)} stations; station: {int((dd['stage_change_st'] < 0).sum())} drier, {int((dd['stage_change_st'] > 0).sum())} wetter; "
                             f"TerraClimate: {int((dd['stage_change_tc'] < 0).sum())} drier, {int((dd['stage_change_tc'] > 0).sum())} wetter")
            if len(d):
                ax.scatter(d[f"{nm}_st"], d[f"{nm}_tc"], s=16)
                lo_ = min(d[f"{nm}_st"].min(), d[f"{nm}_tc"].min())
                hi_ = max(d[f"{nm}_st"].max(), d[f"{nm}_tc"].max())
                pad = 0.1 * (hi_ - lo_ + 1e-9)
                ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], color="grey", lw=0.8)
                ax.axhline(0, color="black", lw=0.5)
                ax.axvline(0, color="black", lw=0.5)
            ax.set_xlabel(f"{nm} station ({units[nm]})", fontsize=8)
            ax.set_ylabel(f"{nm} TerraClimate ({units[nm]})", fontsize=8)
            ax.set_title(f"{tag}  n={len(d)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_station_change.png"), dpi=170)
    plt.close(fig)
    L += ["", "Reading: points on the grey line mean TerraClimate reproduces the station change.  GHCN records are not tested",
          "for homogeneity (station moves, instrument changes); a mismatch can come from either side.",
          "Daily PRCP before 2000 is almost absent in GHCN for northern Algeria, so dP and dQ2 before 2001 rest on very few stations;",
          "TMAX/TMIN are much more complete, so dM and dm are the meaningful checks of the early TerraClimate years."]
    open(os.path.join(OUT, "station_change_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("\nDONE. Results in", OUT)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, t):
        for st in self.streams:
            try:
                st.write(t)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    _log = open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        open(os.path.join(OUT, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\station_change\\error_log.txt - send it to Claude.")
        sys.exit(1)
