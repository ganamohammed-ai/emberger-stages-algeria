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
    """Climatological Q2 of a window from dicts of valid values.  None if coverage is too low."""
    n = y1 - y0 + 1
    py = [P[y] for y in range(y0, y1 + 1) if y in P]
    if len(py) < MIN_FRAC_P * n:
        return None
    mx, mn = [], []
    for m in range(1, 13):
        a = [TX[(y, m)] for y in range(y0, y1 + 1) if (y, m) in TX]
        b = [TN[(y, m)] for y in range(y0, y1 + 1) if (y, m) in TN]
        if len(a) < MIN_FRAC_M * n or len(b) < MIN_FRAC_M * n:
            return None
        mx.append(np.mean(a))
        mn.append(np.mean(b))
    Pm, M, m_ = float(np.mean(py)), float(max(mx)), float(min(mn))
    den = (M + 273.15) ** 2 - (m_ + 273.15) ** 2
    if den <= 0:
        return None
    return dict(n_yearsP=len(py), P=Pm, M=M, m=m_, Q2=2000.0 * Pm / den)


def stage_of(q2):
    return int(np.digitize(q2, STAGE_LIMITS))


def pct(a, b):
    return 100.0 * (b - a) / a if a else np.nan


def summarize_pair(df):
    """df: rows with dQ2_st, dQ2_tc (percent). Returns dict of summary numbers."""
    d = df.dropna(subset=["dQ2_st", "dQ2_tc"])
    n = len(d)
    if n < 3:
        return dict(n=n)
    diff = d["dQ2_tc"] - d["dQ2_st"]
    return dict(n=n, mean_st=d["dQ2_st"].mean(), mean_tc=d["dQ2_tc"].mean(), median_st=d["dQ2_st"].median(),
                median_tc=d["dQ2_tc"].median(), mean_diff_tc_minus_st=diff.mean(),
                se_diff=diff.std(ddof=1) / math.sqrt(n), r=float(np.corrcoef(d["dQ2_st"], d["dQ2_tc"])[0, 1]),
                sign_agree_pct=100.0 * float(np.mean(np.sign(d["dQ2_st"]) == np.sign(d["dQ2_tc"]))),
                MAE=float(np.abs(diff).mean()))


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
        a = W[W["window"] == w0].set_index("id")
        b = W[W["window"] == w1].set_index("id")
        for sid in a.index.intersection(b.index):
            r0, r1 = a.loc[sid], b.loc[sid]
            ch.append(dict(pair=f"{w0}->{w1}", id=sid, name=r0["name"], dQ2_st=pct(r0["Q2_st"], r1["Q2_st"]),
                           dQ2_tc=pct(r0["Q2_tc"], r1["Q2_tc"]), dP_st=pct(r0["P_st"], r1["P_st"]), dP_tc=pct(r0["P_tc"], r1["P_tc"]),
                           dM_st=r1["M_st"] - r0["M_st"], dM_tc=r1["M_tc"] - r0["M_tc"],
                           stage_change_st=stage_of(r1["Q2_st"]) - stage_of(r0["Q2_st"]),
                           stage_change_tc=stage_of(r1["Q2_tc"]) - stage_of(r0["Q2_tc"])))
    C = pd.DataFrame(ch)
    C.round(2).to_csv(os.path.join(OUT, "station_changes.csv"), index=False, encoding="utf-8-sig")

    L = ["Change of Q2 at stations versus TerraClimate (matched years), GHCN-Daily stations north of 33 N", ""]
    L.append(f"stations with a usable window per window: {W.groupby('window')['id'].nunique().to_dict()}")
    L.append("")
    fig, axs = plt.subplots(1, len(PAIRS), figsize=(4.2 * len(PAIRS), 4.2))
    for ax, (w0, w1) in zip(np.atleast_1d(axs), PAIRS):
        tag = f"{w0}->{w1}"
        d = C[C["pair"] == tag] if len(C) else C
        sm = summarize_pair(d) if len(d) else dict(n=0)
        L.append(f"{tag} ({WINDOWS[w0][0]}-{WINDOWS[w0][1]} -> {WINDOWS[w1][0]}-{WINDOWS[w1][1]}): " +
                 ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in sm.items()))
        if len(d):
            for st_c, tc_c in zip(d["stage_change_st"], d["stage_change_tc"]):
                pass
            agree = float(np.mean(d["stage_change_st"] == d["stage_change_tc"]))
            L.append(f"    stage-change agreement (same number of stages gained/lost): {100*agree:.0f}% of {len(d)} stations; "
                     f"station: {int((d['stage_change_st'] < 0).sum())} drier, {int((d['stage_change_st'] > 0).sum())} wetter; "
                     f"TerraClimate: {int((d['stage_change_tc'] < 0).sum())} drier, {int((d['stage_change_tc'] > 0).sum())} wetter")
            ax.scatter(d["dQ2_st"], d["dQ2_tc"], s=18)
            lim = [min(d["dQ2_st"].min(), d["dQ2_tc"].min()) - 3, max(d["dQ2_st"].max(), d["dQ2_tc"].max()) + 3]
            ax.plot(lim, lim, color="grey", lw=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("change of Q2 at station (%)")
        ax.set_ylabel("change of Q2 in TerraClimate (%)")
        ax.set_title(f"{tag}  (n={len(d)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_station_change.png"), dpi=200)
    plt.close(fig)
    L += ["", "Reading: points on the grey line mean TerraClimate reproduces the station change.  GHCN records are not tested",
          "for homogeneity (station moves, instrument changes); a mismatch can come from either side.",
          "Few stations have complete 1961-1980 records; report n with every number."]
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
