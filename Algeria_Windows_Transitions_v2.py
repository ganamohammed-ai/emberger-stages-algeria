#!/usr/bin/env python3
"""
30-year windows, transition matrices, boundary-shift velocity, change-point test and robustness indices
(De Martonne, UNEP) for northern Algeria, from TerraClimate (native grid, about 4 km).

Run by double-clicking RUN_Windows_Transitions.bat.  Needs TerraClimate annual files
TerraClimate_{ppt,tmax,tmin[,pet]}_{YYYY}_NordALG.nc in D:\\DATA ALG_CLIP\\14_Climate_TerraClimate
(years 1958-2025; missing years are skipped; PET is optional: without it the UNEP index is skipped).

Outputs in results_grid\\windows\\ :
  transition_matrix_W0_W1.csv / transition_matrix_P1_P2.csv   (km2 by stage, NORD sans Sahara)
  moving_window_series.csv, fig_moving_windows.png            (30-year windows ending 1987..2025)
  boundary_shift.csv, fig_boundary_shift.png                  (km/decade; + = northward)
  annual_Q2_changepoint.csv                                   (Pettitt test per zone)
  robustness_indices.csv                                      (De Martonne, UNEP change per zone)
  Q2_W0.tif, Q2_W1.tif, dQ2_pct_W0_W1.tif, windows_summary.txt
Windows: W0 = 1961-1990, W1 = 1991-2020 (the classical normals); P1/P2 = 1985-2004 / 2005-2025.
Q2 = 2000 P / (M^2 - m^2), M = max monthly mean TMAX, m = min monthly mean TMIN (Kelvin).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Grid_Q2_Shift as G
import Algeria_Wilaya_Zones as Z

OUTW = os.path.join(G.OUT, "windows")
W0 = (1961, 1990)
W1 = (1991, 2020)
YEARS = range(1958, 2026)
STAGE_N = G.STAGE_NAMES
# De Martonne limits (EDIT if needed): <5 desert, 5-10 arid, 10-20 semi-arid, 20-30 mediterranean/subhumid, 30-60 humid, >=60 very humid
DM_LIM = [5, 10, 20, 30, 60]
DM_NAMES = ["desert", "aride", "semi-aride", "mediterraneen", "humide", "tres humide"]
# UNEP aridity index P/PET: <0.05 hyper-arid, <0.2 arid, <0.5 semi-arid, <0.65 dry subhumid, >=0.65 humid
UNEP_LIM = [0.05, 0.2, 0.5, 0.65]
UNEP_NAMES = ["hyper-aride", "aride", "semi-aride", "subhumide sec", "humide"]


# ------------------------------------------------------------------ pure numpy functions
def year_stats(ppt, tmax, tmin, pet=None):
    """ppt,tmax,tmin,pet: (12,y,x).  Returns dict of per-year fields; NaN unless all 12 months exist."""
    full = np.isfinite(ppt).all(axis=0)
    d = dict(P=np.where(full, ppt.sum(axis=0), np.nan), Tx=tmax, Tn=tmin,
             Tm=np.where(full, 0.5 * (tmax + tmin).mean(axis=0), np.nan))
    d["PET"] = np.where(np.isfinite(pet).all(axis=0), pet.sum(axis=0), np.nan) if pet is not None else None
    return d


def window_fields(per_year, y0, y1):
    """Mean P, PET, Tm and Emberger M, m over years y0..y1 (all years must be present)."""
    ys = list(range(y0, y1 + 1))
    if any(y not in per_year for y in ys):
        return None
    P = np.nanmean(np.stack([per_year[y]["P"] for y in ys]), axis=0)
    Tm = np.nanmean(np.stack([per_year[y]["Tm"] for y in ys]), axis=0)
    M = np.nanmax(np.nanmean(np.stack([per_year[y]["Tx"] for y in ys]), axis=0), axis=0)
    m = np.nanmin(np.nanmean(np.stack([per_year[y]["Tn"] for y in ys]), axis=0), axis=0)
    PET = None
    if all(per_year[y]["PET"] is not None for y in ys):
        PET = np.nanmean(np.stack([per_year[y]["PET"] for y in ys]), axis=0)
    return dict(P=P, Tm=Tm, M=M, m=m, PET=PET, Q2=G.q2_from(P, M, m))


def de_martonne(P, Tm):
    with np.errstate(invalid="ignore", divide="ignore"):
        i = P / (Tm + 10.0)
    i[(Tm + 10.0) <= 0] = np.nan
    return i


def unep_ai(P, PET):
    with np.errstate(invalid="ignore", divide="ignore"):
        a = P / PET
    a[~np.isfinite(a)] = np.nan
    return a


def class_index(v, limits):
    idx = np.digitize(v, limits).astype(np.float32)
    idx[~np.isfinite(v)] = np.nan
    return idx


def transition_matrix(s0, s1, area, mask, names):
    """km2 moving from stage i (rows, earlier) to stage j (columns, later)."""
    n = len(names)
    T = np.zeros((n, n))
    ok = mask & np.isfinite(s0) & np.isfinite(s1)
    a, b, w = s0[ok].astype(int), s1[ok].astype(int), area[ok]
    np.add.at(T, (a, b), w)
    return pd.DataFrame(T.round(0), index=[f"{x} (before)" for x in names], columns=[f"{x} (after)" for x in names])


def isoline_lat(profile, lat, limit):
    """Southernmost latitude where the (south->north ascending) profile first reaches `limit` (linear interpolation)."""
    ok = np.isfinite(profile)
    if ok.sum() < 3:
        return np.nan
    la, p = lat[ok], profile[ok]
    above = p >= limit
    if not above.any() or above.all():
        return np.nan
    k = int(np.argmax(above))
    if k == 0:
        return np.nan
    f = (limit - p[k - 1]) / (p[k] - p[k - 1]) if p[k] != p[k - 1] else 0.0
    return float(la[k - 1] + f * (la[k] - la[k - 1]))


def boundary_shift(Q0, Q1, lat, lon, limit, bands, mask, years_between):
    """Per longitude band: latitude shift of the Q2=limit isoline between two windows, in km per decade (+ = northward)."""
    rows = []
    for lo, hi in bands:
        cols = (lon >= lo) & (lon < hi)
        if cols.sum() == 0:
            continue
        m = mask[:, cols]
        p0 = np.where(m, Q0[:, cols], np.nan)
        p1 = np.where(m, Q1[:, cols], np.nan)
        if np.isfinite(p0).sum() < 10:
            continue
        with np.errstate(all="ignore"):
            prof0 = np.nanmedian(p0, axis=1)
            prof1 = np.nanmedian(p1, axis=1)
        l0, l1 = isoline_lat(prof0, lat, limit), isoline_lat(prof1, lat, limit)
        if np.isfinite(l0) and np.isfinite(l1):
            rows.append(dict(lon_band=f"{lo:g}..{hi:g}", limit=limit, lat_W0=round(l0, 3), lat_W1=round(l1, 3),
                             shift_km=round((l1 - l0) * 111.32, 1),
                             km_per_decade=round((l1 - l0) * 111.32 / (years_between / 10.0), 1)))
    return pd.DataFrame(rows)


def pettitt(x):
    """Pettitt change-point test. Returns (index of last point of 1st regime, U, approx p)."""
    x = np.asarray(x, float)
    n = len(x)
    s = np.sign(x[:, None] - x[None, :])
    U = np.array([s[: t + 1, t + 1:].sum() for t in range(n - 1)])
    k = int(np.argmax(np.abs(U)))
    K = abs(U[k])
    p = min(1.0, 2.0 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2)))
    return k, float(K), float(p)


def zone_mean(field, area, mask):
    ok = mask & np.isfinite(field)
    return float(np.average(field[ok], weights=area[ok])) if ok.any() else np.nan


# ------------------------------------------------------------------ I/O
def load_all():
    per_year = {}
    lat = lon = None
    for y in YEARS:
        arrs = {}
        for var in ("ppt", "tmax", "tmin", "pet"):
            a, la, lo = G.load_tc_year(var, y)
            if a is None:
                if var == "pet":
                    continue
                arrs = None
                break
            arrs[var] = a
            lat, lon = la, lo
        if arrs is None:
            print(f"  {y}: missing, skipped")
            continue
        per_year[y] = year_stats(arrs["ppt"], arrs["tmax"], arrs["tmin"], arrs.get("pet"))
        print(f"  {y} read", flush=True)
    return per_year, lat, lon


def zone_labels(lat, lon):
    import geopandas as gpd
    com = gpd.read_file(Z.SHP).to_crs(4326)
    com["geometry"] = com.geometry.simplify(0.003, preserve_topology=True).make_valid()
    com = com[com.geometry.notna() & ~com.geometry.is_empty]
    wil = com.dissolve(by="name_fr_1", as_index=False)[["name_fr_1", "geometry"]]
    wil["geometry"] = wil.geometry.buffer(0.004).buffer(-0.004)
    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoDataFrame({"k": np.arange(LON.size)}, geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326)
    j = gpd.sjoin(pts, wil, predicate="within", how="inner")
    lab = np.full(LON.size, "", dtype=object)
    lab[j["k"].values] = [Z.ZONE_OF.get(Z.norm(w), "Autre") for w in j["name_fr_1"].values]
    return lab.reshape(LON.shape)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTW, exist_ok=True)

    print("Reading TerraClimate 1958-2025 (several minutes) ...", flush=True)
    per_year, lat, lon = load_all()
    if not per_year:
        print("No TerraClimate files found in", G.TC_DIR)
        sys.exit(1)
    if lat[0] > lat[-1]:                       # make latitude ascending (south -> north)
        lat = lat[::-1]
        for d in per_year.values():
            for k, v in list(d.items()):
                if v is not None:
                    d[k] = v[..., ::-1, :]
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    LATG = np.repeat(lat[:, None], len(lon), axis=1)
    area = (111.32 * dlat) * (111.32 * dlon) * np.cos(np.deg2rad(LATG))
    print("Assigning cells to wilayas/zones (Nord_Comm_Wilaya) ...", flush=True)
    lab = zone_labels(lat, lon)
    nord = np.isin(lab, Z.NORD_ZONES)
    masks = {zn: (lab == zn) for zn in Z.ZONES}
    masks["NORD sans Sahara"] = nord
    zorder = Z.NORD_ZONES + ["NORD sans Sahara", "Sahara"]
    lines = [f"Windows and transitions, TerraClimate, W0={W0}, W1={W1}, P1={G.P1}, P2={G.P2}",
             f"cells in NORD sans Sahara: {int(nord.sum())}, area {area[nord].sum():.0f} km2", ""]

    fw0 = window_fields(per_year, *W0)
    fw1 = window_fields(per_year, *W1)
    fp1 = window_fields(per_year, *G.P1)
    fp2 = window_fields(per_year, *G.P2)
    if fw0 is None or fw1 is None:
        lines.append("W0 or W1 incomplete (missing years)")

    # ---------- transition matrices ----------
    for tag, a, b in (("W0_W1", fw0, fw1), ("P1_P2", fp1, fp2)):
        if a is None or b is None:
            continue
        s0, s1 = G.stage_index(a["Q2"]), G.stage_index(b["Q2"])
        T = transition_matrix(s0, s1, area, nord, STAGE_N)
        T.to_csv(os.path.join(OUTW, f"transition_matrix_{tag}.csv"), encoding="utf-8-sig")
        tot = T.values.sum()
        diag = np.trace(T.values)
        dry = np.tril(T.values, -1).sum()
        wet = np.triu(T.values, 1).sum()
        lines += [f"Transition matrix {tag} (km2, NORD sans Sahara):", T.to_string(), "",
                  f"  unchanged {100*diag/tot:.1f}%, to drier stage {100*dry/tot:.1f}%, to wetter stage {100*wet/tot:.1f}%", ""]
        if tag == "W0_W1":
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            pct = 100 * T.values / np.maximum(T.values.sum(axis=1, keepdims=True), 1)
            im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
            for i in range(pct.shape[0]):
                for k in range(pct.shape[1]):
                    ax.text(k, i, f"{pct[i, k]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if pct[i, k] > 55 else "black")
            ax.set_xticks(range(len(STAGE_N)))
            ax.set_xticklabels(STAGE_N, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(STAGE_N)))
            ax.set_yticklabels(STAGE_N, fontsize=8)
            ax.set_xlabel(f"stage {W1[0]}-{W1[1]}")
            ax.set_ylabel(f"stage {W0[0]}-{W0[1]}")
            ax.set_title("Transition matrix (% of the earlier stage's area), NORD sans Sahara", fontsize=9)
            fig.colorbar(im, ax=ax, label="%")
            fig.savefig(os.path.join(OUTW, "fig_transition_matrix.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ---------- choose the comparison pair: the 30-year normals if available, else P1 -> P2 ----------
    if fw0 is not None and fw1 is not None:
        TAG, YB, LBL = "W0_W1", ((W1[0] + W1[1]) - (W0[0] + W0[1])) / 2.0, f"{W0[0]}-{W0[1]} -> {W1[0]}-{W1[1]}"
        A0, A1 = fw0, fw1
    else:
        TAG, YB, LBL = "P1_P2", ((G.P2[0] + G.P2[1]) - (G.P1[0] + G.P1[1])) / 2.0, f"{G.P1[0]}-{G.P1[1]} -> {G.P2[0]}-{G.P2[1]}"
        A0, A1 = fp1, fp2
        lines.append(f"NOTE: 30-year normals unavailable (TerraClimate files start in {min(per_year)}); using {LBL} for rasters, boundary shift and indices.")
        lines.append("")
    # ---------- rasters and zone table ----------
    if A0 is not None and A1 is not None:
        dq = 100 * (A1["Q2"] - A0["Q2"]) / A0["Q2"]
        for nm, arr in ((f"Q2_{TAG.split(chr(95))[0]}", A0["Q2"]), (f"Q2_{TAG.split(chr(95))[1]}", A1["Q2"]), (f"dQ2_pct_{TAG}", dq)):
            try:
                G.write_tif(os.path.join(OUTW, nm + ".tif"), arr, lat, lon)
            except Exception as e:
                print("  could not write", nm, e)
        lines.append(f"Zone means {LBL} (area-weighted):")
        for zn in zorder:
            mk = masks[zn]
            q0, q1 = zone_mean(A0["Q2"], area, mk), zone_mean(A1["Q2"], area, mk)
            s0, s1 = G.stage_index(A0["Q2"]), G.stage_index(A1["Q2"])
            A = area[mk & np.isfinite(s0) & np.isfinite(s1)]
            dry = 100 * A[(s1 < s0)[mk & np.isfinite(s0) & np.isfinite(s1)]].sum() / A.sum() if A.size else np.nan
            lines.append(f"  {zn:27s} Q2 {q0:6.1f} -> {q1:6.1f} ({100*(q1-q0)/q0:+5.1f}%)  area to drier stage {dry:5.1f}%")
        lines.append("")

        # ---------- boundary velocity ----------
        bands = [(-2.5, -1.0), (-1.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 6.5), (6.5, 8.0), (8.0, 9.2)]
        yb = YB
        mask_geo = np.isfinite(A0["Q2"]) & np.isfinite(A1["Q2"]) & (lab != "Sahara") & (lab != "")
        bs = pd.concat([boundary_shift(A0["Q2"], A1["Q2"], lat, lon, L, bands, mask_geo, yb)
                        for L in (20, 40, 70)], ignore_index=True)
        bs.to_csv(os.path.join(OUTW, "boundary_shift.csv"), index=False, encoding="utf-8-sig")
        lines.append(f"Boundary shift (isoline of Q2 = 20 / 40 / 70) between window centres, {yb:.0f} years apart; + = northward:")
        if len(bs):
            for L, g in bs.groupby("limit"):
                lines.append(f"  Q2={L:g}: median {g['km_per_decade'].median():+.1f} km/decade over {len(g)} longitude bands "
                             f"(range {g['km_per_decade'].min():+.1f} to {g['km_per_decade'].max():+.1f})")
            fig, ax = plt.subplots(figsize=(8, 4))
            for k, (L, g) in enumerate(bs.groupby("limit")):
                ax.bar(np.arange(len(g)) + 0.27 * (k - 1), g["km_per_decade"], 0.27, label=f"Q2 = {L:g}")
                if k == 1:
                    ax.set_xticks(np.arange(len(g)))
                    ax.set_xticklabels(g["lon_band"], rotation=30, ha="right", fontsize=7)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("km per decade (+ = northward)")
            ax.set_title("Shift of stage boundaries by longitude band, W0 to W1", fontsize=10)
            ax.legend(fontsize=8)
            fig.savefig(os.path.join(OUTW, "fig_boundary_shift.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        lines.append("")

        # ---------- robustness indices ----------
        rob = []
        for zn in zorder:
            mk = masks[zn]
            row = dict(zone=zn)
            dm0, dm1 = de_martonne(A0["P"], A0["Tm"]), de_martonne(A1["P"], A1["Tm"])
            row["DeMartonne_before"], row["DeMartonne_after"] = zone_mean(dm0, area, mk), zone_mean(dm1, area, mk)
            c0, c1 = class_index(dm0, DM_LIM), class_index(dm1, DM_LIM)
            ok = mk & np.isfinite(c0) & np.isfinite(c1)
            row["DM_pct_area_drier_class"] = 100 * area[ok & (c1 < c0)].sum() / area[ok].sum() if ok.any() else np.nan
            if A0["PET"] is not None and A1["PET"] is not None:
                a0, a1 = unep_ai(A0["P"], A0["PET"]), unep_ai(A1["P"], A1["PET"])
                row["UNEP_AI_before"], row["UNEP_AI_after"] = zone_mean(a0, area, mk), zone_mean(a1, area, mk)
                u0, u1 = class_index(a0, UNEP_LIM), class_index(a1, UNEP_LIM)
                ok = mk & np.isfinite(u0) & np.isfinite(u1)
                row["UNEP_pct_area_drier_class"] = 100 * area[ok & (u1 < u0)].sum() / area[ok].sum() if ok.any() else np.nan
            rob.append(row)
        rob = pd.DataFrame(rob).round(3)
        rob.to_csv(os.path.join(OUTW, "robustness_indices.csv"), index=False, encoding="utf-8-sig")
        lines += [f"Robustness indices {LBL}:", rob.to_string(index=False), ""]
        if A0["PET"] is None or A1["PET"] is None:
            lines.append("  (TerraClimate PET files not found: UNEP index skipped)")

    # ---------- moving 30-year windows ----------
    series = []
    for end in range(1987, 2026):
        f = window_fields(per_year, end - 29, end)
        if f is None:
            continue
        s = G.stage_index(f["Q2"])
        for zn in zorder:
            mk = masks[zn]
            r = dict(end_year=end, zone=zn, Q2_mean=round(zone_mean(f["Q2"], area, mk), 2))
            for k, n in enumerate(STAGE_N):
                r[f"km2_{n}"] = round(area[mk & (s == k)].sum(), 0)
            series.append(r)
    if series:
        ser = pd.DataFrame(series)
        ser.to_csv(os.path.join(OUTW, "moving_window_series.csv"), index=False, encoding="utf-8-sig")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        nn = ser[ser["zone"] == "NORD sans Sahara"]
        for n in STAGE_N:
            axs[0].plot(nn["end_year"], nn[f"km2_{n}"] / 1000, label=n)
        axs[0].set_xlabel("end year of the 30-year window")
        axs[0].set_ylabel("area (1000 km2)")
        axs[0].set_title("Area by bioclimatic stage, NORD sans Sahara", fontsize=10)
        axs[0].legend(fontsize=8)
        for zn in Z.NORD_ZONES:
            g = ser[ser["zone"] == zn]
            axs[1].plot(g["end_year"], g["Q2_mean"], label=zn)
        axs[1].set_xlabel("end year of the 30-year window")
        axs[1].set_ylabel("mean Q2")
        axs[1].set_title("Mean Q2 by zone (30-year moving windows)", fontsize=10)
        axs[1].legend(fontsize=7)
        fig.savefig(os.path.join(OUTW, "fig_moving_windows.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        lines.append(f"Moving windows written: {ser['end_year'].min()}..{ser['end_year'].max()} (ends)")

    # ---------- annual Q2 and Pettitt ----------
    ys = sorted(per_year)
    ann = {zn: [] for zn in zorder}
    for y in ys:
        d = per_year[y]
        q = G.q2_from(d["P"], d["Tx"].max(axis=0), d["Tn"].min(axis=0))
        for zn in zorder:
            ann[zn].append(zone_mean(q, area, masks[zn]))
    cp = []
    for zn in zorder:
        x = np.array(ann[zn])
        ok = np.isfinite(x)
        if ok.sum() < 20:
            continue
        k, K, p = pettitt(x[ok])
        yy = np.array(ys)[ok]
        cp.append(dict(zone=zn, change_after_year=int(yy[k]), mean_before=round(x[ok][:k + 1].mean(), 2),
                       mean_after=round(x[ok][k + 1:].mean(), 2), U=K, p_approx=round(p, 4)))
    if cp:
        cp = pd.DataFrame(cp)
        cp.to_csv(os.path.join(OUTW, "annual_Q2_changepoint.csv"), index=False, encoding="utf-8-sig")
        lines += ["", "Pettitt change-point test on annual Q2 (area-mean per zone):", cp.to_string(index=False),
                  "(annual Q2 uses each year's own hottest/coldest month; p is the usual approximation, ignores autocorrelation)"]

    open(os.path.join(OUTW, "windows_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\nDONE. Results in", OUTW)


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
    os.makedirs(OUTW, exist_ok=True)
    _log = open(os.path.join(OUTW, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        open(os.path.join(OUTW, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\windows\\error_log.txt - send it to Claude.")
        sys.exit(1)
