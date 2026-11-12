#!/usr/bin/env python3
"""
Gridded Emberger Q2 and bioclimatic-stage shift for northern Algeria (TerraClimate), validated at stations.

Needs (already installed on your PC): numpy, pandas, xarray, netCDF4 or h5netcdf, rasterio, matplotlib.
Run by double-clicking RUN_Grid_Q2_Shift.bat.  Inputs:
  - TerraClimate annual files  TerraClimate_{ppt,tmax,tmin}_{YYYY}_NordALG.nc   (D:\\DATA ALG_CLIP\\14_Climate_TerraClimate)
  - CHIRPS monthly tifs        chirps-v3.0.YYYY.MM_NordALG.tif                  (D:\\DATA ALG_CLIP\\06_Climate_CHIRPS)
  - results\\station_Q2_by_period.csv  (from Algeria_Q2_Stage_Shift.py)

Outputs in results_grid\\ :
  station_vs_grid.csv, grid_validation_summary.txt   (station values vs TerraClimate/CHIRPS, 1985-2025)
  Q2_P1.tif, Q2_P2.tif, dQ2_pct.tif, stage_P1.tif, stage_P2.tif, stage_shift.tif
  map_dQ2_pct.png, map_stage_P1_P2.png, map_stage_shift.png
  grid_shift_summary.txt                              (areas by stage, transitions, zonal means)

Method: Q2 = 2000 P / (M^2 - m^2), M,m in Kelvin; M = max over months of the period-mean monthly TMAX,
m = min over months of the period-mean monthly TMIN; P = mean annual precipitation.
"""
import os, sys, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_grid")
STATION_CSV = os.path.join(HERE, "results", "station_Q2_by_period.csv")
TC_DIR = r"D:\DATA ALG_CLIP\14_Climate_TerraClimate"
CHIRPS_DIR = r"D:\DATA ALG_CLIP\06_Climate_CHIRPS"

Y_ALL = (1985, 2025)
P1 = (1985, 2004)
P2 = (2005, 2025)

# stage limits are shared with the station script (edit them there)
from Algeria_Q2_Stage_Shift import STAGES
LIMITS = [lim for lim, _ in STAGES[:-1]]
STAGE_NAMES = [n for _, n in STAGES]


# ---------------- pure numpy functions ----------------
def q2_from(P, M, m):
    denom = (M + 273.15) ** 2 - (m + 273.15) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 2000.0 * P / denom
    q[~np.isfinite(q) | (denom <= 0)] = np.nan
    return q


def stage_index(q2):
    idx = np.digitize(q2, LIMITS).astype(np.float32)
    idx[~np.isfinite(q2)] = np.nan
    return idx


def climatology(tc_years):
    """tc_years: dict year -> (ppt[12,y,x], tmax[12,y,x], tmin[12,y,x]).  Returns P, M, m."""
    ys = sorted(tc_years)
    ppt = np.stack([tc_years[y][0] for y in ys])        # (n,12,y,x)
    tmax = np.stack([tc_years[y][1] for y in ys])
    tmin = np.stack([tc_years[y][2] for y in ys])
    P = np.nanmean(np.nansum(ppt, axis=1), axis=0)
    # require all 12 months present
    full = np.isfinite(ppt).all(axis=1).all(axis=0)
    P = np.where(full, P, np.nan)
    M = np.nanmax(np.nanmean(tmax, axis=0), axis=0)
    m = np.nanmin(np.nanmean(tmin, axis=0), axis=0)
    return P, M, m


# ---------------- I/O ----------------
def load_tc_year(var, year):
    import xarray as xr
    files = glob.glob(os.path.join(TC_DIR, "**", f"TerraClimate_{var}_{year}_NordALG.nc"), recursive=True)
    if not files:
        return None, None, None
    ds = xr.open_dataset(files[0])
    name = var if var in ds.data_vars else [v for v in ds.data_vars if v != "crs" and not v.startswith("crs")][0]
    da = ds[name]
    lat = ds["lat"].values
    lon = ds["lon"].values
    arr = da.transpose("time", "lat", "lon").values.astype("float32")
    ds.close()
    return arr, lat, lon


def build_tc(y0, y1, cache):
    data = {}
    lat = lon = None
    for y in range(y0, y1 + 1):
        if y in cache:
            data[y] = cache[y]
            continue
        parts = []
        for var in ("ppt", "tmax", "tmin"):
            arr, la, lo = load_tc_year(var, y)
            if arr is None:
                parts = None
                break
            parts.append(arr)
            lat, lon = la, lo
        if parts is None:
            print(f"  TerraClimate {y}: missing, skipped")
            continue
        cache[y] = tuple(parts)
        data[y] = cache[y]
    return data, lat, lon


def read_chirps_point_series(points, y0, y1):
    """Mean annual precipitation at points (nearest cell), years with all 12 months only."""
    import rasterio
    from rasterio.transform import rowcol
    annual = {i: [] for i in range(len(points))}
    for y in range(y0, y1 + 1):
        tot = [0.0] * len(points)
        ok = True
        for mth in range(1, 13):
            f = os.path.join(CHIRPS_DIR, f"chirps-v3.0.{y}.{mth:02d}_NordALG.tif")
            if not os.path.exists(f):
                ok = False
                break
            with rasterio.open(f) as r:
                a = r.read(1).astype("float32")
                nod = r.nodata
                for i, (la, lo) in enumerate(points):
                    try:
                        rr, cc = rowcol(r.transform, lo, la)
                        v = a[rr, cc]
                    except Exception:
                        v = np.nan
                    if v == nod or not np.isfinite(v) or v < 0:
                        v = np.nan
                    tot[i] += v
        if ok:
            for i in range(len(points)):
                annual[i].append(tot[i])
    return {i: (float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan) for i, v in annual.items()}


def write_tif(path, arr, lat, lon, nodata=-9999.0):
    import rasterio
    from rasterio.transform import from_origin
    a = arr.copy()
    la = lat.copy()
    if la[0] < la[-1]:           # ascending latitude -> flip so north is up
        a = a[::-1, :]
        la = la[::-1]
    dy = abs(la[1] - la[0])
    dx = abs(lon[1] - lon[0])
    tr = from_origin(lon.min() - dx / 2, la.max() + dy / 2, dx, dy)
    out = np.where(np.isfinite(a), a, nodata).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr, nodata=nodata, compress="lzw") as dst:
        dst.write(out, 1)


def nearest(arr, lat, lon, la, lo):
    i = int(np.argmin(np.abs(lat - la)))
    j = int(np.argmin(np.abs(lon - lo)))
    return float(arr[i, j])


def main():
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    cache = {}

    # ---------- 1. TerraClimate climatologies ----------
    print("Reading TerraClimate (1985-2025) ... a few minutes")
    d_all, lat, lon = build_tc(Y_ALL[0], Y_ALL[1], cache)
    if not d_all:
        print("No TerraClimate files found in", TC_DIR)
        sys.exit(1)
    P_all, M_all, m_all = climatology(d_all)
    Q_all = q2_from(P_all, M_all, m_all)
    d1 = {y: v for y, v in d_all.items() if P1[0] <= y <= P1[1]}
    d2 = {y: v for y, v in d_all.items() if P2[0] <= y <= P2[1]}
    P_1, M_1, m_1 = climatology(d1)
    P_2, M_2, m_2 = climatology(d2)
    Q1, Q2 = q2_from(P_1, M_1, m_1), q2_from(P_2, M_2, m_2)
    dQ = 100.0 * (Q2 - Q1) / Q1
    S1, S2 = stage_index(Q1), stage_index(Q2)
    SH = S2 - S1

    for name, arr in [("Q2_P1", Q1), ("Q2_P2", Q2), ("dQ2_pct", dQ), ("stage_P1", S1),
                      ("stage_P2", S2), ("stage_shift", SH)]:
        write_tif(os.path.join(OUT, name + ".tif"), arr, lat, lon)

    # ---------- 2. validation at stations ----------
    lines = []
    if os.path.exists(STATION_CSV):
        st = pd.read_csv(STATION_CSV)
        st = st[st["RECENT_1985_2025__Q2"].notna()].copy()
        pts = list(zip(st["lat"], st["lon"]))
        print(f"Reading CHIRPS at {len(pts)} stations (548 files) ...")
        chirps_P = read_chirps_point_series(pts, Y_ALL[0], Y_ALL[1])
        rec = []
        for k, (_, r) in enumerate(st.iterrows()):
            la, lo = float(r["lat"]), float(r["lon"])
            g = dict(id=r["id"], name=r["name"], lat=la, lon=lo, elev_station=r["elev"],
                     P_station=r["RECENT_1985_2025__P"], M_station=r["RECENT_1985_2025__M"],
                     m_station=r["RECENT_1985_2025__m"], Q2_station=r["RECENT_1985_2025__Q2"],
                     P_TerraClimate=nearest(P_all, lat, lon, la, lo),
                     M_TerraClimate=nearest(M_all, lat, lon, la, lo),
                     m_TerraClimate=nearest(m_all, lat, lon, la, lo),
                     Q2_TerraClimate=nearest(Q_all, lat, lon, la, lo),
                     P_CHIRPS=chirps_P[k])
            q_ch = q2_from(np.array([[g["P_CHIRPS"]]]), np.array([[g["M_TerraClimate"]]]), np.array([[g["m_TerraClimate"]]]))[0, 0]
            g["Q2_CHIRPSP_TCtemp"] = float(q_ch) if np.isfinite(q_ch) else np.nan
            rec.append(g)
        df = pd.DataFrame(rec)
        for c in ["P", "M", "m", "Q2"]:
            df[f"d{c}_TC"] = df[f"{c}_TerraClimate"] - df[f"{c}_station"]
        df["dP_CHIRPS"] = df["P_CHIRPS"] - df["P_station"]
        df["dQ2_CHIRPS"] = df["Q2_CHIRPSP_TCtemp"] - df["Q2_station"]
        df["stage_station"] = [stage_index(np.array([[q]]))[0, 0] for q in df["Q2_station"]]
        df["stage_TC"] = [stage_index(np.array([[q]]))[0, 0] for q in df["Q2_TerraClimate"]]
        df["stage_CHIRPS"] = [stage_index(np.array([[q]]))[0, 0] for q in df["Q2_CHIRPSP_TCtemp"]]
        df.round(2).to_csv(os.path.join(OUT, "station_vs_grid.csv"), index=False)

        def stats(a, b):
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3:
                return "n<3"
            a, b = np.array(a)[ok], np.array(b)[ok]
            r = np.corrcoef(a, b)[0, 1]
            return f"n={ok.sum()} bias={np.mean(b - a):+.2f} RMSE={np.sqrt(np.mean((b - a) ** 2)):.2f} r={r:.3f}"
        lines += ["Validation at stations (grid minus station; period 1985-2025; nearest cell)", ""]
        lines.append(f"P  TerraClimate vs station (mm): {stats(df['P_station'], df['P_TerraClimate'])}")
        lines.append(f"P  CHIRPS       vs station (mm): {stats(df['P_station'], df['P_CHIRPS'])}")
        lines.append(f"M  TerraClimate vs station (C):  {stats(df['M_station'], df['M_TerraClimate'])}")
        lines.append(f"m  TerraClimate vs station (C):  {stats(df['m_station'], df['m_TerraClimate'])}")
        lines.append(f"Q2 TerraClimate vs station:      {stats(df['Q2_station'], df['Q2_TerraClimate'])}")
        lines.append(f"Q2 CHIRPS-P+TC-T vs station:     {stats(df['Q2_station'], df['Q2_CHIRPSP_TCtemp'])}")
        okc = df["stage_station"].notna() & df["stage_TC"].notna()
        lines.append(f"Same stage as station: TerraClimate {100*(df['stage_station'][okc]==df['stage_TC'][okc]).mean():.0f}%  "
                     f"(n={okc.sum()}),  CHIRPS-P {100*(df['stage_station'][okc]==df['stage_CHIRPS'][okc]).mean():.0f}%")
        lines.append("")
        lines.append("NOTE: stations and grid cells differ in elevation/footprint; TerraClimate (~4 km) smooths relief.")
        open(os.path.join(OUT, "grid_validation_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
        print("\n".join(lines))
    else:
        print("station_Q2_by_period.csv not found - skipping validation.")

    # ---------- 3. area statistics of the shift ----------
    LAT = np.repeat(lat[:, None], len(lon), axis=1)
    W = np.cos(np.deg2rad(LAT))
    valid = np.isfinite(Q1) & np.isfinite(Q2)
    txt = [f"Grid: TerraClimate, P1={P1[0]}-{P1[1]}, P2={P2[0]}-{P2[1]}, valid cells={int(valid.sum())}", ""]
    txt.append("Area share by stage (%), area-weighted")
    txt.append(f"{'stage':12s} {'P1':>7s} {'P2':>7s}")
    for k, n in enumerate(STAGE_NAMES):
        a1 = 100 * W[valid & (S1 == k)].sum() / W[valid].sum()
        a2 = 100 * W[valid & (S2 == k)].sum() / W[valid].sum()
        txt.append(f"{n:12s} {a1:7.1f} {a2:7.1f}")
    txt.append("")
    changed = valid & (SH != 0)
    drier = valid & (SH < 0)
    wetter = valid & (SH > 0)
    txt.append(f"Area that changed stage: {100*W[changed].sum()/W[valid].sum():.1f}%  "
               f"(to drier stage {100*W[drier].sum()/W[valid].sum():.1f}%, to wetter stage {100*W[wetter].sum()/W[valid].sum():.1f}%)")
    txt.append(f"Mean dQ2: {np.nansum(dQ[valid]*W[valid])/W[valid].sum():+.1f}%   "
               f"mean dP: {100*(np.nansum((P_2-P_1)[valid]*W[valid])/W[valid].sum())/(np.nansum(P_1[valid]*W[valid])/W[valid].sum()):+.1f}%   "
               f"mean dM: {np.nansum((M_2-M_1)[valid]*W[valid])/W[valid].sum():+.2f} C   mean dm: {np.nansum((m_2-m_1)[valid]*W[valid])/W[valid].sum():+.2f} C")
    txt.append("")
    txt.append("Transition matrix (% of area), rows = P1 stage, columns = P2 stage")
    txt.append(" " * 12 + "".join(f"{n[:10]:>11s}" for n in STAGE_NAMES))
    for i, n in enumerate(STAGE_NAMES):
        row = []
        for j in range(len(STAGE_NAMES)):
            row.append(100 * W[valid & (S1 == i) & (S2 == j)].sum() / W[valid].sum())
        txt.append(f"{n:12s}" + "".join(f"{v:11.1f}" for v in row))
    txt.append("")
    txt.append("By latitude band")
    for a, b, lab in [(36, 99, ">=36N (littoral/Tell)"), (34, 36, "34-36N (Tell inland / High Plateaux)"), (0, 34, "<34N (Saharan Atlas edge)")]:
        z = valid & (LAT >= a) & (LAT < b)
        if z.sum() == 0:
            continue
        txt.append(f"  {lab:38s} dQ2 {np.nansum(dQ[z]*W[z])/W[z].sum():+6.1f}%   "
                   f"stage changed {100*W[z & (SH!=0)].sum()/W[z].sum():5.1f}%   to drier {100*W[z & (SH<0)].sum()/W[z].sum():5.1f}%")
    txt.append("")
    txt.append("NOTE: P1 vs P2 is a within-1985-2025 comparison (decadal variability matters). Stage limits are editable defaults.")
    open(os.path.join(OUT, "grid_shift_summary.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))

    # ---------- 4. maps ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        ext = [lon.min(), lon.max(), lat.min(), lat.max()]
        origin = "lower" if lat[0] < lat[-1] else "upper"
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(dQ, extent=ext, origin=origin, cmap="RdBu", vmin=-40, vmax=40)
        plt.colorbar(im, label="Change in Q2 (%), P2 vs P1")
        ax.set_title("Emberger Q2 change (TerraClimate)")
        fig.savefig(os.path.join(OUT, "map_dQ2_pct.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        cols = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"]
        cm = ListedColormap(cols[:len(STAGE_NAMES)])
        nm = BoundaryNorm(np.arange(-0.5, len(STAGE_NAMES)), cm.N)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        for ax, S, t in zip(axs, (S1, S2), (f"{P1[0]}-{P1[1]}", f"{P2[0]}-{P2[1]}")):
            im = ax.imshow(S, extent=ext, origin=origin, cmap=cm, norm=nm)
            ax.set_title("Bioclimatic stage " + t)
        cb = fig.colorbar(im, ax=axs, ticks=range(len(STAGE_NAMES)), shrink=0.8)
        cb.ax.set_yticklabels(STAGE_NAMES)
        fig.savefig(os.path.join(OUT, "map_stage_P1_P2.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(SH, extent=ext, origin=origin, cmap="RdBu", vmin=-2, vmax=2)
        plt.colorbar(im, label="Stage steps (negative = drier)")
        ax.set_title("Stage shift P2 vs P1")
        fig.savefig(os.path.join(OUT, "map_stage_shift.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("map drawing failed:", e)
    print("\nDONE. Results in:", OUT)


if __name__ == "__main__":
    main()
