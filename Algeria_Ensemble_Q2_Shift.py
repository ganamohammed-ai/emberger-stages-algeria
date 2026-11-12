#!/usr/bin/env python3
"""
Multi-product ensemble of the bioclimatic-stage shift (Emberger Q2), northern Algeria.

Uses the ERA5-Land HOURLY 2 m temperature (D:\\DATA ALG_CLIP\\16_ERA5Land_Hourly_T2M_NordALG) to build
daily TMAX/TMIN (local time = UTC+1) and monthly means, then combines temperature and precipitation
products on the ERA5 0.1 degree grid:

   variant A : P = TerraClimate , T = TerraClimate
   variant B : P = CHIRPS       , T = ERA5-Land
   variant C : P = CHIRPS       , T = TerraClimate
   variant D : P = TerraClimate , T = ERA5-Land

For every variant: Q2 for P1 (1985-2004) and P2 (2005-2025), stage shift, area statistics.
Agreement among variants shows how robust the shift is.  ERA5 temperature is also compared with
TerraClimate and with the stations (M, m).

Run by double-clicking RUN_Ensemble_Q2_Shift.bat (needs Algeria_Grid_Q2_Shift.py in the same folder).
First run: reads 492 hourly files (about 10-25 min); the monthly result is cached in results_grid.
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Grid_Q2_Shift as G

ERA_DIR = r"D:\DATA ALG_CLIP\16_ERA5Land_Hourly_T2M_NordALG"
OUT = G.OUT
os.makedirs(OUT, exist_ok=True)
CACHE_NPZ = os.path.join(OUT, "ERA5_monthly_TxTn_1985_2025.npz")
LOCAL_UTC_OFFSET = 1            # Algeria = UTC+1 all year
MIN_HOURS_PER_DAY = 22
MIN_DAYS_PER_MONTH = 20
MIN_YEAR_FRACTION = 0.7         # share of years required in a period for a monthly mean


# ---------------- pure functions ----------------
def monthly_txtn_from_hourly(a_celsius):
    """a_celsius: (n_hours, y, x) hourly T2m starting at 00:00 UTC of the 1st day of the month.
    Returns (mean daily max, mean daily min) of LOCAL days (UTC+offset); days with too few hours are skipped."""
    n = a_celsius.shape[0]
    h = np.arange(n)
    local_day = (h + LOCAL_UTC_OFFSET) // 24
    txs, tns = [], []
    for d in range(local_day.max() + 1):
        idx = np.where(local_day == d)[0]
        if len(idx) < MIN_HOURS_PER_DAY:
            continue
        blk = a_celsius[idx]
        txs.append(np.nanmax(blk, axis=0))
        tns.append(np.nanmin(blk, axis=0))
    if len(txs) < MIN_DAYS_PER_MONTH:
        return None, None
    return np.nanmean(np.stack(txs), axis=0), np.nanmean(np.stack(tns), axis=0)


def t_climatology(TX, TN, years, y0, y1):
    """TX, TN: (n_years, 12, y, x). Returns M, m for the years inside [y0, y1]."""
    sel = [i for i, y in enumerate(years) if y0 <= y <= y1]
    tx, tn = TX[sel], TN[sel]
    need = MIN_YEAR_FRACTION * len(sel)
    okx = np.isfinite(tx).sum(axis=0) >= need
    okn = np.isfinite(tn).sum(axis=0) >= need
    mx = np.where(okx, np.nanmean(tx, axis=0), np.nan)       # (12,y,x)
    mn = np.where(okn, np.nanmean(tn, axis=0), np.nan)
    M = np.where(np.isfinite(mx).all(axis=0), np.nanmax(mx, axis=0), np.nan)
    m = np.where(np.isfinite(mn).all(axis=0), np.nanmin(mn, axis=0), np.nan)
    return M, m


# ---------------- I/O ----------------
def era_grid():
    import rasterio
    f = glob.glob(os.path.join(ERA_DIR, "**", "ERA5_Land_T2M_Hourly_*_NordALG.tif"), recursive=True)[0]
    with rasterio.open(f) as r:
        t = r.transform
        lat = t.f + t.e * (np.arange(r.height) + 0.5)
        lon = t.c + t.a * (np.arange(r.width) + 0.5)
    return lat, lon


def build_era5_monthly(years):
    import rasterio
    lat, lon = era_grid()
    TX = np.full((len(years), 12, len(lat), len(lon)), np.nan, "float32")
    TN = TX.copy()
    for iy, y in enumerate(years):
        print(f"  ERA5-Land {y}")
        for m in range(1, 13):
            f = os.path.join(ERA_DIR, str(y), f"ERA5_Land_T2M_Hourly_{y}_{m:02d}_NordALG.tif")
            if not os.path.exists(f):
                print("   missing", os.path.basename(f))
                continue
            with rasterio.open(f) as r:
                a = r.read().astype("float32")
                nod = r.nodata
            if nod is not None:
                a[a == nod] = np.nan
            a = a - 273.15
            tx, tn = monthly_txtn_from_hourly(a)
            if tx is not None:
                TX[iy, m - 1], TN[iy, m - 1] = tx, tn
    np.savez_compressed(CACHE_NPZ, TX=TX, TN=TN, years=np.array(years), lat=lat, lon=lon)
    return TX, TN, lat, lon


def block_mean_to_grid(arr, tr, lat_t, lon_t):
    """Average the cells of `arr` (north-up raster with affine tr) whose centres fall inside each target cell
    (target = regular grid with centres lat_t (descending) / lon_t (ascending))."""
    res_t_y = abs(lat_t[1] - lat_t[0])
    res_t_x = abs(lon_t[1] - lon_t[0])
    top_t = lat_t[0] + res_t_y / 2
    left_t = lon_t[0] - res_t_x / 2
    rr, cc = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij")
    lat_c = tr.f + tr.e * (rr + 0.5)
    lon_c = tr.c + tr.a * (cc + 0.5)
    ti = np.floor((top_t - lat_c) / res_t_y).astype(int)
    tj = np.floor((lon_c - left_t) / res_t_x).astype(int)
    inside = (ti >= 0) & (ti < len(lat_t)) & (tj >= 0) & (tj < len(lon_t))
    fin = np.isfinite(arr) & inside
    flat = ti[fin] * len(lon_t) + tj[fin]
    n = len(lat_t) * len(lon_t)
    sums = np.bincount(flat, weights=arr[fin], minlength=n)
    cnt = np.bincount(flat, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    return out.reshape(len(lat_t), len(lon_t)).astype("float32")


def chirps_annual_on(lat_t, lon_t, y0, y1):
    """Annual CHIRPS sums averaged onto the target grid (block mean); years with all 12 months only."""
    import rasterio
    out, years = [], []
    for y in range(y0, y1 + 1):
        tot, ok, tr = None, True, None
        for m in range(1, 13):
            f = os.path.join(G.CHIRPS_DIR, f"chirps-v3.0.{y}.{m:02d}_NordALG.tif")
            if not os.path.exists(f):
                ok = False
                break
            with rasterio.open(f) as r:
                a = r.read(1).astype("float32")
                a[(a == r.nodata) | (a < 0)] = np.nan
                tr = r.transform
            tot = a if tot is None else tot + a
        if not ok:
            continue
        out.append(block_mean_to_grid(tot, tr, lat_t, lon_t))
        years.append(y)
    return np.stack(out), years


def tc_on(lat_t, lon_t, tc_lat, tc_lon, arr):
    ri = np.abs(tc_lat[:, None] - lat_t[None, :]).argmin(axis=0)
    ci = np.abs(tc_lon[:, None] - lon_t[None, :]).argmin(axis=0)
    return arr[np.ix_(ri, ci)]


def main():
    import pandas as pd
    years = list(range(G.Y_ALL[0], G.Y_ALL[1] + 1))

    # ---- ERA5 monthly Tmax/Tmin ----
    if os.path.exists(CACHE_NPZ):
        z = np.load(CACHE_NPZ)
        TX, TN, lat, lon = z["TX"], z["TN"], z["lat"], z["lon"]
        print("ERA5 monthly cache loaded.")
    else:
        print("Building ERA5 monthly Tmax/Tmin from hourly files (first run) ...")
        TX, TN, lat, lon = build_era5_monthly(years)

    # ---- TerraClimate on the ERA5 grid ----
    print("Reading TerraClimate ...")
    d_all, tc_lat, tc_lon = G.build_tc(G.Y_ALL[0], G.Y_ALL[1], {})
    # ---- CHIRPS annual on the ERA5 grid ----
    print("Reading CHIRPS (548 files) ...")
    chp_ann, chp_years = chirps_annual_on(lat, lon, G.Y_ALL[0], G.Y_ALL[1])

    periods = {"P1": G.P1, "P2": G.P2, "ALL": G.Y_ALL}
    clim = {}
    for pn, (a, b) in periods.items():
        dd = {y: v for y, v in d_all.items() if a <= y <= b}
        Ptc, Mtc, mtc = G.climatology(dd)
        sel = [i for i, y in enumerate(chp_years) if a <= y <= b]
        need = MIN_YEAR_FRACTION * (b - a + 1)
        Pch = np.nanmean(chp_ann[sel], axis=0) if len(sel) >= need else np.full(lat.shape + lon.shape, np.nan)
        Mer, mer = t_climatology(TX, TN, years, a, b)
        clim[pn] = dict(P_TC=tc_on(lat, lon, tc_lat, tc_lon, Ptc), M_TC=tc_on(lat, lon, tc_lat, tc_lon, Mtc),
                        m_TC=tc_on(lat, lon, tc_lat, tc_lon, mtc), P_CH=Pch, M_ER=Mer, m_ER=mer)

    variants = {"A_TCprec_TCtemp": ("P_TC", "TC"), "B_CHIRPS_ERA5": ("P_CH", "ER"),
                "C_CHIRPS_TCtemp": ("P_CH", "TC"), "D_TCprec_ERA5": ("P_TC", "ER")}
    LAT = np.repeat(lat[:, None], len(lon), axis=1)
    W = np.cos(np.deg2rad(LAT))

    Q = {}
    for vn, (pk, tk) in variants.items():
        for pn in ("P1", "P2"):
            c = clim[pn]
            Q[(vn, pn)] = G.q2_from(c[pk].copy(), c["M_" + tk], c["m_" + tk])
    # common valid mask: all variants finite
    valid = np.ones(Q[("A_TCprec_TCtemp", "P1")].shape, bool)
    for k in Q:
        valid &= np.isfinite(Q[k])
    txt = [f"Ensemble on ERA5-Land grid (0.1 deg); P1={G.P1}, P2={G.P2}; common valid cells={int(valid.sum())}", ""]
    SH, DQ = {}, {}
    for vn in variants:
        q1, q2 = Q[(vn, "P1")], Q[(vn, "P2")]
        s1, s2 = G.stage_index(q1), G.stage_index(q2)
        SH[vn] = s2 - s1
        DQ[vn] = 100 * (q2 - q1) / q1
        ww = W[valid].sum()
        txt.append(f"== {vn}")
        txt.append(f"  mean dQ2 {np.nansum(DQ[vn][valid]*W[valid])/ww:+.1f}%   area changed stage {100*W[valid&(SH[vn]!=0)].sum()/ww:.1f}%  "
                   f"(drier {100*W[valid&(SH[vn]<0)].sum()/ww:.1f}%, wetter {100*W[valid&(SH[vn]>0)].sum()/ww:.1f}%)")
        for k, n in enumerate(G.STAGE_NAMES):
            a1 = 100 * W[valid & (s1 == k)].sum() / ww
            a2 = 100 * W[valid & (s2 == k)].sum() / ww
            txt.append(f"    {n:11s} P1 {a1:5.1f}%  P2 {a2:5.1f}%")
        G.write_tif(os.path.join(OUT, f"ens_dQ2_pct_{vn}.tif"), np.where(valid, DQ[vn], np.nan), lat, lon)
    # agreement
    dq_stack = np.stack([DQ[v] for v in variants])
    drier_n = np.sum(dq_stack < 0, axis=0)
    ww = W[valid].sum()
    txt.append("")
    txt.append("Agreement among the 4 variants (area %):")
    for k in range(5):
        txt.append(f"  {k} of 4 variants show decreasing Q2: {100*W[valid&(drier_n==k)].sum()/ww:5.1f}%")
    mean_dq = np.nanmean(dq_stack, axis=0)
    sd_dq = np.nanstd(dq_stack, axis=0)
    txt.append(f"Ensemble mean dQ2 {np.nansum(mean_dq[valid]*W[valid])/ww:+.1f}%; between-variant SD (mean) {np.nansum(sd_dq[valid]*W[valid])/ww:.1f} points")
    G.write_tif(os.path.join(OUT, "ens_dQ2_pct_mean.tif"), np.where(valid, mean_dq, np.nan), lat, lon)
    G.write_tif(os.path.join(OUT, "ens_n_variants_drier.tif"), np.where(valid, drier_n.astype("float32"), np.nan), lat, lon)

    # ---- save ensemble arrays for the wilaya / climagram script ----
    Qm1 = np.nanmean(np.stack([Q[(v, "P1")] for v in variants]), axis=0)
    Qm2 = np.nanmean(np.stack([Q[(v, "P2")] for v in variants]), axis=0)
    mm1 = np.nanmean(np.stack([clim["P1"]["m_TC"], clim["P1"]["m_ER"]]), axis=0)
    mm2 = np.nanmean(np.stack([clim["P2"]["m_TC"], clim["P2"]["m_ER"]]), axis=0)
    np.savez_compressed(os.path.join(OUT, "ensemble_arrays.npz"), lat=lat, lon=lon, valid=valid,
                        Q1=Qm1, Q2=Qm2, m1=mm1, m2=mm2, drier_n=drier_n, mean_dq=mean_dq,
                        **{f"Q1_{v}": Q[(v, "P1")] for v in variants}, **{f"Q2_{v}": Q[(v, "P2")] for v in variants})
    G.write_tif(os.path.join(OUT, "ens_Q2_P1_mean.tif"), np.where(valid, Qm1, np.nan), lat, lon)
    G.write_tif(os.path.join(OUT, "ens_Q2_P2_mean.tif"), np.where(valid, Qm2, np.nan), lat, lon)

    # temperature product comparison (ERA5 vs TerraClimate), P1->P2 change too
    txt.append("")
    txt.append("ERA5-Land minus TerraClimate (area mean, period ALL): "
               f"M {np.nansum((clim['ALL']['M_ER']-clim['ALL']['M_TC'])[valid]*W[valid])/ww:+.2f} C, "
               f"m {np.nansum((clim['ALL']['m_ER']-clim['ALL']['m_TC'])[valid]*W[valid])/ww:+.2f} C")
    for nm, (a, b) in (("M", ("M_ER", "M_TC")), ("m", ("m_ER", "m_TC"))):
        d_er = clim["P2"][a] - clim["P1"][a]
        d_tc = clim["P2"][b] - clim["P1"][b]
        txt.append(f"  warming P2-P1 of {nm}:  ERA5 {np.nansum(d_er[valid]*W[valid])/ww:+.2f} C   TerraClimate {np.nansum(d_tc[valid]*W[valid])/ww:+.2f} C")
    txt.append("")

    # stations
    if os.path.exists(G.STATION_CSV):
        st = pd.read_csv(G.STATION_CSV)
        st = st[st["RECENT_1985_2025__Q2"].notna()]
        rows = []
        for _, r in st.iterrows():
            la, lo = float(r["lat"]), float(r["lon"])
            rows.append(dict(id=r["id"], name=r["name"], M_station=r["RECENT_1985_2025__M"], m_station=r["RECENT_1985_2025__m"],
                             M_ERA5=G.nearest(clim["ALL"]["M_ER"], lat, lon, la, lo), M_TC=G.nearest(clim["ALL"]["M_TC"], lat, lon, la, lo),
                             m_ERA5=G.nearest(clim["ALL"]["m_ER"], lat, lon, la, lo), m_TC=G.nearest(clim["ALL"]["m_TC"], lat, lon, la, lo)))
        df = pd.DataFrame(rows)
        df.round(2).to_csv(os.path.join(OUT, "station_vs_ERA5_temperature.csv"), index=False)
        txt.append("Stations (n=%d), bias grid minus station / RMSE (C):" % len(df))
        for v in ("M", "m"):
            for p in ("ERA5", "TC"):
                d = (df[f"{v}_{p}"] - df[f"{v}_station"]).dropna()
                txt.append(f"  {v} {p:5s}: bias {d.mean():+.2f}  RMSE {np.sqrt((d**2).mean()):.2f}  (n={len(d)})")
    open(os.path.join(OUT, "ensemble_shift_summary.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))

    # maps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ext = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]
        fig, axs = plt.subplots(2, 2, figsize=(13, 7))
        for ax, vn in zip(axs.ravel(), variants):
            im = ax.imshow(np.where(valid, DQ[vn], np.nan), extent=ext, cmap="RdBu", vmin=-40, vmax=40)
            ax.set_title(vn)
        fig.colorbar(im, ax=axs, label="dQ2 (%) P2 vs P1", shrink=0.8)
        fig.savefig(os.path.join(OUT, "map_ensemble_dQ2.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(np.where(valid, drier_n, np.nan), extent=ext, cmap="viridis", vmin=0, vmax=4)
        plt.colorbar(im, label="number of variants (of 4) with decreasing Q2")
        fig.savefig(os.path.join(OUT, "map_ensemble_agreement.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("maps failed:", e)
    print("\nDONE. Results in", OUT)


if __name__ == "__main__":
    main()
