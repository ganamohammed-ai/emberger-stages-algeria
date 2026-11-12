#!/usr/bin/env python3
"""
Fixed natural mask for the north and the role of elevation (DEM), TerraClimate native grid.

Run by double-clicking RUN_Mask_DEM_v7.bat.  Needs Algeria_Isoline_v5.py, Algeria_Robustness_v4.py,
Algeria_Windows_Transitions_v3.py, Algeria_Grid_Q2_Shift.py and Algeria_Wilaya_Zones.py in the same folder, and
the DEM  D:\\DATA ALG_CLIP\\04_Relief_DEM\\DEM_Mosaic_NordALG.tif  (resampled to the TerraClimate grid by averaging).

Outputs in results_grid\\mask_dem\\ :
  dem_on_grid.npy                 mean elevation (m) of each TerraClimate cell (kept so the DEM is read only once)
  mask_comparison.csv             wilaya mask (used so far) versus fixed natural masks (long-term mean P >= 100 / 150 mm,
                                  1961-2020, the same cells for every period): area, change of mean Q2, % area drier / wetter
  elevation_bands.csv             the same quantities by elevation band, inside the wilaya mask
  isoline_elevation.csv           mean elevation of the Q2 = 20, 40, 70 isolines in the two periods of each pair
                                  (a rise means the line moved upslope as well as northward)
  gradients.txt                   Q2 regressed on latitude and elevation (W0), and the equivalent northward / upslope shift
  mask_dem_summary.txt, fig_mask_dem.png

Why: (1) the wilaya mask keeps Saharan parts of northern wilayas, so a mask fixed for all periods and defined by climate
(inside Algerian territory) is a safer reference; (2) in mountains a drying can show as a line moving upslope instead of
northward, which the isoline speed alone cannot separate.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Isoline_v5 as I
import Algeria_Robustness_v4 as R
import Algeria_Windows_Transitions_v3 as T
import Algeria_Grid_Q2_Shift as G

DEM_PATH = r"D:\DATA ALG_CLIP\04_Relief_DEM\DEM_Mosaic_NordALG.tif"
OUTM = os.path.join(G.OUT, "mask_dem")
NAT_P_LIMITS = (100.0, 150.0)                 # mm; fixed mask = long-term mean annual P at or above this (per-arid limit of Daget 1977 is 100 mm)
NAT_YEARS = (1961, 2020)
ELEV_BANDS = [(-1e9, 200), (200, 500), (500, 800), (800, 1100), (1100, 1500), (1500, 1e9)]
SMOOTH = 2
LEVELS = (20, 40, 70)


# ------------------------------------------------------------------ DEM
def dem_on_grid(path, lat, lon):
    """Mean elevation of the DEM inside each grid cell; lat ascending; returns array (nlat, nlon) with NaN where no DEM."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    top = float(lat.max()) + dlat / 2.0
    left = float(lon.min()) - dlon / 2.0
    tr = from_origin(left, top, dlon, dlat)
    with rasterio.open(path) as src:
        nod = src.nodata
        with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=len(lon), height=len(lat),
                       resampling=Resampling.average, src_nodata=nod, nodata=-9999.0) as vrt:
            a = vrt.read(1).astype("float64")
    a[(a <= -9000) | ~np.isfinite(a)] = np.nan
    return a[::-1, :]                          # north-up raster -> lat ascending


def get_dem(lat, lon):
    f = os.path.join(OUTM, "dem_on_grid.npy")
    if os.path.exists(f):
        d = np.load(f)
        if d.shape == (len(lat), len(lon)):
            print("DEM on grid read from", f)
            return d
    print("Resampling the DEM to the TerraClimate grid (a few minutes) ...", flush=True)
    d = dem_on_grid(DEM_PATH, lat, lon)
    np.save(f, d)
    return d


# ------------------------------------------------------------------ pure functions
def area_mean(x, area, mask):
    ok = mask & np.isfinite(x)
    return float(np.sum(x[ok] * area[ok]) / area[ok].sum()) if ok.any() else np.nan


def change_stats(Q0, Q1, area, mask, limits):
    """Mean Q2 in the two periods, % change, % area drier / wetter, for cells valid in both."""
    both = mask & np.isfinite(Q0) & np.isfinite(Q1)
    if not both.any():
        return dict(area_km2=0.0, Q2_t0=np.nan, Q2_t1=np.nan, dQ2_pct=np.nan, drier_pct=np.nan, wetter_pct=np.nan)
    q0, q1 = area_mean(Q0, area, both), area_mean(Q1, area, both)
    dry, wet, _ = R.drier_wetter(Q0, Q1, limits, area, both)
    return dict(area_km2=round(float(area[both].sum()), 0), Q2_t0=round(q0, 1), Q2_t1=round(q1, 1),
                dQ2_pct=round(100 * (q1 - q0) / q0, 1), drier_pct=round(dry, 1), wetter_pct=round(wet, 1))


def contour_elevation(Q, lat, lon, level, mask, dem):
    """Length-weighted mean elevation (m) along the contour Q = level inside the mask; NaN if none."""
    import contourpy
    z = np.ma.masked_invalid(np.where(mask, Q, np.nan))
    gen = contourpy.contour_generator(lon, lat, z)
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    num = den = 0.0
    for line in gen.lines(level):
        if len(line) < 2:
            continue
        a, b = line[:-1], line[1:]
        mlon, mlat = 0.5 * (a[:, 0] + b[:, 0]), 0.5 * (a[:, 1] + b[:, 1])
        seg = np.hypot((b[:, 0] - a[:, 0]) * 111.32 * np.cos(np.deg2rad(mlat)), (b[:, 1] - a[:, 1]) * 111.32)
        i = np.clip(np.rint((mlat - lat[0]) / dlat).astype(int), 0, len(lat) - 1)
        j = np.clip(np.rint((mlon - lon[0]) / dlon).astype(int), 0, len(lon) - 1)
        e = dem[i, j]
        ok = np.isfinite(e)
        num += float(np.sum(e[ok] * seg[ok]))
        den += float(seg[ok].sum())
    return num / den if den > 0 else np.nan


def gradients(Q, lat2d, lon2d, dem, mask):
    """OLS of Q on latitude (per km north), longitude (per degree) and elevation (per 100 m)."""
    ok = mask & np.isfinite(Q) & np.isfinite(dem)
    if ok.sum() < 50:
        return None
    y = Q[ok]
    X = np.column_stack([np.ones(ok.sum()), (lat2d[ok] - lat2d[ok].mean()) * 111.32,
                         lon2d[ok] - lon2d[ok].mean(), dem[ok] / 100.0])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(per_km_north=beta[1], per_deg_east=beta[2], per_100m=beta[3], r2=r2, n=int(ok.sum()))


# ------------------------------------------------------------------ main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTM, exist_ok=True)
    per_year, lat, lon, area, lab, masks = R.prepare()
    nord = masks["NORD sans Sahara"]
    LON, LAT = np.meshgrid(lon, lat)
    dem = get_dem(lat, lon)
    print(f"DEM on grid: {np.isfinite(dem).sum()} cells, {np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m")

    # fixed natural masks from the long-term mean precipitation
    ys = [y for y in range(NAT_YEARS[0], NAT_YEARS[1] + 1) if y in per_year]
    Pmean = np.nanmean(np.stack([per_year[y]["P"] for y in ys]), axis=0)
    inside = np.asarray(lab != "")            # cells inside an Algerian wilaya (the grid box also holds Morocco and Tunisia)
    print(f"cells inside Algerian wilayas: {inside.sum()} of {inside.size}")
    nat = {f"natural mask P>={int(p)} mm": inside & np.isfinite(Pmean) & (Pmean >= p) for p in NAT_P_LIMITS}
    allmasks = {"wilaya mask (north without Sahara)": nord}
    allmasks.update(nat)
    allmasks["wilaya mask AND P>=100 mm"] = nord & (Pmean >= 100.0)
    allmasks["Algeria, any wilaya, all P (reference)"] = inside

    fields = {}
    for tag, p0, p1 in I.PAIRS:
        f0, f1 = T.window_fields(per_year, *p0), T.window_fields(per_year, *p1)
        if f0 is not None and f1 is not None:
            fields[tag] = (f0["Q2"], f1["Q2"], ((p1[0] + p1[1]) - (p0[0] + p0[1])) / 2.0)
    lim = R.BASE_LIMITS

    # ---------- masks ----------
    rows = []
    for mn, mk in allmasks.items():
        for tag, (Q0, Q1, yb) in fields.items():
            r = dict(mask=mn, pair=tag, mask_area_km2=round(float(area[mk].sum()), 0))
            r.update(change_stats(Q0, Q1, area, mk, lim))
            rows.append(r)
    MC = pd.DataFrame(rows)
    MC.to_csv(os.path.join(OUTM, "mask_comparison.csv"), index=False, encoding="utf-8-sig")
    inter = {mn: float(area[mk & nord].sum()) for mn, mk in allmasks.items()}
    L = ["Fixed natural mask versus wilaya mask (TerraClimate, base stage limits %s)" % lim, "",
         f"Natural masks: long-term mean annual P {NAT_YEARS[0]}-{NAT_YEARS[1]} at or above the stated value (same cells in all periods).", "",
         "Area of each mask and of its overlap with the wilaya mask (km2):"]
    for mn, mk in allmasks.items():
        L.append(f"  {mn:40s} area {area[mk].sum():10.0f}   in wilaya mask {inter[mn]:10.0f}   outside {area[mk & ~nord].sum():9.0f}")
    L += ["", "Change by mask and period pair:"]
    L.append(MC.drop(columns=["mask_area_km2"]).to_string(index=False))

    # ---------- elevation bands ----------
    rows = []
    for tag, (Q0, Q1, yb) in fields.items():
        for lo, hi in ELEV_BANDS:
            mk = nord & np.isfinite(dem) & (dem >= lo) & (dem < hi)
            lab_b = f"<{hi:.0f}" if lo < -1e8 else (f">={lo:.0f}" if hi > 1e8 else f"{lo:.0f}-{hi:.0f}")
            r = dict(pair=tag, elevation_m=lab_b)
            r.update(change_stats(Q0, Q1, area, mk, lim))
            rows.append(r)
    EB = pd.DataFrame(rows)
    EB.to_csv(os.path.join(OUTM, "elevation_bands.csv"), index=False, encoding="utf-8-sig")
    L += ["", "Change by elevation band inside the wilaya mask:"]
    for tag in fields:
        L.append(f"  {tag}")
        L.append(EB[EB["pair"] == tag].drop(columns=["pair"]).to_string(index=False))
    for tag, (Q0, Q1, yb) in fields.items():
        ok = nord & np.isfinite(dem) & np.isfinite(Q0) & np.isfinite(Q1)
        d = 100 * (Q1[ok] - Q0[ok]) / Q0[ok]
        if ok.sum() > 10 and np.std(d) > 0:
            rho = pd.Series(dem[ok]).corr(pd.Series(d), method="spearman")
            L.append(f"  Spearman correlation of elevation with the % change of Q2, cell by cell, {tag}: {rho:+.2f}")

    # ---------- isoline elevation ----------
    rows = []
    for tag, (Q0r, Q1r, yb) in fields.items():
        both = nord & np.isfinite(Q0r) & np.isfinite(Q1r)
        Q0 = I.smooth_nan(np.where(both, Q0r, np.nan), SMOOTH)
        Q1 = I.smooth_nan(np.where(both, Q1r, np.nan), SMOOTH)
        for lev in LEVELS:
            e0 = contour_elevation(Q0, lat, lon, lev, both, dem)
            e1 = contour_elevation(Q1, lat, lon, lev, both, dem)
            rows.append(dict(pair=tag, level=lev, mean_elev_t0_m=round(e0, 0), mean_elev_t1_m=round(e1, 0),
                             rise_m=round(e1 - e0, 0), rise_m_per_decade=round((e1 - e0) / (yb / 10.0), 1)))
    IE = pd.DataFrame(rows)
    IE.to_csv(os.path.join(OUTM, "isoline_elevation.csv"), index=False, encoding="utf-8-sig")
    L += ["", "Mean elevation along the isolines (smoothed %d cells); rise = period 2 minus period 1:" % SMOOTH, IE.to_string(index=False)]

    # ---------- gradients ----------
    g_lines = ["Q2 regressed on latitude, longitude and elevation inside the wilaya mask:", ""]
    if "W0->W1" in fields:
        Q0, Q1, yb = fields["W0->W1"]
        for nm, Q in (("1961-1990", Q0), ("1991-2020", Q1)):
            g = gradients(Q, LAT, LON, dem, nord)
            if g:
                g_lines.append(f"  {nm}: Q2 changes by {g['per_km_north']:+.3f} per km northward, {g['per_100m']:+.2f} per 100 m of elevation, "
                               f"{g['per_deg_east']:+.2f} per degree east (R2 {g['r2']:.2f}, n = {g['n']})")
                if nm == "1961-1990":
                    g0 = g
        dQ = area_mean(Q1 - Q0, area, nord & np.isfinite(Q0) & np.isfinite(Q1))
        if g0 and g0["per_km_north"] != 0 and g0["per_100m"] != 0:
            g_lines += ["",
                        f"  Mean change of Q2 between the two windows: {dQ:+.2f} (area-weighted, north).",
                        f"  The same loss equals a displacement of about {abs(dQ / g0['per_km_north']):.0f} km northward, "
                        f"or about {abs(dQ / g0['per_100m']) * 100:.0f} m upslope, in the 1961-1990 gradients.",
                        "  Reading: the northward and the upslope reading are two descriptions of the same change; the isoline elevation table shows",
                        "  which one the lines actually followed (rise_m_per_decade close to 0 = mainly northward)."]
    L += [""] + g_lines
    open(os.path.join(OUTM, "gradients.txt"), "w", encoding="utf-8").write("\n".join(g_lines))
    open(os.path.join(OUTM, "mask_dem_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

    # ---------- figure ----------
    try:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        ext = [lon.min(), lon.max(), lat.min(), lat.max()]
        im = axs[0].imshow(np.ma.masked_invalid(dem), extent=ext, origin="lower", cmap="terrain", vmin=0, vmax=2000)
        axs[0].contour(lon, lat, nord.astype(float), levels=[0.5], colors="k", linewidths=0.8)
        for (nm, mk), col in zip(nat.items(), ("#d62728", "#1f77b4")):
            axs[0].contour(lon, lat, mk.astype(float), levels=[0.5], colors=col, linewidths=0.9)
        axs[0].set_title("Elevation (m); black = wilaya mask; red / blue = P >= 100 / 150 mm", fontsize=9)
        fig.colorbar(im, ax=axs[0], shrink=0.8)
        labs = list(EB[EB["pair"] == next(iter(fields))]["elevation_m"])
        w = 0.8 / max(len(fields), 1)
        for k, tag in enumerate(fields):
            v = EB[EB["pair"] == tag]["dQ2_pct"].values.astype(float)
            axs[1].bar(np.arange(len(labs)) + k * w, v, w, label=tag)
        axs[1].axhline(0, color="k", lw=0.8)
        axs[1].set_xticks(np.arange(len(labs)) + 0.4 - w / 2)
        axs[1].set_xticklabels(labs, fontsize=8)
        axs[1].set_xlabel("elevation band (m)")
        axs[1].set_ylabel("change of mean Q2 (%)")
        axs[1].legend(fontsize=8)
        axs[1].set_title("Change of Q2 by elevation band (wilaya mask)", fontsize=9)
        fig.savefig(os.path.join(OUTM, "fig_mask_dem.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("figure failed:", e)
    print("\nDONE. Results in", OUTM)


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
    os.makedirs(OUTM, exist_ok=True)
    _log = open(os.path.join(OUTM, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        open(os.path.join(OUTM, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\mask_dem\\error_log.txt - send it to Claude.")
        sys.exit(1)
