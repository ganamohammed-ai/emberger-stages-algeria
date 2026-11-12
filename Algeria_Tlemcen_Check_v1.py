#!/usr/bin/env python3
"""
Regional check: Tlemcen wilaya, TerraClimate (4 km), compared with the Tlemcen studies of the literature.
Run by double-clicking RUN_Tlemcen_Check_v1.bat.  Needs Algeria_Robustness_v4.py, Algeria_Windows_Transitions_v3.py,
Algeria_Grid_Q2_Shift.py and Algeria_Wilaya_Zones.py in the same folder (and the DEM array results_grid\\mask_dem\\dem_on_grid.npy if present).

Compares climatological Q2 (P, M, m) between
   OP 1985-1999 and NP 2004-2018   (the periods of Siba et al.; Siba and Aboura)
   W0 1961-1990 and W1 1991-2020   (our main pair)
for (a) all Tlemcen-wilaya cells, (b) elevation bands, (c) the cells nearest to Ghazaouet, Maghnia, Beni-Saf (approximate coordinates).
Output: results_grid\\tlemcen\\tlemcen_summary.txt, tlemcen_cells.csv, tlemcen_points.csv, tlemcen_bands.csv, run_log.txt/error_log.txt
"""
import os, sys, traceback
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Robustness_v4 as R
import Algeria_Windows_Transitions_v3 as T
import Algeria_Grid_Q2_Shift as G
import Algeria_Wilaya_Zones as Z

OUTD = os.path.join(G.OUT, "tlemcen")
PAIRS = {"OP 1985-1999 -> NP 2004-2018": ((1985, 1999), (2004, 2018)),
         "W0 1961-1990 -> W1 1991-2020": ((1961, 1990), (1991, 2020))}
POINTS = {"Ghazaouet (approx 35.10N, 1.86W)": (35.10, -1.86),
          "Maghnia (approx 34.85N, 1.73W)": (34.85, -1.73),
          "Beni-Saf (approx 35.30N, 1.38W)": (35.30, -1.38)}
BANDS = [(-1e9, 300), (300, 600), (600, 900), (900, 1e9)]
LIMS = [20, 40, 70, 100]


def tlemcen_mask(lat, lon):
    import geopandas as gpd
    com = gpd.read_file(Z.SHP).to_crs(4326)
    com["geometry"] = com.geometry.simplify(0.003, preserve_topology=True).make_valid()
    wil = com.dissolve(by="name_fr_1", as_index=False)[["name_fr_1", "geometry"]]
    wil = wil[[Z.norm(w) == Z.norm("Tlemcen") for w in wil["name_fr_1"]]]
    if len(wil) == 0:
        raise RuntimeError("Wilaya Tlemcen not found in the shapefile")
    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoDataFrame({"k": np.arange(LON.size)}, geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326)
    j = gpd.sjoin(pts, wil, predicate="within", how="inner")
    m = np.zeros(LON.size, bool)
    m[j["k"].values] = True
    return m.reshape(LON.shape)


def stat(Q0, Q1, P0, P1, M0, M1, m0, m1, area, mk):
    ok = mk & np.isfinite(Q0) & np.isfinite(Q1)
    if not ok.any():
        return {}
    w = area[ok]
    wm = lambda x: float(np.sum(x[ok] * w) / w.sum())
    s0, s1 = R.stage_with(Q0, LIMS), R.stage_with(Q1, LIMS)
    dry = float(w[(s1 < s0)[ok]].sum() / w.sum() * 100)
    wet = float(w[(s1 > s0)[ok]].sum() / w.sum() * 100)
    q0, q1 = wm(Q0), wm(Q1)
    return dict(n_cells=int(ok.sum()), area_km2=round(float(w.sum()), 0),
                P0=round(wm(P0), 1), P1=round(wm(P1), 1), dP_pct=round((wm(P1) / wm(P0) - 1) * 100, 1),
                M0=round(wm(M0), 2), M1=round(wm(M1), 2), m0=round(wm(m0), 2), m1=round(wm(m1), 2),
                Q2_0=round(q0, 1), Q2_1=round(q1, 1), dQ2_pct=round((q1 / q0 - 1) * 100, 1),
                pct_area_drier=round(dry, 1), pct_area_wetter=round(wet, 1))


def main():
    os.makedirs(OUTD, exist_ok=True)
    per_year, lat, lon, area, lab, masks = R.prepare()
    tl = tlemcen_mask(lat, lon)
    print("Tlemcen cells:", int(tl.sum()))
    dem = None
    f = os.path.join(G.OUT, "mask_dem", "dem_on_grid.npy")
    if os.path.exists(f):
        d = np.load(f)
        if d.shape == (len(lat), len(lon)):
            dem = d
    rows, prow, brow = [], [], []
    for tag, (p0, p1) in PAIRS.items():
        f0, f1 = T.window_fields(per_year, *p0), T.window_fields(per_year, *p1)
        if f0 is None or f1 is None:
            print("missing years for", tag); continue
        args = (f0["Q2"], f1["Q2"], f0["P"], f1["P"], f0["M"], f1["M"], f0["m"], f1["m"], area)
        r = dict(pair=tag, scope="Tlemcen wilaya (all cells)"); r.update(stat(*args, tl)); rows.append(r)
        if dem is not None:
            for lo, hi in BANDS:
                mk = tl & np.isfinite(dem) & (dem >= lo) & (dem < hi)
                if mk.sum() < 3: continue
                r = dict(pair=tag, scope=f"elevation {max(lo,0):.0f}-{'max' if hi>1e8 else int(hi)} m"); r.update(stat(*args, mk)); brow.append(r)
        for nm, (la, lo_) in POINTS.items():
            i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo_)))
            mk = np.zeros_like(tl); mk[i, j] = True
            r = dict(pair=tag, scope=nm, cell_lat=round(float(lat[i]), 3), cell_lon=round(float(lon[j]), 3),
                     elev_m=(round(float(dem[i, j]), 0) if dem is not None and np.isfinite(dem[i, j]) else None))
            r.update(stat(*args, mk)); prow.append(r)
        # per-cell table for the first pair only needs once: save for both
        ok = tl & np.isfinite(f0["Q2"]) & np.isfinite(f1["Q2"])
        ii, jj = np.where(ok)
        pd.DataFrame(dict(lat=lat[ii], lon=lon[jj], elev=(dem[ii, jj] if dem is not None else np.nan),
                          Q2_0=f0["Q2"][ii, jj], Q2_1=f1["Q2"][ii, jj])).to_csv(
            os.path.join(OUTD, "tlemcen_cells_%s.csv" % tag.split()[0]), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUTD, "tlemcen_summary.csv"), index=False)
    pd.DataFrame(prow).to_csv(os.path.join(OUTD, "tlemcen_points.csv"), index=False)
    pd.DataFrame(brow).to_csv(os.path.join(OUTD, "tlemcen_bands.csv"), index=False)
    L = ["Tlemcen check, TerraClimate, climatological Q2, stage limits %s" % LIMS,
         "Literature: Siba et al. report a wetter bioclimate 1985-1999 -> 2004-2018; Siba & Aboura: Ghazaouet Q2 46.3 -> 49.7, Sidi Djilali 40.8 -> 44.0;",
         "Belkhodja et al. (1913-1938 vs 1995-2020): Maghnia Q2 50.0 -> 33.6, Beni-Saf 62.9 -> 74.3 (different periods, other data).", ""]
    for name, tab in (("WILAYA", rows), ("ELEVATION BANDS", brow), ("POINTS", prow)):
        L.append("== " + name)
        for r in tab:
            L.append("  " + "; ".join(f"{k}={v}" for k, v in r.items()))
        L.append("")
    open(os.path.join(OUTD, "tlemcen_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    try:
        os.makedirs(OUTD, exist_ok=True)
        main()
    except Exception:
        open(os.path.join(OUTD, "error_log.txt"), "w", encoding="utf-8").write(traceback.format_exc())
        raise
