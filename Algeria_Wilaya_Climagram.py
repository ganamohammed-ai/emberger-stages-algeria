#!/usr/bin/env python3
"""
Wilaya-level statistics of the bioclimatic-stage shift and Emberger climagram.

Input : results_grid\\ensemble_arrays.npz   (made by Algeria_Ensemble_Q2_Shift.py, new version)
        D:\\DATA ALG_CLIP\\0_Decoupage\\Nord_Comm_Wilaya.shp  (communes with wilaya name in 'name_fr_1')
        results\\station_pairs_shift.csv  (optional, old vs modern stations)
Output: results_grid\\wilaya_stage_shift.csv, wilaya_summary.txt, map_wilaya_dQ2.png, climagram_Emberger.png
Unit of analysis: ERA5-Land 0.1 degree cells (about 120 km2) whose centre lies inside the wilaya;
wilayas with few cells are flagged (n_cells < 8).  Q2 = ensemble mean of the 4 variants.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Grid_Q2_Shift as G

NPZ = os.path.join(G.OUT, "ensemble_arrays.npz")
SHP = r"D:\DATA ALG_CLIP\0_Decoupage\Nord_Comm_Wilaya.shp"
PAIRS = os.path.join(HERE, "results", "station_pairs_shift.csv")
MIN_CELLS = 8
KM_PER_DEG = 111.32


def summarize(df):
    """df: one row per cell with columns wilaya, lat, Q1, Q2, m1, m2, drier_n, area_km2."""
    rows = []
    for w, g in df.groupby("wilaya"):
        A = g["area_km2"].values
        s1 = G.stage_index(g["Q1"].values)
        s2 = G.stage_index(g["Q2"].values)
        tot = A.sum()
        r = dict(wilaya=w, n_cells=len(g), area_km2=round(tot, 0), small_sample=len(g) < MIN_CELLS,
                 Q2_P1=round(np.average(g["Q1"], weights=A), 1), Q2_P2=round(np.average(g["Q2"], weights=A), 1),
                 m_P1=round(np.average(g["m1"], weights=A), 2), m_P2=round(np.average(g["m2"], weights=A), 2))
        r["dQ2_pct"] = round(100 * (r["Q2_P2"] - r["Q2_P1"]) / r["Q2_P1"], 1)
        r["stage_of_mean_P1"] = G.STAGE_NAMES[int(G.stage_index(np.array([r["Q2_P1"]]))[0])]
        r["stage_of_mean_P2"] = G.STAGE_NAMES[int(G.stage_index(np.array([r["Q2_P2"]]))[0])]
        r["pct_area_changed_stage"] = round(100 * A[s2 != s1].sum() / tot, 1)
        r["pct_area_to_drier"] = round(100 * A[s2 < s1].sum() / tot, 1)
        r["pct_area_to_wetter"] = round(100 * A[s2 > s1].sum() / tot, 1)
        r["pct_area_>=3of4_variants_drier"] = round(100 * A[g["drier_n"].values >= 3].sum() / tot, 1)
        for k, n in enumerate(G.STAGE_NAMES):
            r[f"km2_{n}_P1"] = round(A[s1 == k].sum(), 0)
            r[f"km2_{n}_P2"] = round(A[s2 == k].sum(), 0)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(NPZ):
        print("ensemble_arrays.npz not found: run RUN_Ensemble_Q2_Shift.bat (new version) first.")
        sys.exit(1)
    z = np.load(NPZ)
    lat, lon, valid = z["lat"], z["lon"], z["valid"]
    LON, LAT = np.meshgrid(lon, lat)
    ii, jj = np.where(valid)
    cells = pd.DataFrame(dict(i=ii, j=jj, lon=LON[ii, jj], lat=LAT[ii, jj], Q1=z["Q1"][ii, jj], Q2=z["Q2"][ii, jj],
                              m1=z["m1"][ii, jj], m2=z["m2"][ii, jj], drier_n=z["drier_n"][ii, jj]))
    cells["area_km2"] = (KM_PER_DEG * 0.1) ** 2 * np.cos(np.deg2rad(cells["lat"]))
    cells = cells[np.isfinite(cells["Q1"]) & np.isfinite(cells["Q2"])]

    print("Reading communes and dissolving to wilayas (about 1 minute) ...", flush=True)
    com = gpd.read_file(SHP).to_crs(4326)
    # simplify (tolerance 0.003 deg, about 300 m: harmless for 0.1 deg cells) and repair, to make dissolving fast and safe
    com["geometry"] = com.geometry.simplify(0.003, preserve_topology=True).make_valid()
    com = com[com.geometry.notna() & ~com.geometry.is_empty]
    wil = com.dissolve(by="name_fr_1", as_index=False)[["name_fr_1", "name_ar_1", "geometry"]]
    print(f"  {len(com)} communes -> {len(wil)} wilayas")
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs=4326)
    j = gpd.sjoin(pts, wil, predicate="within", how="inner")
    j = j.rename(columns={"name_fr_1": "wilaya"})
    res = summarize(j)
    ar = wil.set_index("name_fr_1")["name_ar_1"].to_dict()
    res.insert(1, "wilaya_ar", res["wilaya"].map(ar))
    res = res.sort_values("dQ2_pct")
    res.to_csv(os.path.join(G.OUT, "wilaya_stage_shift.csv"), index=False, encoding="utf-8-sig")

    # region-wide line
    allr = summarize(cells.assign(wilaya="NORTH_ALL"))
    tot = allr.iloc[0]
    lines = [f"Wilaya statistics, ensemble Q2 (4 variants), P1={G.P1}, P2={G.P2}; {len(res)} wilayas with >=1 cell centre", "",
             f"NORTH domain: Q2 {tot['Q2_P1']} -> {tot['Q2_P2']} ({tot['dQ2_pct']:+.1f}%), area changed stage {tot['pct_area_changed_stage']}% "
             f"(drier {tot['pct_area_to_drier']}%, wetter {tot['pct_area_to_wetter']}%)", "",
             f"{'wilaya':24s} {'cells':>5s} {'Q2 P1':>7s} {'Q2 P2':>7s} {'dQ2%':>6s} {'chg%':>5s} {'dry%':>5s} {'>=3/4':>6s}  stage P1 -> P2"]
    for _, r in res.iterrows():
        flag = "*" if r["small_sample"] else " "
        lines.append(f"{r['wilaya'][:23]:23s}{flag} {r['n_cells']:5d} {r['Q2_P1']:7.1f} {r['Q2_P2']:7.1f} {r['dQ2_pct']:6.1f} "
                     f"{r['pct_area_changed_stage']:5.1f} {r['pct_area_to_drier']:5.1f} {r['pct_area_>=3of4_variants_drier']:6.1f}  "
                     f"{r['stage_of_mean_P1']} -> {r['stage_of_mean_P2']}")
    lines += ["", "* fewer than %d cells: treat as indicative only." % MIN_CELLS,
              "chg% = % of area whose stage changed; dry% = % of area moving to a drier stage; >=3/4 = % of area where >=3 of the 4 variants show decreasing Q2."]
    open(os.path.join(G.OUT, "wilaya_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

    # choropleth map
    m = wil.merge(res[["wilaya", "dQ2_pct"]], left_on="name_fr_1", right_on="wilaya", how="left")
    fig, ax = plt.subplots(figsize=(11, 6))
    m.plot(column="dQ2_pct", cmap="RdBu", vmin=-30, vmax=30, legend=True, ax=ax,
           missing_kwds=dict(color="#eeeeee"), edgecolor="#555555", linewidth=0.3,
           legend_kwds=dict(label="Change in Q2 (%), P2 vs P1"))
    ax.set_xlim(-2.4, 9.2)
    ax.set_ylim(30.6, 37.2)
    ax.set_title("Change in Emberger Q2 by wilaya (ensemble of 4 variants)")
    fig.savefig(os.path.join(G.OUT, "map_wilaya_dQ2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Emberger climagram
    fig, ax = plt.subplots(figsize=(10, 7))
    lims = [lim for lim, _ in G.STAGES[:-1]]
    for lim in lims:
        ax.axhline(lim, color="#999999", lw=0.8)
    xs = ax.get_xlim()
    bounds = [4] + lims + [250]
    for k, n in enumerate(G.STAGE_NAMES):
        ax.text(11.8, np.sqrt(bounds[k] * bounds[k + 1]), n, ha="right", va="center", fontsize=9, color="#444444")
    for v in (0, 3, 7):
        ax.axvline(v, color="#bbbbbb", lw=0.8, ls=":")
    big = res[~res["small_sample"]]
    for _, r in big.iterrows():
        ax.annotate("", xy=(r["m_P2"], r["Q2_P2"]), xytext=(r["m_P1"], r["Q2_P1"]),
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.0, alpha=0.8))
        ax.text(r["m_P2"], r["Q2_P2"], r["wilaya"][:9], fontsize=6, color="#2c7fb8")
    if os.path.exists(PAIRS):
        pr = pd.read_csv(PAIRS)
        for _, r in pr.iterrows():
            ax.annotate("", xy=(r["m_new"], r["Q2_new"]), xytext=(r["m_old"], r["Q2_old"]),
                        arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=1.2,
                                        ls="-" if r["continuous"] else "--"))
            ax.text(r["m_new"], r["Q2_new"], str(r["new_name"])[:10], fontsize=6, color="#d95f0e")
    ax.set_yscale("log")
    ax.set_ylim(4, 250)
    ax.set_xlim(-3, 12)
    ax.set_xlabel("m: mean minimum of the coldest month (C)")
    ax.set_ylabel("Emberger Q2 (log scale)")
    ax.set_title("Emberger climagram: blue = wilayas (P1 to P2), orange = stations (1913-1938 to 1985-2025; dashed = different station)")
    fig.savefig(os.path.join(G.OUT, "climagram_Emberger.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nDONE. Results in", G.OUT)


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
    os.makedirs(G.OUT, exist_ok=True)
    _log = open(os.path.join(G.OUT, "run_log.txt"), "w", encoding="utf-8")
    _log.write("script started\n")
    _log.flush()
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        os.makedirs(G.OUT, exist_ok=True)
        open(os.path.join(G.OUT, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\error_log.txt - send it to Claude.")
        sys.exit(1)
