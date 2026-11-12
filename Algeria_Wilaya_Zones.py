#!/usr/bin/env python3
"""
Wilaya- and zone-level statistics of the bioclimatic-stage shift and Emberger climagram.

Input : results_grid\\ensemble_arrays.npz   (made by Algeria_Ensemble_Q2_Shift.py)
        D:\\DATA ALG_CLIP\\0_Decoupage\\Nord_Comm_Wilaya.shp  (reference layer; wilaya name in 'name_fr_1')
        results\\station_pairs_shift.csv  (optional, old vs modern stations)
Output: results_grid\\wilaya_stage_shift.csv, wilaya_summary.txt, zone_summary.txt, main_results_table.csv,
        map_wilaya_dQ2.png, fig_zone_dQ2.png, climagram_Emberger.png
Unit  : ERA5-Land 0.1 degree cells (about 120 km2) whose centre lies inside the wilaya.
The Sahara wilayas are reported separately and excluded from "NORD sans Sahara" (Q2 is not informative in the desert).
"""
import os, sys, re, unicodedata
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

ZONES = {
    "Littoral": ["tlemcen", "ain temouchent", "oran", "mostaganem", "chlef", "tipaza", "alger", "boumerdes",
                 "tizi ouzou", "bejaia", "jijel", "skikda", "annaba", "el tarf"],
    "Tell interieur": ["sidi bel abbes", "mascara", "relizane", "ain defla", "blida", "medea", "bouira",
                       "tissemsilt", "guelma", "souk ahras", "constantine", "mila"],
    "Hauts Plateaux et Aures": ["tiaret", "saida", "djelfa", "m sila", "bordj bou arreridj", "setif",
                                "oum el bouaghi", "batna", "khenchela", "tebessa"],
    "Atlas saharien et bordure": ["naama", "el bayadh", "laghouat", "biskra", "ouled djellal"],
    "Sahara": ["ouargla", "el oued", "touggourt", "el meniaa", "el menia", "ghardaia", "el mghair", "el m ghair", "el meghaier"],
}
NORD_ZONES = ["Littoral", "Tell interieur", "Hauts Plateaux et Aures", "Atlas saharien et bordure"]
ORDER = NORD_ZONES + ["NORD sans Sahara", "Sahara", "TOUT LE DOMAINE"]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


ZONE_OF = {norm(w): z for z, ws in ZONES.items() for w in ws}


def summarize(df, name_col="wilaya"):
    """df: one row per cell with <name_col>, lat, Q1, Q2, m1, m2, drier_n, area_km2 (+ optional q1_<v>, q2_<v>)."""
    vcols = sorted(c[3:] for c in df.columns if c.startswith("q1_"))
    rows = []
    for w, g in df.groupby(name_col):
        A = g["area_km2"].values
        s1 = G.stage_index(g["Q1"].values)
        s2 = G.stage_index(g["Q2"].values)
        tot = A.sum()
        r = {name_col: w, "n_cells": len(g), "area_km2": round(tot, 0), "small_sample": len(g) < MIN_CELLS,
             "Q2_P1": round(np.average(g["Q1"], weights=A), 1), "Q2_P2": round(np.average(g["Q2"], weights=A), 1),
             "m_P1": round(np.average(g["m1"], weights=A), 2), "m_P2": round(np.average(g["m2"], weights=A), 2)}
        r["dQ2_pct"] = round(100 * (r["Q2_P2"] - r["Q2_P1"]) / r["Q2_P1"], 1)
        r["stage_of_mean_P1"] = G.STAGE_NAMES[int(G.stage_index(np.array([r["Q2_P1"]]))[0])]
        r["stage_of_mean_P2"] = G.STAGE_NAMES[int(G.stage_index(np.array([r["Q2_P2"]]))[0])]
        r["pct_area_changed_stage"] = round(100 * A[s2 != s1].sum() / tot, 1)
        r["pct_area_to_drier"] = round(100 * A[s2 < s1].sum() / tot, 1)
        r["pct_area_to_wetter"] = round(100 * A[s2 > s1].sum() / tot, 1)
        r["pct_area_>=3of4_variants_drier"] = round(100 * A[g["drier_n"].values >= 3].sum() / tot, 1)
        r["pct_area_4of4_variants_drier"] = round(100 * A[g["drier_n"].values >= 4].sum() / tot, 1)
        dq = []
        for v in vcols:
            a = np.average(g["q1_" + v], weights=A)
            b = np.average(g["q2_" + v], weights=A)
            dq.append(100 * (b - a) / a)
        if dq:
            r["dQ2_pct_min_variant"] = round(min(dq), 1)
            r["dQ2_pct_max_variant"] = round(max(dq), 1)
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
        print("ensemble_arrays.npz not found: run RUN_Ensemble_Q2_Shift.bat first.")
        sys.exit(1)
    z = np.load(NPZ)
    lat, lon, valid = z["lat"], z["lon"], z["valid"]
    LON, LAT = np.meshgrid(lon, lat)
    ii, jj = np.where(valid)
    d = dict(i=ii, j=jj, lon=LON[ii, jj], lat=LAT[ii, jj], Q1=z["Q1"][ii, jj], Q2=z["Q2"][ii, jj],
             m1=z["m1"][ii, jj], m2=z["m2"][ii, jj], drier_n=z["drier_n"][ii, jj])
    for k in z.files:
        if k.startswith("Q1_"):
            v = k[3:]
            if "Q2_" + v in z.files:
                d["q1_" + v] = z["Q1_" + v][ii, jj]
                d["q2_" + v] = z["Q2_" + v][ii, jj]
    cells = pd.DataFrame(d)
    cells["area_km2"] = (KM_PER_DEG * 0.1) ** 2 * np.cos(np.deg2rad(cells["lat"]))
    cells = cells[np.isfinite(cells["Q1"]) & np.isfinite(cells["Q2"])]

    print("Reading communes (reference layer Nord_Comm_Wilaya) and dissolving to wilayas (about 1 minute) ...", flush=True)
    com = gpd.read_file(SHP).to_crs(4326)
    com["geometry"] = com.geometry.simplify(0.003, preserve_topology=True).make_valid()
    com = com[com.geometry.notna() & ~com.geometry.is_empty]
    wil = com.dissolve(by="name_fr_1", as_index=False)[["name_fr_1", "name_ar_1", "geometry"]]
    wil["geometry"] = wil.geometry.buffer(0.004).buffer(-0.004)   # close slivers between communes
    print(f"  {len(com)} communes -> {len(wil)} wilayas")

    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs=4326)
    j = gpd.sjoin(pts, wil, predicate="within", how="inner").rename(columns={"name_fr_1": "wilaya"})
    j = j.drop(columns=[c for c in ("index_right",) if c in j.columns])
    j["zone"] = j["wilaya"].map(lambda w: ZONE_OF.get(norm(w), "Autre"))
    unmatched = sorted(set(w for w in j["wilaya"].unique() if norm(w) not in ZONE_OF))
    if unmatched:
        print("  WARNING wilayas not assigned to a zone:", unmatched)

    # ---- wilayas ----
    res = summarize(j)
    ar = wil.set_index("name_fr_1")["name_ar_1"].to_dict()
    res.insert(1, "wilaya_ar", res["wilaya"].map(ar))
    res.insert(2, "zone", res["wilaya"].map(lambda w: ZONE_OF.get(norm(w), "Autre")))
    res = res.sort_values("dQ2_pct")
    res.to_csv(os.path.join(G.OUT, "wilaya_stage_shift.csv"), index=False, encoding="utf-8-sig")

    # ---- zones ----
    j_nord = j[j["zone"].isin(NORD_ZONES)]
    zt = pd.concat([summarize(j, "zone"),
                    summarize(j_nord.assign(zone="NORD sans Sahara"), "zone"),
                    summarize(j.assign(zone="TOUT LE DOMAINE"), "zone")], ignore_index=True)
    zt["o"] = zt["zone"].map(lambda x: ORDER.index(x) if x in ORDER else 99)
    zt = zt.sort_values("o").drop(columns="o")
    zt.to_csv(os.path.join(G.OUT, "main_results_table.csv"), index=False, encoding="utf-8-sig")

    zl = [f"Zone statistics, ensemble Q2 (4 variants), P1={G.P1}, P2={G.P2}", "",
          f"{'zone':27s} {'km2':>8s} {'Q2 P1':>7s} {'Q2 P2':>7s} {'dQ2%':>6s} {'range(4 var)':>14s} {'chg%':>5s} {'dry%':>5s} {'wet%':>5s} {'4/4%':>5s}"]
    for _, r in zt.iterrows():
        lo, hi = r.get("dQ2_pct_min_variant"), r.get("dQ2_pct_max_variant")
        rng = f"{lo:+.1f}/{hi:+.1f}" if pd.notna(lo) else ""
        zl.append(f"{r['zone'][:27]:27s} {r['area_km2']:8.0f} {r['Q2_P1']:7.1f} {r['Q2_P2']:7.1f} {r['dQ2_pct']:6.1f} {rng:>14s} "
                  f"{r['pct_area_changed_stage']:5.1f} {r['pct_area_to_drier']:5.1f} {r['pct_area_to_wetter']:5.1f} "
                  f"{r['pct_area_4of4_variants_drier']:5.1f}")
    zl += ["", "range = min/max of dQ2 among the 4 variants; chg/dry/wet = % of area that changed stage / moved to a drier / a wetter stage;",
           "4/4 = % of area where all 4 variants show decreasing Q2."]
    open(os.path.join(G.OUT, "zone_summary.txt"), "w", encoding="utf-8").write("\n".join(zl))
    print("\n".join(zl))

    lines = [f"Wilaya statistics, ensemble Q2, P1={G.P1}, P2={G.P2}; {len(res)} wilayas", "",
             f"{'wilaya':24s} {'zone':14s} {'cells':>5s} {'Q2 P1':>7s} {'Q2 P2':>7s} {'dQ2%':>6s} {'chg%':>5s} {'dry%':>5s} {'>=3/4':>6s}  stage P1 -> P2"]
    for _, r in res.iterrows():
        flag = "*" if r["small_sample"] else " "
        lines.append(f"{r['wilaya'][:23]:23s}{flag} {r['zone'][:14]:14s} {r['n_cells']:5d} {r['Q2_P1']:7.1f} {r['Q2_P2']:7.1f} {r['dQ2_pct']:6.1f} "
                     f"{r['pct_area_changed_stage']:5.1f} {r['pct_area_to_drier']:5.1f} {r['pct_area_>=3of4_variants_drier']:6.1f}  "
                     f"{r['stage_of_mean_P1']} -> {r['stage_of_mean_P2']}")
    lines += ["", f"* fewer than {MIN_CELLS} cells: indicative only."]
    open(os.path.join(G.OUT, "wilaya_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))

    # ---- map ----
    m = wil.merge(res[["wilaya", "dQ2_pct"]], left_on="name_fr_1", right_on="wilaya", how="left")
    fig, ax = plt.subplots(figsize=(11, 6))
    m.plot(column="dQ2_pct", cmap="RdBu", vmin=-30, vmax=30, legend=True, ax=ax,
           missing_kwds=dict(color="#eeeeee"), edgecolor="#444444", linewidth=0.4,
           legend_kwds=dict(label="Change in Q2 (%), P2 vs P1"))
    ax.set_xlim(-2.4, 9.2)
    ax.set_ylim(30.6, 37.2)
    ax.set_title("Change in Emberger Q2 by wilaya (ensemble of 4 variants)")
    fig.savefig(os.path.join(G.OUT, "map_wilaya_dQ2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- zone figure ----
    zz = zt[zt["zone"].isin(ORDER)]
    xs = np.arange(len(zz))
    vals = zz["dQ2_pct"].values.astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(xs, vals, color=["#d6604d" if v < 0 else "#4393c3" for v in vals])
    if "dQ2_pct_min_variant" in zz and zz["dQ2_pct_min_variant"].notna().all():
        lo = np.abs(vals - zz["dQ2_pct_min_variant"].values)
        hi = np.abs(zz["dQ2_pct_max_variant"].values - vals)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace(" et ", "\net ").replace(" sans ", "\nsans ").replace("TOUT ", "TOUT\n") for t in zz["zone"]], fontsize=7)
    ax.set_ylabel("Change in Q2 (%), P2 vs P1")
    ax.set_title("Change in Q2 by zone (bar = ensemble mean, whiskers = range among the 4 variants)", fontsize=10)
    fig.savefig(os.path.join(G.OUT, "fig_zone_dQ2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- climagram ----
    fig, ax = plt.subplots(figsize=(10, 7))
    lims = [lim for lim, _ in G.STAGES[:-1]]
    for lim in lims:
        ax.axhline(lim, color="#999999", lw=0.8)
    bounds = [4] + lims + [250]
    for k, n in enumerate(G.STAGE_NAMES):
        ax.text(11.8, np.sqrt(bounds[k] * bounds[k + 1]), n, ha="right", va="center", fontsize=9, color="#444444")
    for v in (0, 3, 7):
        ax.axvline(v, color="#bbbbbb", lw=0.8, ls=":")
    for _, r in res[~res["small_sample"]].iterrows():
        ax.annotate("", xy=(r["m_P2"], r["Q2_P2"]), xytext=(r["m_P1"], r["Q2_P1"]),
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.0, alpha=0.8))
        ax.text(r["m_P2"], r["Q2_P2"], r["wilaya"][:9], fontsize=6, color="#2c7fb8")
    if os.path.exists(PAIRS):
        pr = pd.read_csv(PAIRS)
        for _, r in pr.iterrows():
            ax.annotate("", xy=(r["m_new"], r["Q2_new"]), xytext=(r["m_old"], r["Q2_old"]),
                        arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=1.2, ls="-" if r["continuous"] else "--"))
            ax.text(r["m_new"], r["Q2_new"], str(r["new_name"])[:10], fontsize=6, color="#d95f0e")
    ax.set_yscale("log")
    ax.set_ylim(4, 250)
    ax.set_xlim(-3, 12)
    ax.set_xlabel("m: mean minimum of the coldest month (C)")
    ax.set_ylabel("Emberger Q2 (log scale)")
    ax.set_title("Emberger climagram: blue = wilayas (P1 to P2), orange = stations (1913-1938 to 1985-2025; dashed = different station)", fontsize=9)
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
        open(os.path.join(G.OUT, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\error_log.txt - send it to Claude.")
        sys.exit(1)
