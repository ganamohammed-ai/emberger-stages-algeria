#!/usr/bin/env python3
"""
Station validation of the gridded products (no raster reading needed).

Inputs (already produced by earlier runs), in results_grid\\ :
  station_vs_grid.csv              station vs TerraClimate / CHIRPS: P, M, m, Q2 and stage
  station_vs_ERA5_temperature.csv  station vs ERA5-Land / TerraClimate: M and m
Outputs in results_grid\\validation\\ :
  validation_metrics.csv           bias, MAE, RMSE, r, n for P, M, m, Q2
  validation_stage_agreement.csv   exact / within-one-stage agreement, macro-F1
  validation_confusion_TC.csv, validation_confusion_CHIRPS.csv
  validation_summary.txt
Limits of this check: 20-25 GHCN stations, one multi-year climatology each, no homogeneity test.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "results_grid")
OUT = os.path.join(GRID, "validation")
STAGES = ["aride", "semi-aride", "subhumide", "humide", "per-humide"]


def metrics(obs, mod):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), bias=np.nan, MAE=np.nan, RMSE=np.nan, r=np.nan, rel_bias_pct=np.nan)
    d = mod[ok] - obs[ok]
    return dict(n=int(ok.sum()), bias=d.mean(), MAE=np.abs(d).mean(), RMSE=np.sqrt((d ** 2).mean()),
                r=np.corrcoef(obs[ok], mod[ok])[0, 1], rel_bias_pct=100 * d.mean() / obs[ok].mean())


def confusion(obs, mod, k=5):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    C = np.zeros((k, k), int)
    for a, b in zip(obs[ok].astype(int), mod[ok].astype(int)):
        C[a, b] += 1
    return C


def agreement(C):
    n = C.sum()
    exact = np.trace(C) / n if n else np.nan
    idx = np.arange(C.shape[0])
    within1 = C[np.abs(idx[:, None] - idx[None, :]) <= 1].sum() / n if n else np.nan
    f1 = []
    for i in range(C.shape[0]):
        tp, fp, fn = C[i, i], C[:, i].sum() - C[i, i], C[i, :].sum() - C[i, i]
        if C[i, :].sum() + C[:, i].sum() > 0:           # stage present in either obs or model
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    return exact, within1, float(np.nanmean(f1)) if f1 else np.nan, n


def main():
    os.makedirs(OUT, exist_ok=True)
    g = pd.read_csv(os.path.join(GRID, "station_vs_grid.csv"))
    e = pd.read_csv(os.path.join(GRID, "station_vs_ERA5_temperature.csv"))
    rows = []
    for var, so, mods in (("P (mm/yr)", "P_station", {"TerraClimate": "P_TerraClimate", "CHIRPS": "P_CHIRPS"}),
                          ("M (C)", "M_station", {"TerraClimate": "M_TerraClimate"}),
                          ("m (C)", "m_station", {"TerraClimate": "m_TerraClimate"}),
                          ("Q2", "Q2_station", {"TerraClimate": "Q2_TerraClimate", "CHIRPS (P) + TC temp": "Q2_CHIRPSP_TCtemp"})):
        for name, col in mods.items():
            rows.append(dict(variable=var, product=name, **metrics(g[so], g[col])))
    for var, so, mods in (("M (C)", "M_station", {"ERA5-Land": "M_ERA5", "TerraClimate": "M_TC"}),
                          ("m (C)", "m_station", {"ERA5-Land": "m_ERA5", "TerraClimate": "m_TC"})):
        for name, col in mods.items():
            rows.append(dict(variable=var + " [ERA5 file]", product=name, **metrics(e[so], e[col])))
    M = pd.DataFrame(rows).round(3)
    M.to_csv(os.path.join(OUT, "validation_metrics.csv"), index=False, encoding="utf-8-sig")

    ag = []
    lines = ["Station validation of gridded products (GHCN stations, one climatology each)", ""]
    lines += [M.to_string(index=False), ""]
    for name, col in (("TerraClimate", "stage_TC"), ("CHIRPS P + TC temp", "stage_CHIRPS")):
        C = confusion(g["stage_station"], g[col])
        pd.DataFrame(C, index=[f"{s} (station)" for s in STAGES], columns=[f"{s} ({name})" for s in STAGES]).to_csv(
            os.path.join(OUT, f"validation_confusion_{'TC' if col == 'stage_TC' else 'CHIRPS'}.csv"), encoding="utf-8-sig")
        ex, w1, f1, n = agreement(C)
        ag.append(dict(product=name, n_stations=int(n), exact_stage_agreement=round(ex, 3),
                       within_one_stage=round(w1, 3), macro_F1=round(f1, 3)))
        lines.append(f"{name}: n={n}, exact stage agreement {100*ex:.0f}%, within one stage {100*w1:.0f}%, macro-F1 {f1:.2f}")
        wrong = g[(g["stage_station"] != g[col]) & g[col].notna()]
        for _, r in wrong.iterrows():
            lines.append(f"    {r['name']:26s} station stage {int(r['stage_station'])} vs {int(r[col])}  (Q2 station {r['Q2_station']:.1f})")
    pd.DataFrame(ag).to_csv(os.path.join(OUT, "validation_stage_agreement.csv"), index=False, encoding="utf-8-sig")
    lines += ["", "Reading: bias is product minus station. A grid cell (4-5 km) is compared with a point, and station",
              "elevation differs from the cell mean elevation, so part of the error is representativeness, not product error.",
              "These stations validate the climatology (level), not the 1961-1990 to 1991-2020 change."]
    open(os.path.join(OUT, "validation_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
