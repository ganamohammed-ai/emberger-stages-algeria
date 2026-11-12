#!/usr/bin/env python3
"""
Annual precipitation of the Tlemcen wilaya, TerraClimate 1958-2025: was 1985-1999 a dry phase?
Double-click RUN_Tlemcen_Rain_v1.bat.  Output in results_grid\\tlemcen\\ :
  tlemcen_annual_P.csv      year, area-weighted mean P (mm), anomaly vs 1961-1990 (%)
  tlemcen_rain_summary.txt  means of fixed windows, 5-year means, rank of 1985-1999 among all 15-year windows,
                            Mann-Kendall / Pettitt (when available in Algeria_Robustness_v4)
  fig_tlemcen_rain.png
"""
import os, sys, traceback
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Robustness_v4 as R
import Algeria_Grid_Q2_Shift as G
import Algeria_Tlemcen_Check_v1 as C

OUTD = os.path.join(G.OUT, "tlemcen")


def main():
    os.makedirs(OUTD, exist_ok=True)
    per_year, lat, lon, area, lab, masks = R.prepare()
    tl = C.tlemcen_mask(lat, lon)
    ys = sorted(per_year)
    P = []
    for y in ys:
        p = per_year[y]["P"]
        ok = tl & np.isfinite(p)
        P.append(float(np.sum(p[ok] * area[ok]) / area[ok].sum()))
    df = pd.DataFrame({"year": ys, "P_mm": np.round(P, 1)})
    ref = df[(df.year >= 1961) & (df.year <= 1990)].P_mm.mean()
    df["anom_pct_vs_1961_90"] = np.round((df.P_mm / ref - 1) * 100, 1)
    df.to_csv(os.path.join(OUTD, "tlemcen_annual_P.csv"), index=False)
    s = df.set_index("year").P_mm
    L = ["Tlemcen wilaya, annual P, TerraClimate (area-weighted), %d-%d" % (ys[0], ys[-1]), ""]
    for a, b in [(1961, 1990), (1991, 2020), (1961, 1984), (1985, 1999), (2000, 2003), (2004, 2018), (2019, 2025), (1961, 1980), (1981, 2000), (2001, 2020)]:
        L.append(f"  mean {a}-{b}: {s.loc[a:b].mean():6.1f} mm  (n={len(s.loc[a:b])})")
    L.append("")
    L.append("5-year means:")
    for a in range(1960, 2025, 5):
        L.append(f"  {a}-{a+4}: {s.loc[a:a+4].mean():6.1f}")
    # all 15-year windows ranked
    w = {y0: s.loc[y0:y0 + 14].mean() for y0 in range(1958, 2012) if y0 + 14 <= ys[-1]}
    rk = sorted(w, key=w.get)
    L.append("")
    L.append("Driest 15-year windows (start year, mean mm): " + "; ".join(f"{y}-{y+14}: {w[y]:.0f}" for y in rk[:6]))
    L.append("Rank of 1985-1999 among %d windows (1 = driest): %d" % (len(w), rk.index(1985) + 1))
    L.append("Rank of 2004-2018 (1 = driest): %d" % (rk.index(2004) + 1))
    try:
        L.append("")
        L.append("Mann-Kendall (prewhitened) 1958-2025: " + str(R.mk_prewhitened(s.values)))
        L.append("Mann-Kendall (prewhitened) 1985-2025: " + str(R.mk_prewhitened(s.loc[1985:].values)))
        L.append("Sen slope 1958-2025 (mm/yr): " + str(R.sen_slope(s.values)))
    except Exception as e:
        L.append("trend tests skipped: %s" % e)
    open(os.path.join(OUTD, "tlemcen_rain_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(s.index, s.values, color="#9bb7c9", width=0.8)
        ax.plot(s.index, s.rolling(5, center=True).mean(), color="#1f3b57", lw=2, label="5-year mean")
        for (a, b, c, t) in [(1985, 1999, "#c0392b", "1985-1999"), (2004, 2018, "#2e7d32", "2004-2018")]:
            ax.hlines(s.loc[a:b].mean(), a, b, colors=c, lw=3, label=f"{t} mean {s.loc[a:b].mean():.0f} mm")
        ax.axhline(ref, color="grey", ls="--", lw=1, label=f"1961-1990 mean {ref:.0f} mm")
        ax.set_ylabel("Annual precipitation (mm)"); ax.set_title("Tlemcen wilaya, TerraClimate")
        ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
        fig.savefig(os.path.join(OUTD, "fig_tlemcen_rain.png"), dpi=200)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    try:
        os.makedirs(OUTD, exist_ok=True)
        main()
    except Exception:
        open(os.path.join(OUTD, "error_log.txt"), "w", encoding="utf-8").write(traceback.format_exc())
        raise
