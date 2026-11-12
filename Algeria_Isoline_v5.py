#!/usr/bin/env python3
"""
Speed of the stage boundaries (isolines Q2 = 20, 40, 70), recomputed without matching profiles by latitude.

Method (area balance).  For a level L let A = cells with Q2 >= L (the wetter side).  The boundary moves by
    displacement = (area(A at time 0) - area(A at time 1)) / mean isoline length
(km, positive = the wetter side retreats = the boundary moves toward the drier side, i.e. northward in Algeria).
The isoline length is measured on the contour lines (marching squares) in km, so no segment matching between the two
maps is needed and several crossings per longitude band are handled.  Speed = displacement / years between window centres.

Uncertainty reported for every level and period pair:
  * bootstrap over longitude bands (resampling bands, 5,000 times, length-weighted) -> 95% interval
  * the same calculation for the limit moved by -10% and +10% (the limit itself is uncertain)
  * several period pairs (30-year windows, 20-year blocks, 1985-2004 vs 2005-2025)
  * number of bands whose median latitude profile crosses the level more than once (where the old method was ambiguous)
  * the old latitude-crossing estimate (median over bands) for comparison
Run by double-clicking RUN_Isoline_v5.bat.  Needs Algeria_Robustness_v4.py and Algeria_Windows_Transitions_v3.py.
Q2 is smoothed with a Gaussian (2 cells by default; 0 and 4 reported) before contouring, because cell-level noise
lengthens the contour and would bias the speed low.
Outputs in results_grid\\isoline\\ : isoline_velocity.csv, isoline_velocity_bands.csv, isoline_summary.txt,
fig_isolines_map.png, fig_isoline_velocity.png
Limits of the method: it gives the net displacement of the boundary along its whole length; it does not separate
movement northward from movement to higher ground (no elevation model is used), and the km are computed with a
local cos(latitude) scale, not an equal-area projection (error well below 1% at this extent).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Robustness_v4 as R
import Algeria_Windows_Transitions_v3 as T
import Algeria_Grid_Q2_Shift as G

OUTI = os.path.join(G.OUT, "isoline")
LEVELS = (20, 40, 70)
LEVEL_SCALES = (0.9, 1.0, 1.1)
BANDS = [(-2.5, -1.0), (-1.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 6.5), (6.5, 8.0), (8.0, 9.2)]
PAIRS = [("W0->W1", (1961, 1990), (1991, 2020)), ("B1->B2", (1961, 1980), (1981, 2000)),
         ("B2->B3", (1981, 2000), (2001, 2020)), ("P1->P2", (1985, 2004), (2005, 2025))]
SMOOTHS = (0, 2, 4)             # Gaussian smoothing (cells, about 4 km each) applied to Q2 before contouring
SMOOTH_MAIN = 2
BOOT_N = 5000
SEED = 2026


# ------------------------------------------------------------------ pure functions
def smooth_nan(Q, sigma):
    """Gaussian smoothing (sigma in cells) that ignores NaN (normalised convolution).  sigma 0 returns Q unchanged."""
    if sigma <= 0:
        return Q
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    w = np.isfinite(Q).astype(float)
    v = np.where(np.isfinite(Q), Q, 0.0)

    def conv(a):
        out = np.zeros_like(a)
        for ax in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[ax] = (r, r)
            ap = np.pad(a if ax == 0 else out, pad, mode="constant")
            res = np.zeros_like(a)
            n = a.shape[ax]
            for i, kv in enumerate(k):
                res += kv * (ap[i:i + n, :] if ax == 0 else ap[:, i:i + n])
            out = res
        return out
    num, den = conv(v), conv(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        o = num / den
    o[den < 0.5] = np.nan
    o[~np.isfinite(Q)] = np.nan
    return o


def isoline_length_km(Q, lat, lon, level, mask, lon_lo=None, lon_hi=None):
    """Length (km) of the contour Q = level inside the mask, counting only segments whose midpoint lies in [lon_lo, lon_hi)."""
    import contourpy
    z = np.ma.masked_invalid(np.where(mask, Q, np.nan))
    gen = contourpy.contour_generator(lon, lat, z)
    total = 0.0
    for line in gen.lines(level):
        if len(line) < 2:
            continue
        a, b = line[:-1], line[1:]
        mid_lon, mid_lat = 0.5 * (a[:, 0] + b[:, 0]), 0.5 * (a[:, 1] + b[:, 1])
        dx = (b[:, 0] - a[:, 0]) * 111.32 * np.cos(np.deg2rad(mid_lat))
        dy = (b[:, 1] - a[:, 1]) * 111.32
        seg = np.hypot(dx, dy)
        if lon_lo is not None:
            seg = seg[(mid_lon >= lon_lo) & (mid_lon < lon_hi)]
        total += float(seg.sum())
    return total


def wet_area(Q, level, area, mask, cols=None):
    ok = mask & np.isfinite(Q)
    if cols is not None:
        ok = ok & cols[None, :]
    return float(area[ok & (Q >= level)].sum())


def band_displacement(Q0, Q1, lat, lon, level, area, mask, lo, hi):
    """Returns (net area lost by the wetter side in km2, mean isoline length in km, displacement in km) for one band."""
    both = mask & np.isfinite(Q0) & np.isfinite(Q1)          # same cells in both maps
    cols = (lon >= lo) & (lon < hi)
    dA = wet_area(Q0, level, area, both, cols) - wet_area(Q1, level, area, both, cols)
    L0 = isoline_length_km(Q0, lat, lon, level, both, lo, hi)
    L1 = isoline_length_km(Q1, lat, lon, level, both, lo, hi)
    Lm = 0.5 * (L0 + L1)
    return dA, Lm, (dA / Lm if Lm > 0 else np.nan)


def n_crossings(profile, level):
    p = profile[np.isfinite(profile)] - level
    if len(p) < 3:
        return 0
    s = np.sign(p)
    s = s[s != 0]
    return int((np.diff(s) != 0).sum())


def boot_ci(dA, Lm, n_boot=BOOT_N, seed=SEED):
    """Length-weighted displacement with a 95% interval from resampling bands."""
    dA, Lm = np.asarray(dA, float), np.asarray(Lm, float)
    ok = np.isfinite(dA) & (Lm > 0)
    dA, Lm = dA[ok], Lm[ok]
    if len(dA) < 3:
        return np.nan, np.nan, np.nan
    est = dA.sum() / Lm.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dA), (n_boot, len(dA)))
    b = dA[idx].sum(axis=1) / Lm[idx].sum(axis=1)
    return float(est), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def old_method_median(Q0, Q1, lat, lon, level, mask, years_between):
    bs = T.boundary_shift(Q0, Q1, lat, lon, level, BANDS, mask, years_between)
    return (float(bs["km_per_decade"].median()), len(bs)) if len(bs) else (np.nan, 0)


# ------------------------------------------------------------------ main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTI, exist_ok=True)
    per_year, lat, lon, area, lab, masks = R.prepare()
    nord = masks["NORD sans Sahara"]
    fields = {}
    for tag, p0, p1 in PAIRS:
        f0, f1 = T.window_fields(per_year, *p0), T.window_fields(per_year, *p1)
        if f0 is not None and f1 is not None:
            fields[tag] = (f0["Q2"], f1["Q2"], ((p1[0] + p1[1]) - (p0[0] + p0[1])) / 2.0)
    rows_b, rows = [], []
    for tag, (Q0r, Q1r, yb) in fields.items():
        both = nord & np.isfinite(Q0r) & np.isfinite(Q1r)
        for sm in SMOOTHS:
            Q0, Q1 = smooth_nan(np.where(both, Q0r, np.nan), sm), smooth_nan(np.where(both, Q1r, np.nan), sm)
            for L in LEVELS:
                for sc in LEVEL_SCALES:
                    if sm != SMOOTH_MAIN and sc != 1.0:
                        continue
                    lev = L * sc
                    dAs, Ls = [], []
                    for lo, hi in BANDS:
                        dA, Lm, d = band_displacement(Q0, Q1, lat, lon, lev, area, nord, lo, hi)
                        cols = (lon >= lo) & (lon < hi)
                        okc = both[:, cols].any() if cols.any() else False
                        prof = np.nanmedian(np.where(both[:, cols], Q0r[:, cols], np.nan), axis=1) if okc else np.array([])
                        rows_b.append(dict(pair=tag, level=L, limit_scale=sc, smooth_cells=sm, band=f"{lo:g}..{hi:g}",
                                           dA_km2=round(dA, 0), isoline_km=round(Lm, 1), displacement_km=round(d, 2),
                                           km_per_decade=round(d / (yb / 10.0), 2) if np.isfinite(d) else np.nan,
                                           crossings_W0=n_crossings(prof, lev) if len(prof) else 0))
                        dAs.append(dA)
                        Ls.append(Lm)
                    est, lo_ci, hi_ci = boot_ci(dAs, Ls)
                    k = lambda v: v / (yb / 10.0)
                    rec = dict(pair=tag, level=L, limit_scale=sc, smooth_cells=sm, years_between=yb,
                               speed_km_per_decade=k(est), ci95_low=k(lo_ci), ci95_high=k(hi_ci), total_displacement_km=est)
                    if sc == 1.0 and sm == SMOOTH_MAIN:
                        old, nb = old_method_median(Q0r, Q1r, lat, lon, L, both, yb)
                        rec["old_method_median_km_per_decade"] = old
                        rec["old_method_n_bands"] = nb
                    rows.append(rec)
    B = pd.DataFrame(rows_b)
    S = pd.DataFrame(rows).round(2)
    B.to_csv(os.path.join(OUTI, "isoline_velocity_bands.csv"), index=False, encoding="utf-8-sig")
    S.to_csv(os.path.join(OUTI, "isoline_velocity.csv"), index=False, encoding="utf-8-sig")
    Bm = B[(B["limit_scale"] == 1.0) & (B["smooth_cells"] == SMOOTH_MAIN)]
    multi = Bm.groupby(["pair", "level"])["crossings_W0"].apply(lambda s: int((s > 1).sum()))
    L = ["Isoline speed (km per decade; + = toward the drier side), area-balance method, NORD sans Sahara", "",
         f"Per level and period pair (limit scale 1.0, smoothing {SMOOTH_MAIN} cells), 95% interval from bootstrapping the longitude bands:"]
    base = S[(S["limit_scale"] == 1.0) & (S["smooth_cells"] == SMOOTH_MAIN)]
    L.append(base.drop(columns=["limit_scale", "smooth_cells"]).to_string(index=False))
    L += ["", "Effect of the smoothing (cells) on the speed, limit scale 1.0:"]
    for (tag, lev), g in S[S["limit_scale"] == 1.0].groupby(["pair", "level"]):
        v = {int(r["smooth_cells"]): r["speed_km_per_decade"] for _, r in g.iterrows()}
        L.append(f"  {tag} Q2={lev}: " + ", ".join(f"smooth {k_}: {v[k_]:+.1f}" for k_ in sorted(v)))
    L += ["", "Effect of moving the limit by -10% / +10% (speed, km per decade):"]
    for (tag, lev), g in S[S["smooth_cells"] == SMOOTH_MAIN].groupby(["pair", "level"]):
        v = {r["limit_scale"]: r["speed_km_per_decade"] for _, r in g.iterrows()}
        L.append(f"  {tag} Q2={lev}: {v.get(0.9, np.nan):+.1f} / {v.get(1.0, np.nan):+.1f} / {v.get(1.1, np.nan):+.1f}   "
                 f"(bands with more than one crossing in the W0 median profile: {multi.get((tag, lev), 0)})")
    L += ["", "Reading: the interval comes from variation between longitude bands (spatial), not from product or station error;",
          "compare the period pairs for the temporal spread, and the limit scales for the effect of the uncertain stage limits."]
    open(os.path.join(OUTI, "isoline_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

    # ---------- figures ----------
    if "W0->W1" in fields:
        Q0, Q1, _ = fields["W0->W1"]
        fig, ax = plt.subplots(figsize=(8, 5))
        zz = lambda Q: np.ma.masked_invalid(smooth_nan(np.where(nord, Q, np.nan), SMOOTH_MAIN))
        for lev, col in zip(LEVELS, ("#b2182b", "#ef8a62", "#2166ac")):
            ax.contour(lon, lat, zz(Q0), levels=[lev], colors=[col], linewidths=1.2, linestyles="solid")
            ax.contour(lon, lat, zz(Q1), levels=[lev], colors=[col], linewidths=1.2, linestyles="dashed")
            ax.plot([], [], color=col, label=f"Q2 = {lev}")
        ax.plot([], [], color="black", ls="solid", label="1961-1990")
        ax.plot([], [], color="black", ls="dashed", label="1991-2020")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"Isolines of Q2 (smoothed {SMOOTH_MAIN} cells), 1961-1990 solid, 1991-2020 dashed", fontsize=10)
        fig.savefig(os.path.join(OUTI, "fig_isolines_map.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tags = [t for t, _, _ in PAIRS if t in fields]
    w = 0.8 / max(1, len(tags))
    for i, tag in enumerate(tags):
        g = base[base["pair"] == tag].set_index("level").reindex(LEVELS)
        x = np.arange(len(LEVELS)) + (i - (len(tags) - 1) / 2) * w
        err = [g["speed_km_per_decade"] - g["ci95_low"], g["ci95_high"] - g["speed_km_per_decade"]]
        ax.bar(x, g["speed_km_per_decade"], w * 0.9, yerr=err, capsize=3, label=tag)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"Q2 = {v}" for v in LEVELS])
    ax.set_ylabel("km per decade (+ = toward the drier side)")
    ax.set_title("Boundary speed by period pair (bars = estimate, whiskers = 95% band bootstrap)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUTI, "fig_isoline_velocity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nDONE. Results in", OUTI)


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
    os.makedirs(OUTI, exist_ok=True)
    _log = open(os.path.join(OUTI, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        open(os.path.join(OUTI, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\isoline\\error_log.txt - send it to Claude.")
        sys.exit(1)
