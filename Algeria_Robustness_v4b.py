#!/usr/bin/env python3
"""
Robustness analyses (step 1 after the external review), TerraClimate native grid, northern Algeria.

Run by double-clicking RUN_Robustness_v4b.bat (v4b adds the Daget 1977 limit set).  Needs Algeria_Windows_Transitions_v3.py (same folder) and the same
inputs as that script.  Outputs in results_grid\\robust\\ :

  A  q2_method_comparison.csv      climatological Q2 (P-mean, M, m of the window)  vs  mean of the annual Q2 values
  B  threshold_sensitivity.csv     % area to a drier / wetter stage for perturbed Q2 limits (and any limit sets you add)
  C  decomposition_P_M_m.csv       change of Q2 split into precipitation, M (hottest month Tmax) and m (coldest month Tmin)
                                   by exact Shapley decomposition of the 3 factors, by zone
  D  changepoint_robust.csv        Pettitt (block-bootstrap p) and Mann-Kendall with Yue-Wang pre-whitening, by zone,
                                   full 1958-2025 series and 1985-2025 series only (avoids the data-source junction)
  E  period_vs_product.csv         same TerraClimate product, several period pairs (separates baseline from product)
  F  fig_stage_maps_W0_W1.png      stage maps of both windows and the transition map, side by side
  robust_summary.txt

Q2 = 2000 P / (M^2 - m^2), M and m in Kelvin.  W0 = 1961-1990, W1 = 1991-2020.
"""
import os, sys, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Algeria_Windows_Transitions_v3 as T
import Algeria_Grid_Q2_Shift as G
import Algeria_Wilaya_Zones as Z

OUTR = os.path.join(G.OUT, "robust")
BASE_LIMITS = list(G.LIMITS)                  # stage limits in use (edit them in Algeria_Q2_Stage_Shift.py)

# ---- EDIT HERE: add published limit sets once you have checked them against the original sources -----------------
# format: "label": [limit_aride_semiaride, limit_semiaride_subhumide, limit_subhumide_humide, limit_humide_perhumide]
# Daget (1977) approximations for m = 0 C, as quoted by Quezel (1979, Foret mediterraneenne 1:1): peraride < 10, aride 10-45,
# semi-aride 45-70, sub-humide 70-110, humide 110-150, perhumide > 150.  The five-stage scheme used here has no peraride stage,
# so peraride is merged with aride.  These limits were read from a secondary source and apply to m = 0 only; verify against Daget (1977).
EXTRA_LIMIT_SETS = {"Daget 1977 (m=0) 45/70/110/150": [45, 70, 110, 150]}
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)            # all limits multiplied by these factors (pure sensitivity, not literature)
BOOT_N = 2000
SEED = 12345


# ------------------------------------------------------------------ pure functions
def stage_with(q2, limits):
    idx = np.digitize(q2, limits).astype(np.float32)
    idx[~np.isfinite(q2)] = np.nan
    return idx


def drier_wetter(q0, q1, limits, area, mask):
    """% of area (within mask) moving to a drier / wetter stage / unchanged, for the given stage limits."""
    s0, s1 = stage_with(q0, limits), stage_with(q1, limits)
    ok = mask & np.isfinite(s0) & np.isfinite(s1)
    A = area[ok].sum()
    if A == 0:
        return np.nan, np.nan, np.nan
    dry = area[ok & (s1 < s0)].sum() / A * 100
    wet = area[ok & (s1 > s0)].sum() / A * 100
    return dry, wet, 100 - dry - wet


def shapley_q2(P0, M0, m0, P1, M1, m1):
    """Exact Shapley split of Q2(P1,M1,m1) - Q2(P0,M0,m0) among the three factors P, M, m (arrays).
    The three parts add up exactly to the total change in every cell."""
    f = G.q2_from
    vals = {}
    for sP, sM, sm in itertools.product((0, 1), repeat=3):
        vals[(sP, sM, sm)] = f(P1 if sP else P0, M1 if sM else M0, m1 if sm else m0)
    contrib = np.zeros((3,) + np.shape(P0))
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        state = [0, 0, 0]
        for fct in perm:
            before = vals[tuple(state)]
            state[fct] = 1
            contrib[fct] += vals[tuple(state)] - before
    contrib /= len(perms)
    total = vals[(1, 1, 1)] - vals[(0, 0, 0)]
    return contrib[0], contrib[1], contrib[2], total


def mean_annual_q2(per_year, y0, y1):
    ys = [y for y in range(y0, y1 + 1) if y in per_year]
    if len(ys) < (y1 - y0 + 1):
        return None
    qs = [G.q2_from(per_year[y]["P"], per_year[y]["Tx"].max(axis=0), per_year[y]["Tn"].min(axis=0)) for y in ys]
    return np.nanmean(np.stack(qs), axis=0)


def pettitt_stat(x):
    x = np.asarray(x, float)
    n = len(x)
    s = np.sign(x[:, None] - x[None, :])
    U = np.array([s[: t + 1, t + 1:].sum() for t in range(n - 1)])
    k = int(np.argmax(np.abs(U)))
    return k, float(abs(U[k]))


def block_bootstrap_pettitt(x, n_boot=BOOT_N, block=None, seed=SEED):
    """Pettitt K with a p-value from a moving-block bootstrap of the demeaned series (keeps autocorrelation).
    Note: for a series with a long trend the resampled series are more variable than a stationary null, so p is conservative."""
    x = np.asarray(x, float)
    n = len(x)
    block = block or max(2, int(round(n ** (1 / 3))))
    k, K = pettitt_stat(x)
    rng = np.random.default_rng(seed)
    z = x - x.mean()
    nb = int(np.ceil(n / block))
    starts_max = n - block
    ge = 0
    for _ in range(n_boot):
        st = rng.integers(0, starts_max + 1, nb)
        xb = np.concatenate([z[s:s + block] for s in st])[:n]
        ge += pettitt_stat(xb)[1] >= K
    return k, K, (ge + 1) / (n_boot + 1), block


def sen_slope(y):
    y = np.asarray(y, float)
    n = len(y)
    sl = [(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return float(np.median(sl))


def mann_kendall(y):
    """Mann-Kendall S, Z and two-sided p (normal approximation, tie-corrected)."""
    from math import erf, sqrt
    y = np.asarray(y, float)
    n = len(y)
    S = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, cnt = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18.0
    z = 0.0 if S == 0 else (S - np.sign(S)) / np.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(S), float(z), float(p)


def mk_prewhitened(y):
    """Yue-Wang trend-free pre-whitening Mann-Kendall.  Returns (Sen slope per step, lag-1 r, Z, p)."""
    y = np.asarray(y, float)
    t = np.arange(len(y))
    b = sen_slope(y)
    d = y - b * t
    d = d - np.median(d)
    r1 = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 3 else 0.0
    r1 = max(0.0, r1)                                  # only remove positive autocorrelation
    yw = d[1:] - r1 * d[:-1] + b * t[1:]
    S, z, p = mann_kendall(yw)
    return b, r1, z, p


def transition_map(s0, s1):
    """-k = k stages drier, 0 = unchanged, +k = wetter (NaN where either stage is missing)."""
    return s1 - s0


# ------------------------------------------------------------------ main
def prepare():
    print("Reading TerraClimate 1958-2025 (several minutes) ...", flush=True)
    per_year, lat, lon = T.load_all()
    if not per_year:
        print("No TerraClimate files found in", G.TC_DIR)
        sys.exit(1)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        for d in per_year.values():
            for k, v in list(d.items()):
                if v is not None:
                    d[k] = v[..., ::-1, :]
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    area = (111.32 * dlat) * (111.32 * dlon) * np.cos(np.deg2rad(np.repeat(lat[:, None], len(lon), axis=1)))
    print("Assigning cells to wilayas/zones ...", flush=True)
    lab = T.zone_labels(lat, lon)
    masks = {zn: (lab == zn) for zn in Z.ZONES}
    masks["NORD sans Sahara"] = np.isin(lab, Z.NORD_ZONES)
    return per_year, lat, lon, area, lab, masks


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    os.makedirs(OUTR, exist_ok=True)
    per_year, lat, lon, area, lab, masks = prepare()
    nord = masks["NORD sans Sahara"]
    zorder = Z.NORD_ZONES + ["NORD sans Sahara", "Sahara"]
    L = [f"Robustness analyses, TerraClimate. Stage limits in use: {BASE_LIMITS}", ""]

    W0, W1 = T.W0, T.W1
    f0, f1 = T.window_fields(per_year, *W0), T.window_fields(per_year, *W1)
    if f0 is None or f1 is None:
        print("W0/W1 incomplete: download TerraClimate 1958-1984 first.")
        sys.exit(1)

    # ---------- A: climatological Q2 versus mean of annual Q2 ----------
    a0, a1 = mean_annual_q2(per_year, *W0), mean_annual_q2(per_year, *W1)
    rowsA = []
    for zn in zorder:
        mk = masks[zn]
        c0, c1 = T.zone_mean(f0["Q2"], area, mk), T.zone_mean(f1["Q2"], area, mk)
        m0, m1 = T.zone_mean(a0, area, mk), T.zone_mean(a1, area, mk)
        rowsA.append(dict(zone=zn, Q2_clim_W0=c0, Q2_clim_W1=c1, change_pct_clim=100 * (c1 - c0) / c0,
                          Q2_meanannual_W0=m0, Q2_meanannual_W1=m1, change_pct_meanannual=100 * (m1 - m0) / m0))
    A = pd.DataFrame(rowsA).round(2)
    A.to_csv(os.path.join(OUTR, "q2_method_comparison.csv"), index=False, encoding="utf-8-sig")
    d1, w1_, u1 = drier_wetter(f0["Q2"], f1["Q2"], BASE_LIMITS, area, nord)
    d2, w2_, u2 = drier_wetter(a0, a1, BASE_LIMITS, area, nord)
    L += ["A. Climatological Q2 (window P-mean, M, m) versus mean of annual Q2 values, W0 -> W1:", A.to_string(index=False),
          f"   NORD area to drier stage: climatological {d1:.1f}% (wetter {w1_:.1f}%), mean-annual {d2:.1f}% (wetter {w2_:.1f}%)",
          "   (the stages in the paper use the climatological form; the annual series is used only for trend / change-point tests)", ""]

    # ---------- B: threshold sensitivity ----------
    sets = {"base": BASE_LIMITS}
    for s in SCALES:
        if s != 1.0:
            sets[f"all limits x {s}"] = [v * s for v in BASE_LIMITS]
    for i in range(len(BASE_LIMITS)):                  # move one limit at a time by +-10%
        for s in (0.9, 1.1):
            lim = list(BASE_LIMITS)
            lim[i] = lim[i] * s
            sets[f"limit {i+1} x {s}"] = lim
    sets.update(EXTRA_LIMIT_SETS)
    rowsB = []
    for name, lim in sets.items():
        dry, wet, unc = drier_wetter(f0["Q2"], f1["Q2"], lim, area, nord)
        rowsB.append(dict(limit_set=name, limits=" / ".join(f"{v:g}" for v in lim), pct_area_drier=dry, pct_area_wetter=wet,
                          pct_unchanged=unc))
    B = pd.DataFrame(rowsB).round(2)
    B.to_csv(os.path.join(OUTR, "threshold_sensitivity.csv"), index=False, encoding="utf-8-sig")
    L += ["B. Sensitivity of the area moving to a drier stage (W0 -> W1, NORD sans Sahara) to the stage limits:", B.to_string(index=False),
          f"   range of % area drier: {B['pct_area_drier'].min():.1f} to {B['pct_area_drier'].max():.1f}", ""]

    # ---------- C: decomposition of the Q2 change (Shapley over P, M, m) ----------
    rowsC = []
    for tag, (y0a, y0b), (y1a, y1b) in (("W0->W1", T.W0, T.W1), ("P1->P2", G.P1, G.P2)):
        g0, g1 = T.window_fields(per_year, y0a, y0b), T.window_fields(per_year, y1a, y1b)
        if g0 is None or g1 is None:
            continue
        cP, cM, cm, tot = shapley_q2(g0["P"], g0["M"], g0["m"], g1["P"], g1["M"], g1["m"])
        for zn in zorder:
            mk = masks[zn]
            q0 = T.zone_mean(g0["Q2"], area, mk)
            t_, p_, M_, m_ = (T.zone_mean(x, area, mk) for x in (tot, cP, cM, cm))
            rowsC.append(dict(pair=tag, zone=zn, Q2_before=q0, dQ2_total=t_, from_precip=p_, from_M_hottest_Tmax=M_,
                              from_m_coldest_Tmin=m_, dP_mm=T.zone_mean(g1["P"] - g0["P"], area, mk),
                              dM_C=T.zone_mean(g1["M"] - g0["M"], area, mk), dm_C=T.zone_mean(g1["m"] - g0["m"], area, mk),
                              precip_share_pct=100 * p_ / t_ if t_ else np.nan, temp_share_pct=100 * (M_ + m_) / t_ if t_ else np.nan))
    C = pd.DataFrame(rowsC).round(2)
    C.to_csv(os.path.join(OUTR, "decomposition_P_M_m.csv"), index=False, encoding="utf-8-sig")
    L += ["C. Change of Q2 split among precipitation, M and m (Shapley, exact; parts add to the total):", C.to_string(index=False),
          "   shares are meaningful only where the total change is clearly non-zero (Littoral P1->P2 is close to zero)", ""]

    # ---------- D: change point and trend with autocorrelation ----------
    ys = sorted(per_year)
    ann = {zn: [] for zn in zorder}
    for y in ys:
        d = per_year[y]
        q = G.q2_from(d["P"], d["Tx"].max(axis=0), d["Tn"].min(axis=0))
        for zn in zorder:
            ann[zn].append(T.zone_mean(q, area, masks[zn]))
    rowsD = []
    for zn in zorder:
        x_all = np.array(ann[zn])
        for lbl, y_from in (("1958-2025", 0), ("1985-2025", 1985)):
            sel = np.array([y >= y_from for y in ys]) & np.isfinite(x_all)
            x, yy = x_all[sel], np.array(ys)[sel]
            if len(x) < 20:
                continue
            k, K, p_boot, blk = block_bootstrap_pettitt(x)
            _, _, p_plain = T.pettitt(x)
            b, r1, z, pmk = mk_prewhitened(x)
            rowsD.append(dict(zone=zn, series=lbl, n=len(x), break_after_year=int(yy[k]), pettitt_p_standard=round(p_plain, 4),
                              pettitt_p_blockboot=round(p_boot, 4), block_len=blk, lag1_r=round(r1, 2),
                              sen_slope_Q2_per_decade=round(10 * b, 2), MK_prewhitened_p=round(pmk, 4)))
    D = pd.DataFrame(rowsD)
    D.to_csv(os.path.join(OUTR, "changepoint_robust.csv"), index=False, encoding="utf-8-sig")
    L += ["D. Change point (Pettitt, standard and block-bootstrap p) and trend (Mann-Kendall, pre-whitened) on annual area-mean Q2:",
          D.to_string(index=False),
          "   1985-2025 rows avoid the junction between the downloaded 1958-1984 files and the local 1985+ files", ""]

    # ---------- E: period versus product ----------
    pairs = [("1961-1990 -> 1991-2020", (1961, 1990), (1991, 2020)),
             ("1961-1980 -> 1981-2000", (1961, 1980), (1981, 2000)),
             ("1981-2000 -> 2001-2020", (1981, 2000), (2001, 2020)),
             ("1961-1984 -> 1985-2004", (1961, 1984), (1985, 2004)),
             ("1985-2004 -> 2005-2025", (1985, 2004), (2005, 2025)),
             ("1985-2004 -> 2005-2024", (1985, 2004), (2005, 2024)),
             ("1961-1990 -> 1995-2024", (1961, 1990), (1995, 2024))]
    rowsE = []
    for name, p0, p1 in pairs:
        g0, g1 = T.window_fields(per_year, *p0), T.window_fields(per_year, *p1)
        if g0 is None or g1 is None:
            continue
        dry, wet, unc = drier_wetter(g0["Q2"], g1["Q2"], BASE_LIMITS, area, nord)
        q0, q1 = T.zone_mean(g0["Q2"], area, nord), T.zone_mean(g1["Q2"], area, nord)
        rowsE.append(dict(product="TerraClimate", pair=name, dQ2_NORD_pct=100 * (q1 - q0) / q0, pct_area_drier=dry,
                          pct_area_wetter=wet, pct_unchanged=unc))
    E = pd.DataFrame(rowsE).round(2)
    E.to_csv(os.path.join(OUTR, "period_vs_product.csv"), index=False, encoding="utf-8-sig")
    L += ["E. Same product (TerraClimate), several period pairs, NORD sans Sahara:", E.to_string(index=False),
          "   The four-variant ensemble (CHIRPS, ERA5-Land) exists only for 1985-2004 -> 2005-2025 and is reported separately.", ""]

    # ---------- F: stage maps side by side ----------
    s0, s1 = stage_with(f0["Q2"], BASE_LIMITS), stage_with(f1["Q2"], BASE_LIMITS)
    tm = transition_map(s0, s1)
    for a in (s0, s1, tm):
        a[~nord] = np.nan
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = ["#d73027", "#fdae61", "#fee08b", "#66bd63", "#1a9850"]
    cm = ListedColormap(cols)
    nm = BoundaryNorm(np.arange(-0.5, 5), cm.N)
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    for ax, s, ttl in ((axs[0], s0, f"Stage {W0[0]}-{W0[1]}"), (axs[1], s1, f"Stage {W1[0]}-{W1[1]}")):
        im = ax.imshow(s, origin="lower", extent=ext, cmap=cm, norm=nm, aspect="auto")
        ax.set_title(ttl, fontsize=10)
    cb = fig.colorbar(im, ax=axs[:2], ticks=range(5), shrink=0.8)
    cb.ax.set_yticklabels(G.STAGE_NAMES)
    cm2 = ListedColormap(["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf"])
    nm2 = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], cm2.N)
    im2 = axs[2].imshow(np.clip(tm, -2, 1), origin="lower", extent=ext, cmap=cm2, norm=nm2, aspect="auto")
    axs[2].set_title("Transition (drier = red)", fontsize=10)
    cb2 = fig.colorbar(im2, ax=axs[2], ticks=[-2, -1, 0, 1], shrink=0.8)
    cb2.ax.set_yticklabels(["2+ stages drier", "1 stage drier", "unchanged", "wetter"])
    for ax in axs:
        ax.set_xlabel("longitude")
    axs[0].set_ylabel("latitude")
    fig.savefig(os.path.join(OUTR, "fig_stage_maps_W0_W1.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    L.append("F. fig_stage_maps_W0_W1.png written.")

    open(os.path.join(OUTR, "robust_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("\nDONE. Results in", OUTR)


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
    os.makedirs(OUTR, exist_ok=True)
    _log = open(os.path.join(OUTR, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        open(os.path.join(OUTR, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        print("\nThe error was saved to results_grid\\robust\\error_log.txt - send it to Claude.")
        sys.exit(1)
