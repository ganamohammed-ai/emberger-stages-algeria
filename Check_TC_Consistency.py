#!/usr/bin/env python3
"""
Consistency check between your TerraClimate files (1985-2025) and the files downloaded for 1958-1984.
It downloads 1985-1990 again with the SAME method as the 1958-1984 files (into results_grid\\tc_check\\) and
compares them cell by cell with your original 1985-1990 files. If both come from the same product, differences are ~0.
Output: results_grid\\tc_consistency.txt
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Download_TerraClimate_1958_1984_v2 as D
import Algeria_Grid_Q2_Shift as G
import Algeria_Windows_Transitions_v3 as W

CHK = os.path.join(HERE, "results_grid", "tc_check")
os.makedirs(CHK, exist_ok=True)
lines = ["Comparison: re-downloaded (THREDDS) minus your local files, 1985-1990", ""]
host, path = D.SOURCES[0]
import xarray as xr
for var in ("ppt", "tmax", "tmin", "pet"):
    diffs, base = [], []
    for y in range(1985, 1991):
        dest = os.path.join(CHK, f"TerraClimate_{var}_{y}_NordALG.nc")
        if not os.path.exists(dest):
            try:
                D.fetch(D.build_url(host, path, var, y), dest)
            except Exception as e:
                lines.append(f"{var} {y}: download failed {str(e)[:120]}")
                continue
        a, la, lo = G.load_tc_year(var, y)          # local files (glob in TC_DIR)
        if a is None:
            continue
        ds = xr.open_dataset(dest)
        nm = var if var in ds.data_vars else [v for v in ds.data_vars if not v.startswith("crs")][0]
        b = ds[nm].transpose("time", "lat", "lon").values.astype("float32")
        lb, ob = ds["lat"].values, ds["lon"].values
        ds.close()
        try:
            b = W.align_to(b, lb, ob, np.asarray(la, float), np.asarray(lo, float))
        except Exception as e:
            lines.append(f"{var} {y}: grid mismatch {e}")
            continue
        d = b - a
        diffs.append(d[np.isfinite(d)])
        base.append(a[np.isfinite(a)])
    if diffs:
        d = np.concatenate(diffs)
        a = np.concatenate(base)
        lines.append(f"{var:5s} mean diff {d.mean():+.3f}  mean abs diff {np.abs(d).mean():.3f}  max abs {np.abs(d).max():.2f}  "
                     f"(mean of local values {a.mean():.2f}; n={d.size})")
lines += ["", "Interpretation: values near 0 (relative to the mean) mean the two sets come from the same product,",
          "so the step seen around 1985-86 is climatic, not a splice artefact."]
open(os.path.join(HERE, "results_grid", "tc_consistency.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
