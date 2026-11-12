#!/usr/bin/env python3
"""Checks which Python packages are installed and whether the climate grids can be opened.
Writes results/env_check.txt. Standard library only; nothing is installed or modified."""
import importlib, os, sys, glob, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
ROOT = r"D:\DATA ALG_CLIP"
FOLDERS = {
    "TerraClimate": os.path.join(ROOT, "14_Climate_TerraClimate"),
    "CHIRPS": os.path.join(ROOT, "06_Climate_CHIRPS"),
    "ERA5Land_monthly": os.path.join(ROOT, "08_Climate_ERA5Land_Monthly"),
}
lines = [f"Python {sys.version.split()[0]} on {sys.platform}", ""]

lines.append("== Packages ==")
for name in ["numpy", "pandas", "scipy", "matplotlib", "xarray", "netCDF4", "h5netcdf", "h5py",
             "rasterio", "osgeo.gdal", "geopandas", "shapely", "pyproj", "rioxarray"]:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", None) or getattr(m, "__gdal_version__", "") or ""
        lines.append(f"  OK       {name} {v}")
    except Exception as e:
        lines.append(f"  MISSING  {name}")

lines.append("")
lines.append("== Data folders ==")
samples = {}
for key, path in FOLDERS.items():
    if not os.path.isdir(path):
        lines.append(f"  {key}: folder not found: {path}")
        continue
    files = []
    for dp, dn, fn in os.walk(path):
        for f in fn:
            files.append(os.path.join(dp, f))
    files.sort()
    exts = {}
    for f in files:
        exts[os.path.splitext(f)[1].lower()] = exts.get(os.path.splitext(f)[1].lower(), 0) + 1
    lines.append(f"  {key}: {len(files)} files, extensions {exts}")
    if files:
        lines.append(f"     first: {os.path.basename(files[0])}")
        lines.append(f"     last : {os.path.basename(files[-1])}")
        samples[key] = files[len(files) // 2]

lines.append("")
lines.append("== Try opening one sample file per folder ==")
for key, f in samples.items():
    lines.append(f"  {key}: {os.path.basename(f)}")
    ext = os.path.splitext(f)[1].lower()
    ok = False
    try:
        if ext in (".nc", ".nc4"):
            try:
                import xarray as xr
                ds = xr.open_dataset(f)
                lines.append(f"     xarray OK: vars={list(ds.data_vars)[:12]} dims={dict(ds.sizes)}")
                ok = True
            except Exception as e:
                lines.append(f"     xarray failed: {str(e)[:150]}")
            if not ok:
                try:
                    import netCDF4
                    ds = netCDF4.Dataset(f)
                    lines.append(f"     netCDF4 OK: vars={list(ds.variables)[:12]}")
                    ok = True
                except Exception as e:
                    lines.append(f"     netCDF4 failed: {str(e)[:150]}")
        elif ext in (".tif", ".tiff"):
            try:
                import rasterio
                with rasterio.open(f) as r:
                    lines.append(f"     rasterio OK: {r.count} band(s), {r.width}x{r.height}, crs={r.crs}, nodata={r.nodata}")
                ok = True
            except Exception as e:
                lines.append(f"     rasterio failed: {str(e)[:150]}")
            if not ok:
                try:
                    from osgeo import gdal
                    d = gdal.Open(f)
                    lines.append(f"     gdal OK: {d.RasterCount} band(s), {d.RasterXSize}x{d.RasterYSize}")
                    ok = True
                except Exception as e:
                    lines.append(f"     gdal failed: {str(e)[:150]}")
    except Exception:
        lines.append("     error: " + traceback.format_exc()[-200:])
    lines.append(f"     RESULT: {'can be read' if ok else 'CANNOT be read with installed packages'}")

txt = "\n".join(lines)
open(os.path.join(OUT, "env_check.txt"), "w", encoding="utf-8").write(txt)
print(txt)
