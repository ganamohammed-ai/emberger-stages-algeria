#!/usr/bin/env python3
"""
Download TerraClimate 1958-1984 (ppt, tmax, tmin, pet) for northern Algeria from the Climatology Lab THREDDS
NetCDF Subset Service, so that the 30-year normals 1961-1990 and 1991-2020 can be computed.

Only the box lat 30.7-37.1, lon -2.3-9.1 is requested, so each file is small.
Files are saved as  TerraClimate_{var}_{YYYY}_NordALG.nc  in  D:\\DATA ALG_CLIP\\14_Climate_TerraClimate
(same naming as your existing files; years already present are skipped).
Run by double-clicking RUN_Download_TerraClimate.bat.  Needs internet; no account or password.
A log is written to  results_grid\\download_log.txt.
"""
import os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"D:\DATA ALG_CLIP\14_Climate_TerraClimate"
LOG = os.path.join(HERE, "results_grid", "download_log.txt")
VARS = ["ppt", "tmax", "tmin", "pet"]
YEARS = range(1958, 1985)
BOX = dict(north=37.1, south=30.7, west=-2.3, east=9.1)
# (host, path template); tried in this order, the first that works is remembered
SOURCES = [
    ("http://thredds.northwestknowledge.net:8080", "/thredds/ncss/TERRACLIMATE_ALL/data/TerraClimate_{var}_{year}.nc"),
    ("https://www.reacchpna.org", "/thredds/ncss/TERRACLIMATE_ALL/data/TerraClimate_{var}_{year}.nc"),
]
PLAUSIBLE = {"ppt": (0, 3000), "tmax": (-15, 60), "tmin": (-30, 45), "pet": (0, 800)}


def log(msg):
    print(msg, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def build_url(host, path, var, year):
    q = (f"var={var}&north={BOX['north']}&west={BOX['west']}&east={BOX['east']}&south={BOX['south']}"
         f"&disableProjSubset=on&horizStride=1&time_start={year}-01-01T00%3A00%3A00Z&time_end={year}-12-31T23%3A59%3A59Z&timeStride=1&accept=netcdf")
    return host + path.format(var=var, year=year) + "?" + q


def fetch(url, dest, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if data[:3] != b"CDF" and data[:4] != b"\x89HDF":
        raise ValueError("response is not a NetCDF file (server returned: %r)" % data[:120])
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return len(data)


def validate(path, var):
    import xarray as xr
    import numpy as np
    ds = xr.open_dataset(path)
    try:
        name = var if var in ds.data_vars else [v for v in ds.data_vars if not v.startswith("crs")][0]
        a = ds[name].values
        nt = a.shape[0] if a.ndim == 3 else -1
        lo, hi = PLAUSIBLE[var]
        v = a[np.isfinite(a)]
        ok = nt == 12 and v.size > 0 and lo <= float(v.min()) and float(v.max()) <= hi
        return ok, f"shape {a.shape}, min {float(v.min()):.1f}, max {float(v.max()):.1f}" if v.size else f"shape {a.shape}, no valid values"
    finally:
        ds.close()


def main():
    if not os.path.isdir(OUT_DIR):
        log(f"Folder not found: {OUT_DIR}")
        sys.exit(1)
    log(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    good = None
    n_ok = n_skip = n_fail = 0
    for year in YEARS:
        for var in VARS:
            dest = os.path.join(OUT_DIR, f"TerraClimate_{var}_{year}_NordALG.nc")
            if os.path.exists(dest) and os.path.getsize(dest) > 10000:
                n_skip += 1
                continue
            order = ([good] if good else []) + [s for s in SOURCES if s != good]
            done = False
            for host, path in order:
                url = build_url(host, path, var, year)
                try:
                    size = fetch(url, dest)
                    ok, info = validate(dest, var)
                    if not ok:
                        os.remove(dest)
                        raise ValueError("validation failed: " + info)
                    good = (host, path)
                    log(f"OK   {var} {year}  {size/1024:.0f} KB  {info}  [{host}]")
                    n_ok += 1
                    done = True
                    break
                except Exception as e:
                    log(f"fail {var} {year} via {host}: {str(e)[:200]}")
                    time.sleep(1)
            if not done:
                n_fail += 1
                if good is None and n_ok == 0:
                    log("\nThe very first file could not be downloaded from any source.")
                    log("Send results_grid\\download_log.txt to Claude (the server address may have changed).")
                    sys.exit(2)
    log(f"=== finished: {n_ok} downloaded, {n_skip} already present, {n_fail} failed ===")
    if n_fail:
        log("Run the .bat again to retry the failed files (finished ones are skipped).")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
