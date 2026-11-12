#!/usr/bin/env python3
"""
Inventory of northern-Algeria meteorological stations in NOAA GHCN-Daily and ISD.
Run on YOUR computer (needs internet):   python Algeria_North_Station_Inventory.py
Outputs (in the current folder):
  stations_north_GHCN.csv   - GHCN stations with TMAX/TMIN/PRCP year counts in the period
  stations_north_ISD.csv    - ISD stations (history file) overlapping the period
  summary.txt               - counts
Parameters below can be edited.
"""
import csv, io, urllib.request, collections

LAT_MIN, LAT_MAX = 30.7, 37.1      # northern Algeria domain (same as NordALG data)
LON_MIN, LON_MAX = -2.25, 9.1
Y0, Y1 = 1985, 2025                # analysis period
MIN_YEARS = 20                     # minimum years with data

GHCN_ST = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_INV = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
ISD_HIST = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

def get(url):
    print("downloading", url)
    return urllib.request.urlopen(url, timeout=120).read().decode("latin-1")

def inside(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX

# ---------- GHCN-Daily ----------
stations = {}
for line in get(GHCN_ST).splitlines():
    sid = line[0:11]
    if not sid.startswith("AG"):          # AG = Algeria
        continue
    lat, lon = float(line[12:20]), float(line[21:30])
    if inside(lat, lon):
        stations[sid] = dict(id=sid, lat=lat, lon=lon, elev=line[31:37].strip(), name=line[41:71].strip())

yrs = collections.defaultdict(dict)       # id -> element -> (first, last)
for line in get(GHCN_INV).splitlines():
    sid = line[0:11]
    if sid in stations:
        el = line[31:35]
        yrs[sid][el] = (int(line[36:40]), int(line[41:45]))

rows = []
for sid, s in stations.items():
    r = dict(s)
    ok = True
    for el in ("TMAX", "TMIN", "PRCP"):
        a, b = yrs[sid].get(el, (None, None))
        r[el + "_first"], r[el + "_last"] = a, b
        # span inside the period (inventory gives first/last year only, gaps must be checked on the data)
        span = 0 if a is None else max(0, min(b, Y1) - max(a, Y0) + 1)
        r[el + "_span_in_period"] = span
    r["usable_span"] = min(r["TMAX_span_in_period"], r["TMIN_span_in_period"], r["PRCP_span_in_period"])
    r["meets_min_years"] = r["usable_span"] >= MIN_YEARS
    rows.append(r)

rows.sort(key=lambda r: -r["usable_span"])
with open("stations_north_GHCN.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---------- ISD ----------
isd = []
for r in csv.DictReader(io.StringIO(get(ISD_HIST))):
    if r["CTRY"] != "AG" or not r["LAT"] or not r["LON"]:
        continue
    lat, lon = float(r["LAT"]), float(r["LON"])
    if not inside(lat, lon):
        continue
    b, e = int(r["BEGIN"][:4]), int(r["END"][:4])
    span = max(0, min(e, Y1) - max(b, Y0) + 1)
    isd.append(dict(USAF=r["USAF"], WBAN=r["WBAN"], name=r["STATION NAME"], lat=lat, lon=lon,
                    elev=r["ELEV(M)"], begin=r["BEGIN"], end=r["END"], span_in_period=span,
                    meets_min_years=span >= MIN_YEARS))
isd.sort(key=lambda r: -r["span_in_period"])
with open("stations_north_ISD.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(isd[0].keys()))
    w.writeheader(); w.writerows(isd)

txt = (f"Domain lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}; period {Y0}-{Y1}; min years {MIN_YEARS}\n"
       f"GHCN stations in domain: {len(rows)}\n"
       f"  with TMAX+TMIN+PRCP spanning >= {MIN_YEARS} y: {sum(r['meets_min_years'] for r in rows)}\n"
       f"ISD stations in domain: {len(isd)}\n"
       f"  spanning >= {MIN_YEARS} y in period: {sum(r['meets_min_years'] for r in isd)}\n"
       "NOTE: inventory spans are first-last year only; actual completeness must be checked on the data.\n")
open("summary.txt", "w").write(txt)
print(txt)
