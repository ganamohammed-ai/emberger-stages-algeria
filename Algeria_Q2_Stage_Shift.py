#!/usr/bin/env python3
"""
Emberger Q2 and bioclimatic-stage shift from GHCN-Daily stations, northern Algeria.

Needs: stations_north_GHCN.csv (made by Algeria_North_Station_Inventory.py) in the same folder.
Run by double-clicking RUN_Q2_Stage_Shift.bat. Standard library only (no pip installs).

Method
  Q2 = 2000 * P / (M^2 - m^2)      (M, m in Kelvin)
  P  = mean annual precipitation (mm)
  M  = mean daily maximum of the hottest month (mean of monthly means over valid years), degC
  m  = mean daily minimum of the coldest month, degC
  A year is valid for P only if all 12 months have >= MIN_DAYS_PRCP valid PRCP days (monthly total scaled to full month).
  A month-year is valid for M/m if it has >= MIN_DAYS_TEMP valid TMAX (or TMIN) days.
  Days with a non-empty GHCN quality flag are discarded.

Outputs (folder 'results'):
  station_Q2_by_period.csv   one row per station, columns per period
  stage_shift_summary.txt    counts of stations that changed stage
  log.txt                    download / data-quality notes
"""
import csv, os, sys, time, math, collections, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS_CSV = os.path.join(HERE, "stations_north_GHCN.csv")
CACHE = os.path.join(HERE, "ghcn_daily_cache")
OUT = os.path.join(HERE, "results")

LAT_MIN_STUDY = 33.0           # exclude Saharan stations south of this latitude
MIN_DAYS_TEMP = 15             # min valid days in a month for TMAX/TMIN monthly mean
MIN_DAYS_PRCP = 20             # min valid days in a month for PRCP (monthly total is scaled to full month)
MONTH_DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# period name: (first year, last year, minimum valid years)
PERIODS = {
    "OLD_1913_1938": (1913, 1938, 8),
    "RECENT_1985_2025": (1985, 2025, 15),
}
# Reference-period comparison pairs (old, new)
COMPARE = ("OLD_1913_1938", "RECENT_1985_2025")

# ---- Emberger stage limits on Q2 (EDIT to match your reference, e.g. Emberger 1955 / Stewart 1969) ----
STAGES = [(20, "aride"), (40, "semi-aride"), (70, "subhumide"), (100, "humide"), (1e9, "per-humide")]
# ---- winter variants on m (degC) (EDIT if needed) ----
WINTER = [(0, "hiver froid"), (3, "hiver frais"), (7, "hiver tempere"), (1e9, "hiver chaud")]

URLS = [  # tried in this order; first one that works is used
    ("dly", "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"),
    ("dly", "https://ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly"),
    ("bystation.gz", "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{}.csv.gz"),
    ("bystation", "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{}.csv"),
    ("access", "https://www.ncei.noaa.gov/data/daily-summaries/access/{}.csv"),
]


def stage_of(q2):
    for lim, name in STAGES:
        if q2 < lim:
            return name


def winter_of(m):
    for lim, name in WINTER:
        if m < lim:
            return name


def download(sid, log):
    """Return (path, kind) or (None, None)."""
    os.makedirs(CACHE, exist_ok=True)
    for kind, url in URLS:
        p = os.path.join(CACHE, f"{sid}.{kind}")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p, kind
    for kind, url in URLS:
        u = url.format(sid)
        try:
            data = urllib.request.urlopen(u, timeout=180).read()
            if len(data) < 1000:
                raise ValueError("file too small")
            p = os.path.join(CACHE, f"{sid}.{kind}")
            open(p, "wb").write(data)
            return p, kind
        except Exception as e:
            log.append(f"{sid}: {u} -> {e}")
    return None, None


def num(s):
    try:
        return float(s)
    except Exception:
        return None


def _add(raw, y, m, el, v):
    raw[(y, m)][el].append(v)


def read_dly(path):
    raw = collections.defaultdict(lambda: {"TMAX": [], "TMIN": [], "PRCP": []})
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) < 21:
                continue
            el = line[17:21]
            if el not in ("TMAX", "TMIN", "PRCP"):
                continue
            y, m = int(line[11:15]), int(line[15:17])
            for d in range(31):
                base = 21 + d * 8
                v = line[base:base + 5]
                qf = line[base + 6:base + 7]
                if not v.strip():
                    continue
                v = int(v)
                if v == -9999 or qf.strip():
                    continue
                _add(raw, y, m, el, v / 10.0)
    return raw


def read_bystation(path, gz):
    import gzip
    raw = collections.defaultdict(lambda: {"TMAX": [], "TMIN": [], "PRCP": []})
    op = gzip.open if gz else open
    with op(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.reader(f):
            if len(r) < 4 or r[2] not in ("TMAX", "TMIN", "PRCP"):
                continue
            if len(r) > 5 and r[5].strip():       # quality flag
                continue
            d = r[1]
            try:
                _add(raw, int(d[:4]), int(d[4:6]), r[2], int(r[3]) / 10.0)
            except ValueError:
                continue
    return raw


def read_access(path):
    raw = collections.defaultdict(lambda: {"TMAX": [], "TMIN": [], "PRCP": []})
    vals = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            d = r.get("DATE", "")
            if len(d) < 7:
                continue
            y, m = int(d[:4]), int(d[5:7])
            for el in ("TMAX", "TMIN", "PRCP"):
                v = num(r.get(el, ""))
                a = r.get(el + "_ATTRIBUTES", "")
                if v is None or (a and len(a.split(",")) > 1 and a.split(",")[1].strip()):
                    continue
                _add(raw, y, m, el, v)
                if el == "TMAX":
                    vals.append(v)
    if vals:
        vals.sort()
        if abs(vals[len(vals) // 2]) > 60:     # tenths -> physical units
            for k in raw:
                for el in raw[k]:
                    raw[k][el] = [x / 10.0 for x in raw[k][el]]
    return raw


def read_station(path, kind):
    if kind == "dly":
        return read_dly(path), 1.0
    if kind == "bystation.gz":
        return read_bystation(path, True), 1.0
    if kind == "bystation":
        return read_bystation(path, False), 1.0
    return read_access(path), 1.0


def mean(x):
    return sum(x) / len(x)


def period_stats(raw, scale_t, y0, y1, min_years):
    # P: valid years
    annualP = {}
    for y in range(y0, y1 + 1):
        tot, ok = 0.0, True
        for m in range(1, 13):
            p = raw.get((y, m), {}).get("PRCP", [])
            if len(p) < MIN_DAYS_PRCP:
                ok = False
                break
            tot += sum(p) / len(p) * MONTH_DAYS[m - 1]
        if ok:
            annualP[y] = tot
    # monthly mean TMAX / TMIN per month across years
    tmax_m = collections.defaultdict(list)
    tmin_m = collections.defaultdict(list)
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            d = raw.get((y, m))
            if not d:
                continue
            if len(d["TMAX"]) >= MIN_DAYS_TEMP:
                tmax_m[m].append(mean(d["TMAX"]) * scale_t)
            if len(d["TMIN"]) >= MIN_DAYS_TEMP:
                tmin_m[m].append(mean(d["TMIN"]) * scale_t)
    if len(annualP) < min_years:
        return None
    mm_max = {m: mean(v) for m, v in tmax_m.items() if len(v) >= min_years}
    mm_min = {m: mean(v) for m, v in tmin_m.items() if len(v) >= min_years}
    if len(mm_max) < 12 or len(mm_min) < 12:
        return None
    P = mean(list(annualP.values()))
    M = max(mm_max.values())
    m_ = min(mm_min.values())
    denom = (M + 273.15) ** 2 - (m_ + 273.15) ** 2
    if denom <= 0:
        return None
    q2 = 2000.0 * P / denom
    return dict(n_years=len(annualP), P=round(P, 1), M=round(M, 2), m=round(m_, 2), Q2=round(q2, 1),
                stage=stage_of(q2), winter=winter_of(m_))


# Old (<=1938) station -> modern station of the same locality.  Same id twice = one continuous station.
# (Pairing by locality: different instrument/site/elevation, see elev_diff_m column.)
PAIRS = [
    ("AGE00147705", "AG000060390"),   # Alger Ville/Universite -> Alger Dar El Beida
    ("AGE00147711", "AGM00060419"),   # Constantine -> Mohamed Boudiaf
    ("AGE00147713", "AGM00060445"),   # Setif -> Setif Ain Arnat
    ("AGE00147715", "AGM00060475"),   # Tebessa -> Cheikh Larbi Tebessi
    ("AGE00147712", "AGM00060425"),   # Orleansville -> Ech Cheliff
    ("AGE00147717", "AGM00060520"),   # Sidi-Bel-Abbes (old) -> Sidi-Bel-Abbes (modern)
    ("AGE00135039", "AGM00060490"),   # Oran Hopital Militaire -> Es Senia
    ("AGE00147714", "AGM00060490"),   # Oran Cap Falcon -> Es Senia
    ("AGE00147704", "AGM00060360"),   # Annaba Cap de Garde -> Annaba
    ("AGE00147720", "AGM00060550"),   # Geryville -> El-Bayadh
    ("AGE00147710", "AGE00147710"),   # Bejaia (continuous)
    ("AGE00147708", "AGE00147708"),   # Tizi Ouzou (continuous)
    ("AGE00147716", "AGE00147716"),   # Nemours/Ghazaouet (continuous)
    ("AGE00147718", "AGE00147718"),   # Biskra (continuous)
    ("AGE00147719", "AGE00147719"),   # Laghouat (continuous)
]


def main():
    if not os.path.exists(STATIONS_CSV):
        print("stations_north_GHCN.csv not found. Run RUN_Inventory.bat first.")
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    log = []
    sts = list(csv.DictReader(open(STATIONS_CSV, encoding="utf-8")))
    sel = []
    for s in sts:
        if float(s["lat"]) < LAT_MIN_STUDY:
            continue
        first = int(s["TMAX_first"]) if s["TMAX_first"] else 9999
        last = int(s["TMAX_last"]) if s["TMAX_last"] else 0
        modern = s["meets_min_years"] == "True"
        old = first <= 1925 and last >= 1930
        if modern or old:
            sel.append(s)
    print(f"{len(sel)} stations selected (lat >= {LAT_MIN_STUDY}).")

    rows = []
    for i, s in enumerate(sel, 1):
        sid = s["id"]
        print(f"[{i}/{len(sel)}] {sid} {s['name']}")
        path, kind = download(sid, log)
        if not path:
            log.append(f"{sid}: skipped (no download)")
            continue
        try:
            raw, scale_t = read_station(path, kind)
        except Exception as e:
            log.append(f"{sid}: read error {e}")
            continue
        row = dict(id=sid, name=s["name"], lat=s["lat"], lon=s["lon"], elev=s["elev"], file_format=kind)
        for pname, (y0, y1, miny) in PERIODS.items():
            st = period_stats(raw, scale_t, y0, y1, miny)
            for k in ("n_years", "P", "M", "m", "Q2", "stage", "winter"):
                row[f"{pname}__{k}"] = st[k] if st else ""
        rows.append(row)

    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "station_Q2_by_period.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- paired comparison old vs modern ----
    by = {r["id"]: r for r in rows}
    a, b = COMPARE
    prow = []
    for oid, nid in PAIRS:
        o, n = by.get(oid), by.get(nid)
        if not o or not n:
            continue
        qa, qb = o[f"{a}__Q2"], n[f"{b}__Q2"]
        if qa == "" or qb == "":
            continue
        pr = dict(old_id=oid, old_name=o["name"], new_id=nid, new_name=n["name"],
                  continuous=(oid == nid), elev_old=o["elev"], elev_new=n["elev"],
                  elev_diff_m=round(float(n["elev"]) - float(o["elev"]), 0),
                  lat=n["lat"], lon=n["lon"],
                  yrs_old=o[f"{a}__n_years"], yrs_new=n[f"{b}__n_years"],
                  P_old=o[f"{a}__P"], P_new=n[f"{b}__P"], M_old=o[f"{a}__M"], M_new=n[f"{b}__M"],
                  m_old=o[f"{a}__m"], m_new=n[f"{b}__m"], Q2_old=qa, Q2_new=qb,
                  stage_old=o[f"{a}__stage"], stage_new=n[f"{b}__stage"],
                  winter_old=o[f"{a}__winter"], winter_new=n[f"{b}__winter"],
                  dQ2=round(qb - qa, 1), dQ2_pct=round(100 * (qb - qa) / qa, 1),
                  dM=round(n[f"{b}__M"] - o[f"{a}__M"], 2), dm=round(n[f"{b}__m"] - o[f"{a}__m"], 2),
                  dP_pct=round(100 * (n[f"{b}__P"] - o[f"{a}__P"]) / o[f"{a}__P"], 1))
        pr["stage_shift"] = pr["stage_old"] != pr["stage_new"]
        prow.append(pr)
    if prow:
        with open(os.path.join(OUT, "station_pairs_shift.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prow[0].keys()))
            w.writeheader()
            w.writerows(prow)

    nvalid_old = sum(1 for r in rows if r[f"{a}__Q2"] != "")
    nvalid_new = sum(1 for r in rows if r[f"{b}__Q2"] != "")
    lines = [f"Stations processed: {len(rows)}",
             f"Stations with a valid {a}: {nvalid_old};  with a valid {b}: {nvalid_new}",
             f"Old/modern pairs with both periods valid: {len(prow)}",
             f"Pairs whose stage changed: {sum(1 for p in prow if p['stage_shift'])}"]
    if prow:
        lines.append(f"Mean dQ2 % : {mean([p['dQ2_pct'] for p in prow]):.1f}   mean dM: {mean([p['dM'] for p in prow]):.2f} degC   mean dm: {mean([p['dm'] for p in prow]):.2f} degC   mean dP %: {mean([p['dP_pct'] for p in prow]):.1f}")
    lines.append("")
    for p in prow:
        lines.append(f"  {p['new_name'][:26]:26s} {p['stage_old']:>10s} -> {p['stage_new']:<10s} Q2 {p['Q2_old']:>6} -> {p['Q2_new']:<6} dP% {p['dP_pct']:>6} dM {p['dM']:>5} dm {p['dm']:>5} elev diff {p['elev_diff_m']:.0f} m {'(continuous)' if p['continuous'] else ''}")
    lines.append("")
    lines.append("NOTE: stage limits in the script are editable defaults; confirm against your reference.")
    lines.append("NOTE: old vs modern stations of one locality differ in site/instrument/elevation; treat single-station shifts with caution.")
    open(os.path.join(OUT, "stage_shift_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8").write("\n".join(log))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
