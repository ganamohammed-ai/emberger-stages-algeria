#!/usr/bin/env python3
"""
Open literature search (no account, no subscription): OpenAlex + Crossref.
Runs the Boolean-style queries below, de-duplicates (DOI / normalised title), pre-flags records that mention
both a place (Algeria / Maghreb / Mediterranean ...) and a bioclimatic/aridity topic, and writes a PRISMA log.

Outputs in results_lit\\ :
  search_log.csv      one row per database x query: date, records retrieved  (the real PRISMA numbers)
  records_all.csv     all unique records (title, year, journal, DOI, cited_by, abstract)
  records_flagged.csv records whose title/abstract match place AND topic terms -> screen these first
  prisma_counts.txt   identified / duplicates removed / flagged
Run by double-clicking RUN_Search_Literature.bat.  Needs internet only.
"""
import os, re, sys, csv, json, time, datetime, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_lit")
UA = "Mozilla/5.0 (literature-screening script)"

QUERIES = [
    "Algeria bioclimatic stages shift climate change",
    "Algeria Emberger quotient climate change",
    "Algeria bioclimatic zones aridification",
    "Algeria isobioclimates ombrotypes",
    "North Africa bioclimatic belts shift",
    "Maghreb bioclimatic stages evolution",
    "étages bioclimatiques Algérie évolution",
    "étages bioclimatiques Algérie changement climatique",
    "quotient pluviothermique Emberger Algérie",
    "Tlemcen bioclimat évolution",
    "Aurès étage bioclimatique changement",
    "Kabylie étages bioclimatiques",
    "Algeria Köppen-Geiger climate zone shift",
    "Algeria rainfall trend rupture Pettitt north",
]
PLACE = re.compile(r"alg[eé]ri|maghreb|north africa|afrique du nord|tell\b|tlemcen|kabylie|aur[eè]s|chellif|cheliff|s[eé]tif|morocc|maroc|tunisi|mediterranean|m[eé]diterran", re.I)
TOPIC = re.compile(r"bioclimat|emberger|ombrotyp|isobioclim|thermotyp|climate zone|climatic zone|k[oö]ppen|aridi|[eé]tage|pluviothermique|rivas", re.I)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = e
            time.sleep(2 * (k + 1))
    raise err


def abstract_from_inverted(inv):
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def clean_doi(d):
    return (d or "").lower().replace("https://doi.org/", "").strip()


def search_openalex(q, pages=3):
    recs = []
    cursor = "*"
    for _ in range(pages):
        url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q) +
               "&per-page=100&cursor=" + urllib.parse.quote(cursor))
        d = get(url)
        for w in d.get("results", []):
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
            recs.append(dict(db="OpenAlex", title=w.get("display_name") or w.get("title") or "", year=w.get("publication_year"),
                             journal=src, doi=clean_doi(w.get("doi")), cited_by=w.get("cited_by_count"),
                             abstract=abstract_from_inverted(w.get("abstract_inverted_index"))))
        cursor = (d.get("meta") or {}).get("next_cursor")
        if not cursor or not d.get("results"):
            break
    return recs


def search_crossref(q, rows=100):
    url = "https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(q) + f"&rows={rows}"
    d = get(url)
    recs = []
    for w in d.get("message", {}).get("items", []):
        t = (w.get("title") or [""])[0]
        y = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        recs.append(dict(db="Crossref", title=t, year=y, journal=(w.get("container-title") or [""])[0],
                         doi=clean_doi(w.get("DOI")), cited_by=w.get("is-referenced-by-count"),
                         abstract=re.sub("<[^>]+>", " ", w.get("abstract") or "")))
    return recs


def dedupe(records):
    seen, out = set(), []
    for r in records:
        key = r["doi"] or norm_title(r["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def flagged(r):
    txt = (r["title"] or "") + " " + (r["abstract"] or "")
    return bool(PLACE.search(txt)) and bool(TOPIC.search(txt))


def main():
    os.makedirs(OUT, exist_ok=True)
    today = datetime.date.today().isoformat()
    log, allrec = [], []
    for q in QUERIES:
        for db, fn in (("OpenAlex", search_openalex), ("Crossref", search_crossref)):
            try:
                r = fn(q)
                print(f"{db:9s} {len(r):4d}  {q}", flush=True)
                log.append(dict(date=today, database=db, query=q, records=len(r), status="ok"))
                for x in r:
                    x["query"] = q
                allrec += r
            except Exception as e:
                print(f"{db:9s} FAILED {q}: {str(e)[:100]}", flush=True)
                log.append(dict(date=today, database=db, query=q, records=0, status="failed: " + str(e)[:80]))
    uniq = dedupe(allrec)
    flg = [r for r in uniq if flagged(r)]
    flg.sort(key=lambda r: (-(r["cited_by"] or 0)))
    cols = ["db", "query", "title", "year", "journal", "doi", "cited_by", "abstract"]
    for name, rows in (("records_all.csv", uniq), ("records_flagged.csv", flg)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(OUT, "search_log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["date", "database", "query", "records", "status"])
        w.writeheader()
        w.writerows(log)
    txt = [f"Search date: {today}", f"Databases: OpenAlex, Crossref ({len(QUERIES)} queries each)",
           f"Records identified (before de-duplication): {len(allrec)}",
           f"Duplicates removed: {len(allrec) - len(uniq)}", f"Unique records: {len(uniq)}",
           f"Flagged for screening (place AND topic terms in title/abstract): {len(flg)}"]
    open(os.path.join(OUT, "prisma_counts.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))
    print("\nDONE. Results in", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        os.makedirs(OUT, exist_ok=True)
        tb = traceback.format_exc()
        open(os.path.join(OUT, "error_log.txt"), "w", encoding="utf-8").write(tb)
        print(tb)
        sys.exit(1)
