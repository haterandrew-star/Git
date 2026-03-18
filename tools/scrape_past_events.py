"""
scrape_past_events.py
---------------------
Scrapes all past CCA tournaments from chessaction.com.
Extracts ONLY two columns from each entry list:
  - player_name
  - registered_time

Output: flat CSV  →  tournament_name | player_name | registered_time

Usage:
    # Full run — all CCA past tournaments
    python tools/scrape_past_events.py --out .tmp/all_registrations.csv

    # Test first 10 tournaments
    python tools/scrape_past_events.py --limit 10 --out .tmp/test_registrations.csv

    # Resume after interruption (safe to re-run anytime)
    python tools/scrape_past_events.py --resume --out .tmp/all_registrations.csv

    # All vendors (not just CCA)
    python tools/scrape_past_events.py --vendor 0 --out .tmp/all_registrations.csv
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────

BASE_URL   = "https://chessaction.com"
AJAX_URL   = f"{BASE_URL}/ajaxFrontGetCompletedTourListForAllNew.php"
ENTRY_BASE = "https://onlineregistration.cc/tournaments/advance_entry_list.php"

HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0 Safari/537.36",
    "Referer":          f"{BASE_URL}/pastevents.php",
    "Content-Type":     "application/x-www-form-urlencoded",
    "X-Requested-With": "XMLHttpRequest",
}

PAGE_SIZE       = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3

TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y",
    "%B %d, %Y", "%b %d, %Y",
]

# ─────────────────────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def post_page(s, start: int, vendor: str) -> tuple[list, int]:
    data = {
        "draw": str(start // PAGE_SIZE + 1), "start": str(start),
        "length": str(PAGE_SIZE), "search[value]": "", "search[regex]": "false",
        "columns[0][data]": "html", "columns[0][name]": "",
        "columns[0][searchable]": "true", "columns[0][orderable]": "false",
        "columns[0][search][value]": "", "columns[0][search][regex]": "false",
        "country_search": "0", "state_search": "0",
        "vendor_search": vendor, "keyword": "",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = s.post(AJAX_URL, data=data, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            p = r.json()
            return p.get("aaData", []), p.get("iTotalRecords", 0)
        except Exception as e:
            print(f"    retry {attempt+1}: {e}", file=sys.stderr)
            time.sleep(5)
    return [], 0


def parse_tournament(row: dict) -> dict:
    soup = BeautifulSoup(row.get("html", ""), "html.parser")
    el   = soup.find(["h3", "h4", "h5", "strong", "b"])
    name = el.get_text(strip=True) if el else soup.get_text(" ", strip=True)[:80]
    url  = None
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]tid=([^&\s\"']+)", a["href"])
        if m:
            href = a["href"]
            if not href.startswith("http"):
                href = BASE_URL + "/" + href.lstrip("/")
            url = href if "advance_entry_list" in href else f"{ENTRY_BASE}?tid={m.group(1)}"
            break
    return {"name": name, "url": url}


def build_index(s, vendor: str, limit=None) -> list[dict]:
    index, start, total = [], 0, None
    while True:
        if total and start >= total:
            break
        if limit and len(index) >= limit:
            break
        rows, total = post_page(s, start, vendor)
        if not rows:
            break
        index.extend(parse_tournament(r) for r in rows)
        pct = f"{len(index)/total*100:.0f}%" if total else "?"
        print(f"  Indexed {len(index)}/{total or '?'} ({pct})", end="\r")
        start += PAGE_SIZE
        time.sleep(0.3)
    print()
    return index[:limit] if limit else index


def normalize_ts(raw: str) -> str:
    if not raw or raw.strip() in ("", "0000-00-00 00:00:00", "N/A", "-"):
        return ""
    raw = raw.strip()
    for fmt in TS_FORMATS:
        # Try full string first, then common truncation lengths (for trailing timezone junk)
        for candidate in [raw] + [raw[:n] for n in (19, 10) if len(raw) > n]:
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return raw


def scrape_entries(s, url: str) -> list[tuple[str, str]]:
    """Returns list of (player_name, registered_time) — only these two columns."""
    for attempt in range(MAX_RETRIES):
        try:
            r = s.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return []
            if r.status_code in (403, 429):
                print("\n  Rate-limited — pausing 30s", file=sys.stderr)
                time.sleep(30)
                continue
            r.raise_for_status()
            break
        except requests.exceptions.Timeout:
            time.sleep(5)
        except Exception:
            return []

    soup   = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return []

    # Prefer the table that has a "name" or "player" header — more reliable than largest table
    table = None
    for t in tables:
        header_text = t.find("tr").get_text(strip=True).lower() if t.find("tr") else ""
        if any(k in header_text for k in ["name", "player", "date"]):
            table = t
            break
    if table is None:
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows  = table.find_all("tr")
    if len(rows) < 2:
        return []

    header   = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
    name_col = next((i for i, h in enumerate(header) if any(k in h for k in ["name", "player"])), 1)
    time_col = next((i for i, h in enumerate(header) if any(k in h for k in ["date", "time", "register"])), None)

    entries = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        name = cells[name_col].get_text(strip=True) if name_col < len(cells) else ""
        if not name or name.lower() in ("player name", "name", "#", "no.", ""):
            continue
        ts = normalize_ts(cells[time_col].get_text(strip=True)) if time_col is not None and time_col < len(cells) else ""
        entries.append((name, ts))

    return entries


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Scrape all past CCA tournament registrations → CSV")
    p.add_argument("--vendor",      default="3",   help="3=CCA, 0=all vendors (default: 3)")
    p.add_argument("--out",         default=".tmp/all_registrations.csv")
    p.add_argument("--limit",       type=int,      help="Max tournaments (for testing)")
    p.add_argument("--delay",       type=float, default=1.5)
    p.add_argument("--resume",      action="store_true", help="Append; skip already-done")
    p.add_argument("--index-cache", default=".tmp/past_events_index.json")
    return p.parse_args()


def main():
    args  = parse_args()
    out   = Path(args.out)
    cache = Path(args.index_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    s     = make_session()

    # Build or load index
    if cache.exists():
        print(f"Loading index from {cache}")
        with open(cache) as f:
            index = json.load(f)
        print(f"  {len(index)} tournaments loaded")
    else:
        print(f"Building tournament index (vendor={args.vendor})...")
        index = build_index(s, args.vendor, limit=args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"  Cached → {cache}")

    if args.limit:
        index = index[:args.limit]

    print(f"  {sum(1 for t in index if t.get('url'))}/{len(index)} have entry list URLs\n")

    # Resume
    done = set()
    if args.resume and out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] != "tournament_name":
                    done.add(row[0])
        print(f"Resume: {len(done)} already recorded\n")

    # Scrape → CSV
    mode   = "a" if args.resume and out.exists() else "w"
    total_rows = scraped = skipped = no_url = errors = 0

    with open(out, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["tournament_name", "player_name", "registered_time"])

        for i, t in enumerate(index):
            name = t["name"]
            url  = t.get("url")

            if name in done:
                skipped += 1
                continue
            if not url:
                no_url += 1
                continue

            print(f"[{i+1}/{len(index)}] {name[:65]}")
            try:
                entries = scrape_entries(s, url)
                for pname, ts in entries:
                    writer.writerow([name, pname, ts])
                f.flush()
                total_rows += len(entries)
                scraped    += 1
                print(f"  → {len(entries):,} registrants")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                errors += 1

            time.sleep(args.delay)

    print(f"\n{'='*55}")
    print(f"  Tournaments scraped:  {scraped:,}")
    print(f"  Skipped (resume):     {skipped:,}")
    print(f"  No URL:               {no_url:,}")
    print(f"  Errors:               {errors:,}")
    print(f"  Total rows:           {total_rows:,}")
    print(f"  Output → {out.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
