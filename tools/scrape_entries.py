"""
scrape_entries.py
-----------------
Scrapes current advance entry list from onlineregistration.cc (CCA registration portal).

Usage:
    python tools/scrape_entries.py --url "https://onlineregistration.cc/tournaments/advance_entry_list.php?tid=XXXX" --out .tmp/entries.json
    python tools/scrape_entries.py --config config.json  # reads url + out from config

Output: JSON array of {"name": str, "registered_time": str|null}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

REQUEST_TIMEOUT = 20   # seconds
RETRY_WAIT = 5         # seconds between retries
MAX_RETRIES = 2


# ── Core scraper ─────────────────────────────────────────────────────────────

class CCAEntryScraper:
    """
    Scrapes the advance entry list from onlineregistration.cc.

    The target page renders a plain HTML table. Columns observed:
        #  |  Player Name  |  Rating  |  Section  |  Date Registered

    Column positions can shift — the scraper searches by header text.
    """

    def __init__(self, url: str, verbose: bool = False):
        self.url = url
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[scrape] {msg}", file=sys.stderr)

    def _fetch(self) -> str:
        """Fetch raw HTML with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._log(f"GET {self.url} (attempt {attempt})")
                resp = self.session.get(self.url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.text
            except requests.exceptions.Timeout:
                self._log(f"Timeout on attempt {attempt}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT)
                else:
                    raise
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response else "?"
                if code in (403, 429):
                    print(f"ERROR: Server returned {code}. Scraping blocked.", file=sys.stderr)
                    sys.exit(1)
                raise

    def _find_column_indices(self, header_row) -> dict:
        """
        Map column names to their indices by inspecting the <th> or first <td> row.
        Returns dict with keys: 'name', 'registered_time' (values are int indices or None).
        """
        cells = header_row.find_all(["th", "td"])
        cols = {cell.get_text(strip=True).lower(): i for i, cell in enumerate(cells)}

        name_idx = None
        time_idx = None

        # Name column — try several common labels
        for label in ["player name", "name", "player", "full name"]:
            if label in cols:
                name_idx = cols[label]
                break

        # Registration time column
        for label in ["date registered", "registered", "registration date", "date", "time"]:
            if label in cols:
                time_idx = cols[label]
                break

        return {"name": name_idx, "time": time_idx}

    def _resolve_advlist_url(self, html: str) -> str | None:
        """
        onlineregistration.cc uses a two-step pattern:
          1. advance_entry_list.php contains JS that loads a static advlist HTML file
          2. The static file has the actual player table at advlists/CCA/<CODE>/<CODE>_alp_n.html

        Parse the PHP page HTML/JS to find the advlist path, return the full URL.
        Returns None if pattern not found (non-CCA page or changed structure).
        """
        import re
        match = re.search(r"advlists/([^'\"]+_alp_n\.html)", html)
        if not match:
            return None
        advlist_path = match.group(0)
        base = "https://onlineregistration.cc/tournaments"
        return f"{base}/{advlist_path}"

    def scrape(self) -> list[dict]:
        """
        Fetch the page and extract entries.

        For onlineregistration.cc tournaments this follows the two-step pattern:
          1. Fetch advance_entry_list.php, extract advlist static URL from JS
          2. Fetch the static advlist HTML and parse the player table

        Returns:
            List of dicts: [{"name": str, "registered_time": str|None}, ...]
        """
        html = self._fetch()

        # Follow the two-step advlist pattern for onlineregistration.cc
        advlist_url = self._resolve_advlist_url(html)
        if advlist_url:
            self._log(f"Following advlist pattern → {advlist_url}")
            try:
                import time as _time
                _time.sleep(1)  # polite delay
                resp = self.session.get(advlist_url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                html = resp.text
                self._log(f"Loaded advlist ({len(html)} bytes)")
            except Exception as e:
                self._log(f"Failed to fetch advlist: {e}  — falling back to PHP page")
        else:
            self._log("No advlist URL found — trying direct table parse")

        soup = BeautifulSoup(html, "html.parser")

        # Find the main data table — look for the largest table on the page
        tables = soup.find_all("table")
        if not tables:
            print("ERROR: No <table> found on page. Page structure may have changed.", file=sys.stderr)
            sys.exit(1)

        # Pick the table with the most rows
        main_table = max(tables, key=lambda t: len(t.find_all("tr")))
        rows = main_table.find_all("tr")

        if len(rows) < 2:
            self._log("Table has fewer than 2 rows — no data.")
            return []

        # Detect column positions from header row
        header_row = rows[0]
        col_idx = self._find_column_indices(header_row)

        if col_idx["name"] is None:
            self._log("Could not detect 'name' column — falling back to column 1")
            col_idx["name"] = 1  # Common fallback: #, Name, Rating, Section, Date

        if col_idx["time"] is None:
            self._log("Could not detect 'registered_time' column — timestamps will be null")

        # Parse data rows
        entries = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # Skip sub-header rows (section headers repeat column names)
            row_text = row.get_text(strip=True).lower()
            if "player name" in row_text or "name" == row_text:
                continue

            name = None
            reg_time = None

            if col_idx["name"] < len(cells):
                name = cells[col_idx["name"]].get_text(strip=True)

            if col_idx["time"] is not None and col_idx["time"] < len(cells):
                raw_time = cells[col_idx["time"]].get_text(strip=True)
                reg_time = self._parse_datetime(raw_time)

            if name and name not in ("", "#", "Player Name", "Name"):
                entries.append({"name": name, "registered_time": reg_time})

        self._log(f"Scraped {len(entries)} entries")
        return entries

    def _parse_datetime(self, raw: str) -> str | None:
        """
        Normalize a date/datetime string to ISO 8601.
        Handles common formats seen on CCA registration pages.
        Returns None if unparseable.
        """
        if not raw or raw == "0000-00-00 00:00:00":
            return None

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt).isoformat()
            except ValueError:
                continue
        return raw  # return raw if all formats fail


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape CCA tournament entry list")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Direct URL to advance entry list page")
    group.add_argument("--config", help="Path to JSON config file with 'url' key")
    parser.add_argument("--out", default=None, help="Output JSON file path (default: stdout)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    url = args.url
    out_path = args.out

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        url = cfg["url"]
        out_path = out_path or cfg.get("out")

    # Ensure output directory exists
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    scraper = CCAEntryScraper(url=url, verbose=args.verbose)
    entries = scraper.scrape()

    result = {
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "url": url,
        "count": len(entries),
        "entries": entries,
    }

    output = json.dumps(result, indent=2, ensure_ascii=False)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {len(entries)} entries to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
