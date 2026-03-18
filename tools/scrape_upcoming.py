"""
scrape_upcoming.py
------------------
Scrapes all upcoming CCA tournaments from chessaction.com and enriches
each entry with a live registration count from onlineregistration.cc.

How it works:
  1. POST to chessaction.com/ajaxFrontGetTourListNew.php
     (vendor_search=3, length=-1) -> returns HTML <li class="list-group-item"> elements
  2. Parse each item: name, dates, state/location, reg URL, entry count from badge [N]
     Entry counts come directly from the page HTML — no second HTTP request needed.
  3. Optionally (--with-counts) do the slow two-step advlist scrape for exact per-player count
  4. Compute a sigmoid-based prediction using prior_params from config.json
     or WO2026 priors scaled by expected_total
  5. Inject scraped data into chess_predictor.html as window.STATIC_UPCOMING_DATA
     so the dashboard works even without the API server running

Usage:
    python tools/scrape_upcoming.py
    python tools/scrape_upcoming.py --out .tmp/upcoming_tournaments.json
    python tools/scrape_upcoming.py --no-counts   # skip live entry count fetch
    python tools/scrape_upcoming.py --limit 5     # only process first 5 tournaments

Output:
    .tmp/upcoming_tournaments.json
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)
CONFIG_PATH = ROOT / "config.json"

BASE_URL = "https://chessaction.com"
AJAX_URL = f"{BASE_URL}/ajaxFrontGetTourListNew.php"
REG_BASE = "https://onlineregistration.cc/tournaments"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": BASE_URL + "/index.php",
    "X-Requested-With": "XMLHttpRequest",
}


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_configured_tournament(tid: str, config: dict) -> dict | None:
    for t in config.get("tournaments", []):
        if t["id"] == tid:
            return t
    return None


# ── Sigmoid prediction (standalone, no imports needed) ────────────────────────

def safe_sig(k, t, m):
    x = -k * (t - m)
    if x > 500: return 0.0
    if x < -500: return 1.0
    return 1.0 / (1.0 + math.exp(x))


def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    return L1 * safe_sig(k1, t, m1) + L2 * safe_sig(k2, t, m2)


def compute_prediction(tournament: dict) -> dict | None:
    """Quick sigmoid prediction using prior_params or config data."""
    params = tournament.get("prior_params")
    if not params:
        return None

    first_reg = tournament.get("first_reg_date")
    tourney_date = tournament.get("tournament_date")
    if not first_reg or not tourney_date:
        return None

    try:
        fr = date.fromisoformat(first_reg)
        td = date.fromisoformat(tourney_date)
        t_today = (date.today() - fr).days
        t_final = (td - fr).days
    except ValueError:
        return None

    if t_today < 0:
        t_today = 0

    L1 = params.get("L1", 0)
    k1 = params.get("k1", 0.05)
    m1 = params.get("m1", t_final * 0.25)
    L2 = params.get("L2", params.get("L", 300))
    k2 = params.get("k2", params.get("k", 0.10))
    m2 = params.get("m2", params.get("m", t_final * 0.75))

    predicted_now = double_sigmoid(t_today, L1, k1, m1, L2, k2, m2)
    predicted_final = double_sigmoid(t_final, L1, k1, m1, L2, k2, m2)

    entry_count = tournament.get("entry_count", 0) or 0
    if predicted_now > 0 and entry_count > 0:
        delta_pct = (entry_count - predicted_now) / predicted_now * 100
    else:
        delta_pct = 0.0

    threshold = 10.0
    if delta_pct > threshold:
        status = "GREEN"
    elif delta_pct < -threshold:
        status = "RED"
    else:
        status = "YELLOW"

    rmse = params.get("rmse", 0)
    margin = max(predicted_final * 0.10, 2 * rmse) if rmse else predicted_final * 0.15

    return {
        "t_today": round(t_today, 1),
        "t_final": round(t_final, 1),
        "predicted_now": round(predicted_now, 1),
        "predicted_final": round(predicted_final, 1),
        "ci_low": round(max(0, predicted_final - margin), 1),
        "ci_high": round(predicted_final + margin, 1),
        "delta_pct": round(delta_pct, 1),
        "status": status,
    }


# ── Step 1: fetch upcoming tournament index ────────────────────────────────────

# State abbreviations for location display
STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    "district of columb": "DC", "internet": "Online", "online": "Online",
}


def fetch_tournament_list(session: requests.Session) -> list[dict]:
    """
    POST to ajaxFrontGetTourListNew.php and parse the HTML response.
    The page returns <li class="list-group-item"> elements — one per tournament.
    Entry counts are embedded in button text as "Entry List [N]" — no second request needed.
    """
    print("Fetching upcoming tournament list from chessaction.com...")
    try:
        resp = session.post(
            AJAX_URL,
            data={"vendor_search": "3", "length": "-1"},
            headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Failed to fetch tournament list: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.find_all("li", class_="list-group-item")

    tournaments = []
    for item in items:
        t = parse_list_item(item)
        if t:
            tournaments.append(t)

    print(f"Found {len(tournaments)} upcoming tournaments")
    return tournaments


def parse_list_item(item) -> dict | None:
    """
    Parse a <li class="list-group-item"> element into a tournament dict.

    Structure:
      <a href="tournaments/index.php?view=zNTizdLa&tid=...">Name</a>
      <div class="text-muted small">Mar 13, 2026 - Mar 15, 2026 ... State: Florida</div>
      <a href="tournaments/advance_entry_list.php?tid=...">Entry List [N]</a>
    """
    # Tournament name: the view=zNTizdLa link
    name_tag = item.find("a", href=lambda h: h and "view=zNTi" in h)
    if not name_tag:
        return None
    name = name_tag.get_text(strip=True)
    if not name:
        return None

    # Filter out CCA admin placeholder entries (year 2099)
    date_div = item.find("div", class_="text-muted small")
    date_text = date_div.get_text(" ", strip=True) if date_div else ""
    if "2099" in date_text:
        return None

    # Parse dates + state
    tournament_date, end_date = parse_dates(date_text)
    state_match = re.search(r"State:\s*([^\|]+?)(?:\s*&|$)", date_text, re.IGNORECASE)
    state_raw = state_match.group(1).strip() if state_match else ""
    state_abbr = STATE_MAP.get(state_raw.lower(), state_raw)
    location = state_abbr  # e.g. "FL", "DC", "Online"

    # Registration URL + TID + entry count from the "advance_entry_list" link
    reg_url = None
    tid = None
    entry_count = None
    for a in item.find_all("a", href=True):
        if "advance_entry_list" in a["href"]:
            reg_url = f"https://onlineregistration.cc/{a['href']}"
            m = re.search(r"tid=([^&\s]+)", a["href"])
            if m:
                tid = m.group(1)
            # Count is in the button text: "Entry List [85]"
            btn = a.find("button")
            btn_text = btn.get_text(strip=True) if btn else a.get_text(strip=True)
            count_m = re.search(r"\[(\d+)\]", btn_text)
            if count_m:
                entry_count = int(count_m.group(1))
            break

    return {
        "name": name.strip(),
        "tournament_date": tournament_date,
        "end_date": end_date,
        "location": location,
        "registration_url": reg_url,
        "tid_raw": tid,
        "entry_count_scraped": entry_count,  # from page HTML — fast, no extra request
        "expected_total": None,
        "raw_dates": date_text,
    }


def parse_dates(text: str) -> tuple[str | None, str | None]:
    """
    Parse date text into start/end ISO dates.
    Handles formats like: "July 2-6, 2026" / "Jul 2 - Jul 6, 2026" / "2026-07-02"
    """
    if not text:
        return None, None

    text = text.strip()

    # Already ISO format
    m = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1), None

    # Month Day-Day, Year (e.g. "July 2-6, 2026" or "July 2 - 6, 2026")
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
    }

    # "Month Start-End, Year"
    m = re.search(
        r"([A-Za-z]+)\s+(\d{1,2})\s*[-–]\s*(\d{1,2}),?\s*(\d{4})",
        text, re.IGNORECASE
    )
    if m:
        mon_str, day_start, day_end, year = m.groups()
        mon = months.get(mon_str.lower())
        if mon:
            try:
                start = date(int(year), mon, int(day_start))
                end = date(int(year), mon, int(day_end))
                return start.isoformat(), end.isoformat()
            except ValueError:
                pass

    # "Month Day, Year"
    m = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})",
        text, re.IGNORECASE
    )
    if m:
        mon_str, day, year = m.groups()
        mon = months.get(mon_str.lower())
        if mon:
            try:
                d = date(int(year), mon, int(day))
                return d.isoformat(), None
            except ValueError:
                pass

    # "MM/DD/YYYY"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        mo, dy, yr = m.groups()
        try:
            d = date(int(yr), int(mo), int(dy))
            return d.isoformat(), None
        except ValueError:
            pass

    return None, None


# ── Step 2: fetch live entry count via two-step advlist pattern ────────────────

def get_live_entry_count(reg_url: str, session: requests.Session) -> int:
    """
    Follow the two-step advlist pattern to count current entries.
    Returns -1 on error, 0 if no entries yet.
    """
    try:
        resp = session.get(reg_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        # Parse JS for advlist static URL
        match = re.search(r"advlists/([^'\"]+_alp_n\.html)", resp.text)
        if not match:
            return 0  # No advlist -> tournament may not be open for registration

        advlist_path = match.group(0)
        advlist_url = f"{REG_BASE}/{advlist_path}"

        time.sleep(0.5)  # polite delay
        resp2 = session.get(advlist_url, headers=HEADERS, timeout=15)
        resp2.raise_for_status()

        soup = BeautifulSoup(resp2.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return 0

        main_table = max(tables, key=lambda t: len(t.find_all("tr")))
        rows = main_table.find_all("tr")
        count = 0
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            text = row.get_text(strip=True).lower()
            if "player name" in text or text in ("name", "#"):
                continue
            if any(c.get_text(strip=True) for c in cells):
                count += 1
        return count

    except Exception as e:
        return -1


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(args):
    config = load_config()
    session = requests.Session()
    session.headers.update(HEADERS)

    # Fetch tournament list
    tournaments = fetch_tournament_list(session)

    if not tournaments:
        print("No upcoming tournaments found.")
        # Save empty result
        out = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": 0,
            "tournaments": [],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return

    if args.limit:
        print(f"Limiting to first {args.limit} tournaments")
        tournaments = tournaments[:args.limit]

    # Enrich each tournament
    enriched = []
    today = date.today()

    for i, t in enumerate(tournaments):
        name = t["name"]
        print(f"\n[{i+1}/{len(tournaments)}] {name}")

        # Skip past tournaments
        if t.get("tournament_date"):
            try:
                td = date.fromisoformat(t["tournament_date"])
                if td < today:
                    print(f"  Skipping (past: {td})")
                    continue
            except ValueError:
                pass

        # Entry count: use count scraped from page HTML (fast, already fetched).
        # If --with-counts is set, do a slower two-step advlist scrape for exact per-player count.
        entry_count = t.get("entry_count_scraped")
        if getattr(args, "with_counts", False) and t.get("registration_url"):
            print(f"  Fetching detailed entry count from {t['registration_url'][:60]}...")
            detailed = get_live_entry_count(t["registration_url"], session)
            if detailed >= 0:
                entry_count = detailed
                print(f"  Entry count (detailed): {entry_count}")
            else:
                print(f"  Entry count (detailed): error — using page count {entry_count}")
            time.sleep(1.0)  # polite delay between requests
        else:
            if entry_count is not None:
                print(f"  Entry count (page): {entry_count}")

        # Pull prior_params from config if this tournament is configured
        prior_params = None
        configured = None
        if t.get("tid_raw"):
            configured = get_configured_tournament(t["tid_raw"], config)
        if not configured:
            # Try matching by name
            for cfg_t in config.get("tournaments", []):
                if cfg_t["name"].lower() in name.lower() or name.lower() in cfg_t["name"].lower():
                    configured = cfg_t
                    break

        if configured:
            prior_params = configured.get("prior_params")
            if not t.get("tournament_date") and configured.get("tournament_date"):
                t["tournament_date"] = configured["tournament_date"]
            if not t.get("expected_total") and configured.get("expected_total"):
                t["expected_total"] = configured["expected_total"]
            if not t.get("registration_url") and configured.get("registration_url"):
                t["registration_url"] = configured["registration_url"]

        result = {
            "id": t.get("tid_raw") or re.sub(r"[^A-Z0-9]", "", name.upper())[:10],
            "name": name,
            "tournament_date": t.get("tournament_date"),
            "end_date": t.get("end_date"),
            "location": t.get("location"),
            "registration_url": t.get("registration_url"),
            "entry_count": entry_count,
            "expected_total": t.get("expected_total"),
            "prior_params": prior_params,
            "raw_dates": t.get("raw_dates"),
        }

        # Compute prediction if we have params
        if prior_params and t.get("tournament_date"):
            result_with_pred = dict(result)
            result_with_pred["entry_count"] = entry_count or 0
            # Inject first_reg_date for prediction (estimate as 6 months before tournament)
            if configured and configured.get("first_reg_date"):
                result_with_pred["first_reg_date"] = configured["first_reg_date"]
            else:
                try:
                    td = date.fromisoformat(t["tournament_date"])
                    from datetime import timedelta
                    est_first_reg = td - timedelta(days=180)
                    result_with_pred["first_reg_date"] = est_first_reg.isoformat()
                except Exception:
                    pass
            pred = compute_prediction(result_with_pred)
            result["prediction"] = pred
        else:
            result["prediction"] = None

        enriched.append(result)
        print(f"  OK {name} | date={result['tournament_date']} | entries={entry_count}")

    # Merge all "World Open" sub-events into a single aggregated entry
    wo_sections = [t for t in enriched if "world open" in t["name"].lower()]
    non_wo = [t for t in enriched if "world open" not in t["name"].lower()]
    if wo_sections:
        # Find the configured WO2026 entry for prior_params / expected_total
        wo_config = next((c for c in config.get("tournaments", [])
                          if c.get("tournament_family") == "world_open" and c.get("active")), None)
        total_entries = sum((t.get("entry_count") or 0) for t in wo_sections)
        # Use earliest start date as tournament_date (earliest section start)
        dates = [t["tournament_date"] for t in wo_sections if t.get("tournament_date")]
        main_date = min(dates) if dates else None
        # Primary registration URL = top 6 sections (largest)
        primary_url = next(
            (t["registration_url"] for t in wo_sections if "top 6" in t["name"].lower()), None
        ) or (wo_sections[0]["registration_url"] if wo_sections else None)
        merged_wo = {
            "id": "WO2026",
            "name": "2026 World Open",
            "tournament_date": main_date,
            "end_date": None,
            "location": "DC",
            "registration_url": primary_url,
            "entry_count": total_entries,
            "expected_total": wo_config.get("expected_total", 1100) if wo_config else 1100,
            "prior_params": wo_config.get("prior_params") if wo_config else None,
            "sections": [
                {"label": t["name"].replace("2026 World Open", "").strip(" ,"),
                 "entry_count": t.get("entry_count") or 0,
                 "registration_url": t.get("registration_url")}
                for t in sorted(wo_sections, key=lambda x: -(x.get("entry_count") or 0))
            ],
            "raw_dates": None,
        }
        # Compute prediction for merged entry
        if merged_wo["prior_params"] and wo_config:
            pred_input = dict(merged_wo)
            pred_input["first_reg_date"] = wo_config.get("first_reg_date", "2026-02-05")
            merged_wo["prediction"] = compute_prediction(pred_input)
        else:
            merged_wo["prediction"] = None
        enriched = non_wo + [merged_wo]
        print(f"\n  [merge] Combined {len(wo_sections)} World Open sections → {total_entries} total entries")

    # Sort by tournament_date
    def sort_key(t):
        return t.get("tournament_date") or "9999-99-99"
    enriched.sort(key=sort_key)

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(enriched),
        "tournaments": enriched,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Inject scraped data into chess_predictor.html for static (no-API) mode.
    # Replaces the line after the SCRAPER_INJECT_UPCOMING_DATA comment.
    html_path = ROOT / "chess_predictor.html"
    if html_path.exists():
        try:
            inject_static_data_into_html(html_path, out)
        except Exception as e:
            print(f"  [warn] Could not update HTML static data: {e}")

    print(f"\n{'='*60}")
    print(f"Saved {len(enriched)} upcoming tournaments -> {out_path}")
    print(f"{'='*60}")
    for t in enriched[:10]:
        pred_str = ""
        if t.get("prediction"):
            pred_str = f"  -> predicted {t['prediction']['predicted_final']:.0f}"
        entries_str = f"  | {t['entry_count']} entries" if t.get("entry_count") is not None else ""
        print(f"  {t['name'][:50]:50s} {t.get('tournament_date', '?')}{entries_str}{pred_str}")
    if len(enriched) > 10:
        print(f"  ... and {len(enriched) - 10} more")


def inject_static_data_into_html(html_path: Path, data: dict):
    """
    Replace the STATIC_UPCOMING_DATA line in chess_predictor.html with fresh scraped data.
    Looks for the line starting with 'window.STATIC_UPCOMING_DATA =' and replaces it.
    """
    import re as _re
    content = html_path.read_text(encoding="utf-8")
    # Build the compact JSON payload (no raw_dates, keeps file size reasonable)
    slim = {
        "generated_at": data["generated_at"],
        "count": data["count"],
        "source": "static",
        "tournaments": [
            {k: v for k, v in t.items() if k != "raw_dates"}
            for t in data["tournaments"]
        ],
    }
    json_str = json.dumps(slim, separators=(",", ":"))
    new_line = f"window.STATIC_UPCOMING_DATA = {json_str};"
    updated = _re.sub(
        r"^window\.STATIC_UPCOMING_DATA\s*=.*?;$",
        new_line,
        content,
        flags=_re.MULTILINE,
    )
    if updated != content:
        html_path.write_text(updated, encoding="utf-8")
        print(f"  Updated HTML static data ({len(slim['tournaments'])} tournaments)")
    else:
        print("  [warn] HTML STATIC_UPCOMING_DATA line not found — skipping injection")


def main():
    parser = argparse.ArgumentParser(description="Scrape upcoming CCA tournaments")
    parser.add_argument("--out", default=".tmp/upcoming_tournaments.json")
    parser.add_argument("--no-counts", action="store_true",
                        help="Deprecated flag kept for compatibility (counts now come from page HTML)")
    parser.add_argument("--with-counts", action="store_true",
                        help="Do slow two-step advlist scrape for exact per-player count (vs. page badge)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N tournaments")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
