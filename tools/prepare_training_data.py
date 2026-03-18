"""
prepare_training_data.py
------------------------
Normalize raw CCA tournament CSV files into standard cumulative
[[t, count], ...] format for model training and ensemble fitting.

Handles:
  - Chicago Open 2022: PlayerFullName, RegisteredTime (some 0000-00-00 entries)
  - North American Open 2024: PlayerFullName, RegisteredTime, ...
  - World Open 2026: PlayerFullName, RegisteredTime (sparse)

The t-axis is calendar days from first valid registration date.

Outputs (to .tmp/):
  cumulative_CHI2022.json   — [[t, cumulative_count], ...]
  cumulative_NAO2024.json
  training_meta.json        — tournament metadata for ensemble

Usage:
    python tools/prepare_training_data.py
    python tools/prepare_training_data.py --tournament chicago
    python tools/prepare_training_data.py --out-dir .tmp
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path


ROOT = Path(__file__).parent.parent
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "chicago": {
        "id": "CHI2022",
        "name": "Chicago Open 2022",
        "csv": r"c:\Users\eliza\Downloads\Chicago Open data for modeling.csv",
        "date_col": "RegisteredTime",
        "date_formats": ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"],
        "tournament_date": "2022-05-27",
        "expected_total": 943,
        "has_early_bird": True,
        "fee_deadlines": [
            {"date": "2022-03-15", "label": "Early Bird"},
            {"date": "2022-05-15", "label": "Late Fee"},
        ],
    },
    "nao2024": {
        "id": "NAO2024",
        "name": "North American Open 2024",
        "csv": r"c:\Users\eliza\OneDrive\Desktop\Claude\2024NorthAmericanOpen2024 entries by day.csv",
        "date_col": "RegisteredTime",
        "date_formats": ["%m/%d/%Y %H:%M"],
        "tournament_date": "2024-12-26",
        "expected_total": 1102,
        "has_early_bird": False,
        "fee_deadlines": [],
    },
    "wo2026": {
        "id": "WO2026",
        "name": "World Open 2026",
        "csv": r"c:\Users\eliza\Downloads\World Open data for modeling.csv",
        "date_col": "RegisteredTime",
        "date_formats": ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"],
        "tournament_date": "2026-07-02",
        "expected_total": 277,
        "has_early_bird": True,
        "fee_deadlines": [
            {"date": "2026-06-01", "label": "Standard Deadline"},
            {"date": "2026-06-22", "label": "Late Fee"},
        ],
    },
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_date(s: str, formats: list[str]) -> datetime | None:
    """Try multiple date formats. Return None for invalid/null dates."""
    s = s.strip()
    if not s or s.startswith("0000"):
        return None
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_csv_timestamps(csv_path: str, date_col: str, formats: list[str]) -> list[datetime]:
    """
    Read a CSV and extract valid registration timestamps.
    Skips rows with invalid/null dates.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"  WARNING: CSV not found at {csv_path}")
        return []

    timestamps = []
    bad_dates = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(date_col, "").strip()
            dt = parse_date(raw, formats)
            if dt is None:
                bad_dates += 1
                continue
            timestamps.append(dt)

    print(f"  Loaded {len(timestamps)} valid timestamps ({bad_dates} skipped)")
    return timestamps


def build_cumulative(timestamps: list[datetime], first_reg_date: date) -> list[list]:
    """
    Convert raw timestamps into [[t, cumulative_count], ...] daily aggregation.
    t = calendar days since first_reg_date.
    Only includes days with at least one registration.
    """
    # Count registrations per day
    daily_counts: dict[int, int] = defaultdict(int)
    for dt in timestamps:
        t = (dt.date() - first_reg_date).days
        if t >= 0:
            daily_counts[t] += 1

    if not daily_counts:
        return []

    # Build cumulative series
    max_t = max(daily_counts.keys())
    cumulative = []
    running = 0
    for t in range(max_t + 1):
        if t in daily_counts:
            running += daily_counts[t]
            cumulative.append([t, running])

    return cumulative


# ── Main processing ───────────────────────────────────────────────────────────

def process_dataset(key: str, out_dir: Path) -> dict | None:
    """Process one dataset, save cumulative JSON, return metadata."""
    cfg = DATASETS[key]
    print(f"\n── {cfg['name']} ──────────────────────────")

    timestamps = load_csv_timestamps(cfg["csv"], cfg["date_col"], cfg["date_formats"])
    if not timestamps:
        print(f"  No data — skipping")
        return None

    # Determine first_reg_date from data
    timestamps.sort()
    first_reg_date = timestamps[0].date()
    tournament_date = date.fromisoformat(cfg["tournament_date"])
    t_final = (tournament_date - first_reg_date).days

    print(f"  First registration: {first_reg_date}  (t=0)")
    print(f"  Tournament date:    {tournament_date}  (t={t_final})")
    print(f"  Total entries:      {len(timestamps)}")
    print(f"  Has early bird:     {cfg['has_early_bird']}")

    cumulative = build_cumulative(timestamps, first_reg_date)
    n_points = len(cumulative)
    print(f"  Cumulative points:  {n_points}")

    out_path = out_dir / f"cumulative_{cfg['id']}.json"
    with open(out_path, "w") as f:
        json.dump(cumulative, f)
    print(f"  Saved → {out_path}")

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "tournament_date": cfg["tournament_date"],
        "first_reg_date": first_reg_date.isoformat(),
        "t_final": t_final,
        "total_entries": len(timestamps),
        "n_cumulative_points": n_points,
        "has_early_bird": cfg["has_early_bird"],
        "expected_total": cfg["expected_total"],
        "fee_deadlines": cfg.get("fee_deadlines", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize tournament CSVs to cumulative format")
    parser.add_argument(
        "--tournament",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    parser.add_argument("--out-dir", default=".tmp", help="Output directory (default: .tmp)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(DATASETS.keys()) if args.tournament == "all" else [args.tournament]

    meta = []
    for key in keys:
        result = process_dataset(key, out_dir)
        if result:
            meta.append(result)

    # Save metadata summary
    meta_path = out_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "datasets": meta,
        }, f, indent=2)
    print(f"\nMetadata → {meta_path}")
    print(f"\nDone. Processed {len(meta)}/{len(keys)} datasets.")

    # Print cumulative summary per dataset
    for m in meta:
        if m["n_cumulative_points"] > 0:
            cum_path = out_dir / f"cumulative_{m['id']}.json"
            with open(cum_path) as f:
                data = json.load(f)
            last_t, last_count = data[-1]
            print(f"  {m['id']:12s} {m['n_cumulative_points']:3d} days, "
                  f"final={last_count}, t_final={m['t_final']}, "
                  f"early_bird={m['has_early_bird']}")


if __name__ == "__main__":
    main()
