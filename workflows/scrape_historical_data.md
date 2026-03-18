# Workflow: Scrape Historical Tournament Data for Model Training

## Objective
Pull registration entry lists from all past CCA (Continental Chess Association) tournaments on chessaction.com / onlineregistration.cc, aggregate them into cumulative time-series datasets, and fit a double-sigmoid model to each one. The resulting parameter distributions become empirical priors for the Bayesian hierarchical model.

**CCA = vendor_search=3** on the chessaction.com AJAX endpoints.
Total past CCA tournaments in the system: ~892 (as of 2026-03). The platform hosts ~2,909 total past events across all vendors; CCA is the largest single vendor.

---

## Why This Matters

The model's prediction quality scales directly with historical data. With ~892 past CCA tournaments available:
- Even if 10% have enough timestamped data, that's ~89 fitted models
- Same-family tournaments (e.g., all past Chicago Opens) give the tightest priors — use `tournament_family` field in config.json to group correctly
- The parameter distributions (mean/SD of k1, k2, m1, m2, L2-fraction) become tight priors that dramatically narrow confidence intervals for new tournaments
- This is what enables meaningful predictions from just 5-10 early registrations

**Flagship event prior data quality (chessevents.com historical results):**

| Family | Years Available | Typical Final Entries | Note |
|--------|----------------|-----------------------|------|
| World Open | 1973–2025 | 1,000–1,400 | 8 sections; largest non-scholastic event in US |
| Chicago Open | ~2000–2025 | 900–950 | 8 sections; record 925 (2017) |
| North American Open | ~2010–2025 | 900–1,100 | 8 sections; single sigmoid |
| National Chess Congress | ~2005–2025 | 300–600 | 10 sections including U800, U600 |

---

## Tools Used

1. `tools/scrape_past_events.py` — Indexes all past tournaments + scrapes entry lists
2. `tools/fit_training_data.py` — Fits double-sigmoid to each tournament, computes priors
3. `tools/fit_double_sigmoid.py` — Used indirectly; the training fitter uses the same logic

---

## Step-by-Step Procedure

### Phase 1 — Build Tournament Index (fast, ~2 min)

```bash
python tools/scrape_past_events.py \
  --vendor 3 \
  --index-only \
  --out .tmp/past_events_index.json
```

- `--vendor 3` = Continental Chess Association only
- Use `--vendor 0` to get all vendors (slower, more data)
- Output: `.tmp/past_events_index.json` — list of all tournaments with metadata and entry URLs
- The index will show how many tournaments have usable entry URLs

**Expected output:**
```
Scraping tournament index (vendor=3)...
  Fetched 100/892 (11%)
  ...
  Fetched 892/892 (100%)
Index complete: 892 tournaments

Summary:
  Total tournaments: 892
  With entry URLs:   741
  With TIDs:         741
```

**If vendor=3 returns no CCA results**, try `--vendor 0` (all vendors) and then filter the index JSON manually.

### Phase 2 — Scrape Entry Lists (slow, hours)

```bash
python tools/scrape_past_events.py \
  --index .tmp/past_events_index.json \
  --out-dir .tmp/past_events/ \
  --training-out .tmp/past_events_training.json \
  --delay 2.0 \
  --resume
```

- `--delay 2.0` = 2 seconds between requests — essential to avoid rate-limiting
- `--resume` = skips tournaments already scraped (safe to re-run after interruption)
- Each tournament saved to `.tmp/past_events/<tid>.json`
- Training data saved to `.tmp/past_events_training.json`

**Time estimate:** ~892 tournaments × 2s delay = ~30 minutes. Run as a background job.

**If blocked (403/429):** Increase `--delay` to 5.0 and re-run with `--resume`.

### Phase 3 — Fit Models to All Historical Tournaments (~5 min)

```bash
python tools/fit_training_data.py \
  --training .tmp/past_events_training.json \
  --out .tmp/all_fitted_params.json \
  --report .tmp/fit_report.json \
  --min-points 15 \
  --min-entries 30
```

Output:
- `.tmp/all_fitted_params.json` — all fitted parameter sets, sorted by R²
- `.tmp/fit_report.json` — prior distributions + top fits ranking

**Expected output excerpt:**
```
EMPIRICAL PRIORS (use for Bayesian model initialization)
  L1: mean=28.4  sd=14.2  [12.1 – 47.8]
  k1: mean=0.052  sd=0.021  [0.028 – 0.081]
  m1: mean=82.3  sd=24.1  [51.2 – 118.4]
  L2: mean=284.2  sd=156.3  [104.3 – 498.1]
  k2: mean=0.238  sd=0.089  [0.142 – 0.371]
  m2: mean=231.6  sd=48.2  [168.3 – 295.1]
```

### Phase 4 — Update config.json with Learned Priors

Open `.tmp/fit_report.json` and copy the `prior_distributions` means/SDs into
`config.json` under a top-level `"empirical_priors"` key. This improves predictions
for any new tournament from day one.

---

## Output File Reference

| File | Description |
|------|-------------|
| `.tmp/past_events_index.json` | All tournaments found — name, tid, entry_url, dates |
| `.tmp/past_events/<tid>.json` | Per-tournament: raw entries with timestamps |
| `.tmp/past_events_training.json` | Aggregated: [[t, cumulative], ...] per tournament |
| `.tmp/all_fitted_params.json` | All fitted {L1,k1,m1,L2,k2,m2,r2,rmse} per tournament |
| `.tmp/fit_report.json` | Prior distributions + top-20 ranked fits |

---

## Filtering and Quality Control

After fitting, the report ranks tournaments by R². Good fits have:
- R² > 0.90 — excellent
- R² 0.75–0.90 — acceptable
- R² < 0.75 — poor (excluded from prior computation)

Poor fits typically occur when:
- Tournament had a single registration spike (one-phase not two)
- Very small tournament (<30 entries) — insufficient shape definition
- Timestamps are missing for most entries (registration done offline)

---

## Known Constraints

- **POST-only API:** The chessaction.com DataTables endpoint requires HTTP POST.
  `scrape_past_events.py` handles this correctly; WebFetch cannot.
- **Rate limiting:** Be respectful. Use `--delay 2.0` minimum. The site has no
  explicit rate-limit documentation but 403s will occur at high speed.
- **Timestamp coverage:** Many older tournaments may not have registration timestamps
  (entered manually after the fact). These are filtered out automatically.
- **Vendor ID:** CCA is vendor_search=3. This may change. If index returns 0 results,
  try vendor_search=0 (all vendors) and search by tournament name.

---

## Improvement Log

- **2026-03-10:** Initial workflow created. 2,909 total past tournaments found (all vendors); ~892 CCA-only.
  AJAX endpoint confirmed as POST-only with vendor_search=3 for CCA.
  scrape_past_events.py + fit_training_data.py tools built.
- **2026-03-12:** Clarified CCA vendor count vs. platform total. Added prior data quality table for flagship events. Added tournament_family grouping guidance.
