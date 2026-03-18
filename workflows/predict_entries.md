# Workflow: Chess Tournament Entry Prediction

## Objective
Fetch current registration data from the CCA (Continental Chess Association) registration portal, fit a double-sigmoid model to the cumulative entry curve, compute a delta vs. expected, and produce a prediction with confidence interval for the tournament's final registration count.

---

## CCA Background

**Continental Chess Association** is the largest independent chess tournament organizer in the US, founded by Bill Goichberg in 1968. They run ~26 tournaments/year via chessaction.com / onlineregistration.cc.

### Tournament Families (for same-family clustering)

| Family | Event | Location | Typical Dates | Typical Total Entries | Has Early Bird |
|--------|-------|----------|---------------|-----------------------|----------------|
| `world_open` | World Open | Philadelphia PA → DC (2026) | July 4th week | 1,000–1,400 | Yes |
| `chicago_open` | Chicago Open | Westin Wheeling IL | Memorial Day weekend | 900–950 | Yes |
| `north_american_open` | North American Open | Las Vegas NV | Dec 26–30 | 900–1,100 | No |
| `national_chess_congress` | National Chess Congress | Philadelphia PA | Thanksgiving | 300–600 | No |
| `continental_open` | Continental Open | Woburn MA | August | 200–400 | No |
| `eastern_open` | Eastern Open | Washington DC | Dec 26–29 | 300–500 | No |
| `liberty_bell` | Liberty Bell Open | Philadelphia PA | January | 300–600 | No |
| `atlantic_city` | Atlantic City Open | Atlantic City NJ | Easter | 200–500 | No |

### Fee Structure and Double-Sigmoid Causation

CCA uses a **3-tier fee structure** at major events that directly drives the double-sigmoid shape:

| Tier | Trigger | Effect on Registrations |
|------|---------|------------------------|
| Early Bird | Deadline ~6–10 weeks before tournament | First sigmoid hump: price spike triggers rush to register |
| Standard | Deadline ~1 week before | Mid-cycle trickle |
| Late/At-Door | Round 1 minus 1 hour | Second sigmoid hump: final surge before doors close |

**Chicago Open 2026 fee tiers:** $207 (by Mar 19) → $227 (by May 19) → $250 (at door)
**World Open 2025 fee tiers:** $318 (by May 15) → $328 (by Jun 28) → $350 (at door)
**NAO 2025 fee tiers:** $245 (by Sep 20) → $275 (by Dec 23) → at door — despite having an early bird deadline, NAO data shows single-sigmoid behavior (see `has_early_bird: false` in config)

### Tournament Sections
Each major CCA event has 8 sections divided by USCF rating (Open/unlimited, U2300, U2100, U1900, U1700, U1500, U1300, U1000). Entry counts shown above are totals across all sections. The prediction model tracks total registrations, not per-section.

### Registration System Notes
- Portal: chessaction.com / onlineregistration.cc (same backend)
- Entry timestamps NOT available on public pages — only in CCA backend CSV exports
- Walk-in entries logged as `0000-00-00` — include in final count, exclude from time-series
- Advance entry list uses two-step pattern: PHP page → parse JS → fetch static HTML file

---

## Required Inputs

| Input | Source | Notes |
|-------|--------|-------|
| `tournament_id` | User / config | Identifier used in the registration URL |
| `tournament_url` | User / config | e.g. `https://onlineregistration.cc/tournaments/advance_entry_list.php?tid=XXXX` |
| `tournament_name` | User / config | Human-readable label |
| `tournament_date` | User / config | ISO date string, e.g. `2026-07-02` |
| `first_reg_date` | User / config | Date of first observed registration (ISO) |
| `prior_params` | Optional | Pre-fitted L1,k1,m1,L2,k2,m2 from the brief or prior run |
| `fee_deadlines` | Optional | List of `{date, label}` objects for deadline markers |

---

## Tools Used (in order)

1. `tools/scrape_entries.py` — Fetches the live registration list and returns `[{name, registered_time}]`
2. `tools/fit_double_sigmoid.py` — Fits the double-sigmoid model to cumulative daily counts
3. `tools/compute_delta.py` — Compares actual count to model prediction at current t, returns delta %
4. *(future)* `tools/send_alert.py` — Sends email/SMS if delta exceeds threshold

---

## Step-by-Step Procedure

### Step 1 — Scrape Current Entries

Run the scraper, passing the tournament URL:

```bash
python tools/scrape_entries.py --url "<tournament_url>" --out .tmp/entries_<tournament_id>.json
```

**Expected output:** `.tmp/entries_<tournament_id>.json` — array of `{name, registered_time}` objects.

**Edge cases:**
- If the page returns 0 entries, the tournament may not be open for registration yet. Log and exit.
- If `registered_time` fields are missing or malformed, fall back to counting entries without timestamps (total count only, no curve fitting possible).
- Rate limit: wait 2s between requests; retry once on timeout. Do not hammer the server.

### Step 2 — Aggregate to Cumulative Daily Counts

Parse timestamps from Step 1. Group by calendar day, compute cumulative sum.

Convert each day to a `t` value:
```
t = (calendar_date - first_reg_date).days
```

Produce a list of `[t, cumulative_count]` pairs. Save to `.tmp/cumulative_<tournament_id>.json`.

**Edge cases:**
- Entries with `0000-00-00` timestamps are walk-ins — include in final count but exclude from the time-series.
- If fewer than 10 data points exist, fitting is unreliable. Use prior params directly (Step 3b).

### Step 3a — Fit Double-Sigmoid Model (if ≥ 10 data points)

Run the fitter:

```bash
python tools/fit_double_sigmoid.py \
  --data .tmp/cumulative_<tournament_id>.json \
  --out .tmp/params_<tournament_id>.json
```

**Expected output:** `{L1, k1, m1, L2, k2, m2, r2, rmse}` — fitted parameters.

The tool uses `scipy.optimize.curve_fit` with bounds:
- `L1, L2 > 0` (positive asymptotes)
- `0 < k1, k2 < 1` (growth rates)
- `m1 < m2` (first sigmoid midpoint before second)

**Initialization strategy:**
- `m1` ≈ t at 25% of current max count
- `m2` ≈ t at 75% of current max count
- `L1` ≈ 10% of expected final total
- `L2` ≈ 90% of expected final total

If `curve_fit` raises `RuntimeError` (convergence failure), fall back to prior params.

### Step 3b — Use Prior Params (if < 10 points or fit failed)

Load from `prior_params` input (e.g. the World Open 2026 brief parameters):
- `L1=20.6, k1=0.0468, m1=79.2, L2=256.7, k2=0.2512, m2=246.7`

Log that prior params were used.

### Step 4 — Compute Delta

Using today's date, compute `t_today`:
```
t_today = (today - first_reg_date).days
```

Compute predicted count at `t_today` from the fitted sigmoid:
```
predicted = sigmoid(t_today, L1, k1, m1, L2, k2, m2)
```

Compute delta:
```
delta_pct = (actual_count - predicted) / predicted * 100
```

Status thresholds:
- `delta_pct > +10%` → GREEN (tracking above expectation)
- `-10% ≤ delta_pct ≤ +10%` → YELLOW (on track)
- `delta_pct < -10%` → RED (tracking below expectation)

### Step 5 — Generate Final Prediction

Compute predicted final total at `t_final` (day of tournament):
```
t_final = (tournament_date - first_reg_date).days
predicted_final = sigmoid(t_final, ...)
```

Confidence interval: ±15% of predicted final (simple heuristic; replace with RMSE-based CI when data is sufficient).

### Step 6 — Output

Print a summary JSON to stdout and save to `.tmp/prediction_<tournament_id>.json`:

```json
{
  "tournament": "World Open 2026",
  "as_of": "2026-03-10",
  "actual_count": 20,
  "predicted_now": 19.4,
  "delta_pct": 3.1,
  "status": "GREEN",
  "predicted_final": 277.3,
  "ci_low": 235.7,
  "ci_high": 318.9,
  "params": {"L1": 20.6, "k1": 0.0468, "m1": 79.2, "L2": 256.7, "k2": 0.2512, "m2": 246.7},
  "r2": 0.964,
  "rmse": 1.09
}
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Network timeout | Retry once after 5s; if still fails, use last cached `.tmp/entries_*.json` |
| Scraper blocked (403/429) | Log and exit; do not retry automatically. Notify user. |
| Fit convergence failure | Fall back to prior params; log warning |
| Fewer than 5 data points | Skip fit entirely; report count only |
| Invalid tournament URL | Exit immediately with clear error message |

---

## Recurring Execution

This workflow is designed to run every 6 hours via scheduler (cron / Celery / Task Scheduler).

Cron example (every 6 hours):
```
0 */6 * * * python /path/to/tools/scrape_entries.py --config /path/to/config.json
```

Each run appends to `.tmp/history_<tournament_id>.jsonl` for trend tracking.

---

## Known Constraints

- `onlineregistration.cc` does not have a public API — HTML scraping required
- Page structure may change; check `<table>` selector if scraper breaks
- Some tournaments use a different registration vendor (ChessBase Events, US Chess) — need separate scraper tools
- Walk-in entries (`0000-00-00`) inflate final count but cannot be modeled — account for in CI

---

## Model Ensemble (Phase 2)

The system now runs five models in parallel for any tournament with cumulative data:

| Model | Best For | Implementation |
|-------|----------|----------------|
| Single Sigmoid | No early bird (most CCA events) | scipy curve_fit, TRF method |
| Double Sigmoid | Early bird discount (two phases) | scipy curve_fit, multiple inits |
| Monte Carlo | Uncertainty quantification | Parametric bootstrap from covariance |
| Bayesian MCMC | Sparse data (<15 points) | Metropolis-Hastings, empirical priors |
| Thompson Sampling | Sequential updates | Normal-Normal conjugate posterior |

**Ensemble weighting:**
- ≥20 data points: emphasize Monte Carlo + Sigmoid models
- 10–19 points: balanced weighting
- <10 points: emphasize Bayesian MCMC + Thompson Sampling

Run ensemble:
```bash
python tools/prepare_training_data.py                    # normalize CSVs → .tmp/
python tools/model_ensemble.py --id WO2026              # run all 5 models
python tools/model_ensemble.py --backtest               # evaluate on historical data
python tools/model_ensemble.py --compare                # run on all available data
```

API endpoint: `GET /tournaments/{tid}/ensemble?refresh=true`

**Two tournament types:**
- **With early bird** (Chicago 2022, WO 2026): Double sigmoid preferred
- **Without early bird** (NAO 2024, most CCA events): Single sigmoid preferred

The `has_early_bird` flag in config.json drives model selection in the ensemble.

## Improvement Log

- **2026-03-10:** Initial workflow created. World Open 2026 using pre-fitted brief params (20 entries). Chicago Open 2022 fit via gradient descent in JS prototype.
- **2026-03-10:** Multi-model ensemble added. Five model families (Single Sigmoid, Double Sigmoid, Monte Carlo, Bayesian MCMC, Thompson Sampling). Training datasets: Chicago 2022 (943 entries), NAO 2024 (1,102 entries). Dashboard updated with model comparison panel.
- **2026-03-12:** Added CCA background section: tournament families, fee structure, section counts, expected final entry ranges sourced from chesstour.com and chessevents.com historical results.
