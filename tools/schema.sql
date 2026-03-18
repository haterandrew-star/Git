-- Chess Tournament Entry Prediction — PostgreSQL Schema
-- Run once to initialize the database.
-- Compatible with PostgreSQL 14+

-- ── Extensions ───────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ── Tournaments ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tournaments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    short_name      TEXT,                        -- e.g. "WO2026"
    tournament_date DATE NOT NULL,               -- first day of tournament
    city            TEXT,
    registration_url TEXT,                       -- onlineregistration.cc URL
    first_reg_date  DATE,                        -- date of first observed registration
    expected_total  INTEGER,                     -- prior estimate of final entries
    -- Model parameters (pre-fitted or from most recent fit)
    param_l1        DOUBLE PRECISION,
    param_k1        DOUBLE PRECISION,
    param_m1        DOUBLE PRECISION,
    param_l2        DOUBLE PRECISION,
    param_k2        DOUBLE PRECISION,
    param_m2        DOUBLE PRECISION,
    param_r2        DOUBLE PRECISION,
    param_rmse      DOUBLE PRECISION,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tournaments_date ON tournaments(tournament_date);
CREATE INDEX IF NOT EXISTS idx_tournaments_active ON tournaments(is_active) WHERE is_active = TRUE;


-- ── Entries ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS entries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id   UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    player_name     TEXT NOT NULL,
    registered_at   TIMESTAMPTZ,                 -- NULL for walk-ins (0000-00-00)
    is_walkin       BOOLEAN GENERATED ALWAYS AS (registered_at IS NULL) STORED,
    section         TEXT,
    rating          INTEGER,
    scraped_at      TIMESTAMPTZ DEFAULT NOW(),    -- when we observed this entry
    UNIQUE (tournament_id, player_name)           -- deduplicate on re-scrape
);

CREATE INDEX IF NOT EXISTS idx_entries_tournament ON entries(tournament_id);
CREATE INDEX IF NOT EXISTS idx_entries_registered_at ON entries(registered_at);


-- ── Scrape Log ────────────────────────────────────────────────────────────────
-- One row per scrape run, tracks how entry count changed over time.

CREATE TABLE IF NOT EXISTS scrape_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id   UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    scraped_at      TIMESTAMPTZ DEFAULT NOW(),
    entry_count     INTEGER NOT NULL,            -- total entries at time of scrape
    new_entries     INTEGER DEFAULT 0,           -- entries added since last scrape
    status          TEXT DEFAULT 'ok',           -- 'ok' | 'error' | 'blocked'
    error_msg       TEXT
);

CREATE INDEX IF NOT EXISTS idx_scrape_log_tournament ON scrape_log(tournament_id, scraped_at DESC);


-- ── Predictions ───────────────────────────────────────────────────────────────
-- One row per prediction run (every 6 hours or on demand).

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id   UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    predicted_at    TIMESTAMPTZ DEFAULT NOW(),
    t_today         DOUBLE PRECISION,            -- t-value for today
    actual_count    INTEGER,                     -- entries at time of prediction
    predicted_now   DOUBLE PRECISION,            -- model's prediction for today
    delta_pct       DOUBLE PRECISION,            -- % above/below model
    status          TEXT,                        -- 'GREEN' | 'YELLOW' | 'RED'
    t_final         DOUBLE PRECISION,            -- t-value for tournament date
    predicted_final DOUBLE PRECISION,            -- model's final prediction
    ci_low          DOUBLE PRECISION,            -- confidence interval low
    ci_high         DOUBLE PRECISION,            -- confidence interval high
    -- Params used for this prediction
    param_l1        DOUBLE PRECISION,
    param_k1        DOUBLE PRECISION,
    param_m1        DOUBLE PRECISION,
    param_l2        DOUBLE PRECISION,
    param_k2        DOUBLE PRECISION,
    param_m2        DOUBLE PRECISION,
    param_r2        DOUBLE PRECISION,
    param_rmse      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_predictions_tournament ON predictions(tournament_id, predicted_at DESC);


-- ── Alerts ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id   UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    triggered_at    TIMESTAMPTZ DEFAULT NOW(),
    alert_type      TEXT NOT NULL,               -- 'below_threshold' | 'surge_detected' | 'model_updated'
    message         TEXT NOT NULL,
    delta_pct       DOUBLE PRECISION,
    channel         TEXT DEFAULT 'email',        -- 'email' | 'sms' | 'slack'
    sent_to         TEXT,
    sent_at         TIMESTAMPTZ,
    acknowledged    BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alerts_tournament ON alerts(tournament_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged) WHERE acknowledged = FALSE;


-- ── Fee Deadlines ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fee_deadlines (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tournament_id   UUID NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    deadline_date   DATE NOT NULL,
    label           TEXT NOT NULL,              -- e.g. "Early Bird -$50"
    price_delta     INTEGER,                    -- price change in dollars (negative = discount)
    display_order   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_deadlines_tournament ON fee_deadlines(tournament_id, deadline_date);


-- ── Useful views ──────────────────────────────────────────────────────────────

-- Latest prediction per tournament
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (tournament_id)
    p.*,
    t.name AS tournament_name,
    t.tournament_date,
    t.short_name
FROM predictions p
JOIN tournaments t ON t.id = p.tournament_id
ORDER BY tournament_id, predicted_at DESC;

-- Daily cumulative entry counts per tournament (for charting)
CREATE OR REPLACE VIEW v_daily_entry_counts AS
SELECT
    tournament_id,
    DATE(registered_at) AS reg_date,
    COUNT(*) AS daily_count,
    SUM(COUNT(*)) OVER (
        PARTITION BY tournament_id
        ORDER BY DATE(registered_at)
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_count
FROM entries
WHERE registered_at IS NOT NULL
GROUP BY tournament_id, DATE(registered_at);


-- ── Seed data: known tournaments ──────────────────────────────────────────────

INSERT INTO tournaments (
    name, short_name, tournament_date, city, first_reg_date,
    param_l1, param_k1, param_m1, param_l2, param_k2, param_m2,
    param_r2, param_rmse
) VALUES (
    'World Open 2026', 'WO2026', '2026-07-02', 'Philadelphia, PA',
    '2025-09-01',  -- approximate; adjust when known
    20.6, 0.0468, 79.2,
    256.7, 0.2512, 246.7,
    0.964, 1.09
) ON CONFLICT DO NOTHING;

INSERT INTO tournaments (
    name, short_name, tournament_date, city, first_reg_date
) VALUES (
    'Chicago Open 2022', 'CHI2022', '2022-05-27', 'Chicago, IL',
    '2022-01-07'
) ON CONFLICT DO NOTHING;

-- Fee deadlines for World Open 2026 (from brief — adjust when official schedule released)
-- These are approximate; update when CCA publishes the official schedule
INSERT INTO fee_deadlines (tournament_id, deadline_date, label, display_order)
SELECT t.id, '2026-06-01', 'Standard Deadline', 1
FROM tournaments t WHERE t.short_name = 'WO2026';

INSERT INTO fee_deadlines (tournament_id, deadline_date, label, display_order)
SELECT t.id, '2026-06-22', 'Late Fee Begins', 2
FROM tournaments t WHERE t.short_name = 'WO2026';
