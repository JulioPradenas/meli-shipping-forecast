-- ============================================================================
-- Build fact_daily_shipments_features
-- ============================================================================
-- Adds lag and rolling features per (state, date) for use in forecasting.
--
-- Features computed (all per customer_state, ordered by shipment_date):
--   * lag_1, lag_7, lag_14, lag_28   : value 1/7/14/28 days ago
--   * rolling_mean_7,  rolling_std_7  : 7-day rolling stats (excl. current day)
--   * rolling_mean_14, rolling_std_14 : 14-day rolling stats
--   * rolling_mean_28, rolling_std_28 : 28-day rolling stats
--
-- Design notes:
--   * All rolling windows exclude the current day (`ROWS BETWEEN N PRECEDING
--     AND 1 PRECEDING`) to avoid label leakage. The current row's value is
--     the target, not a feature.
--   * NULL values are expected for rows near the start of each state's history;
--     they should be handled in feature engineering (imputation or filtering).
--   * Stored as a separate table to keep the raw aggregate clean and to allow
--     fast feature engineering iteration without touching the source.
-- ============================================================================

DROP TABLE IF EXISTS fact_daily_shipments_features;

CREATE TABLE fact_daily_shipments_features AS
WITH base AS (
    SELECT
        shipment_date,
        customer_state,
        n_shipments
    FROM fact_daily_shipments_by_state
)
SELECT
    shipment_date,
    customer_state,
    n_shipments,

    -- Lags (value at a specific lag in time)
    LAG(n_shipments,  1) OVER w AS lag_1,
    LAG(n_shipments,  7) OVER w AS lag_7,
    LAG(n_shipments, 14) OVER w AS lag_14,
    LAG(n_shipments, 28) OVER w AS lag_28,

    -- Rolling means (exclude current day to avoid leakage)
    AVG(n_shipments) OVER (
        PARTITION BY customer_state
        ORDER BY shipment_date
        ROWS BETWEEN  7 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_7,

    AVG(n_shipments) OVER (
        PARTITION BY customer_state
        ORDER BY shipment_date
        ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_14,

    AVG(n_shipments) OVER (
        PARTITION BY customer_state
        ORDER BY shipment_date
        ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_28,

    -- Rolling std (volatility proxy)
    -- Note: SQLite computes population std, not sample. Acceptable for our use.
    -- We use a CASE to return NULL when window is incomplete (n < 2).
    CASE
        WHEN COUNT(*) OVER (
            PARTITION BY customer_state
            ORDER BY shipment_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) >= 2
        THEN (
            SELECT SQRT(AVG((x - mu) * (x - mu)))
            FROM (
                SELECT n_shipments AS x,
                       AVG(n_shipments) OVER (
                           PARTITION BY customer_state
                           ORDER BY shipment_date
                           ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                       ) AS mu
                FROM fact_daily_shipments_by_state s2
                WHERE s2.customer_state = base.customer_state
                  AND s2.shipment_date BETWEEN
                      DATE(base.shipment_date, '-7 days')
                      AND DATE(base.shipment_date, '-1 day')
            )
        )
        ELSE NULL
    END AS rolling_std_7

FROM base
WINDOW w AS (
    PARTITION BY customer_state
    ORDER BY shipment_date
);

CREATE INDEX idx_features_state_date
    ON fact_daily_shipments_features(customer_state, shipment_date);

-- Sanity check
SELECT
    COUNT(*)                          AS total_rows,
    COUNT(lag_1)                      AS rows_with_lag_1,
    COUNT(lag_28)                     AS rows_with_lag_28,
    COUNT(rolling_mean_7)             AS rows_with_rmean_7,
    COUNT(rolling_std_7)              AS rows_with_rstd_7,
    MIN(shipment_date)                AS min_date,
    MAX(shipment_date)                AS max_date
FROM fact_daily_shipments_features;
