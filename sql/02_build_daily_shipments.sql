-- ============================================================================
-- Build fact_daily_shipments_by_state
-- ============================================================================
-- Produces a dense table with one row per (shipment_date, customer_state),
-- including zero-shipment days for forecasting integrity.
--
-- Definition of "shipment":
--   * order_status IN ('delivered', 'shipped')
--   * order_delivered_carrier_date IS NOT NULL
--
-- Granularity: 1 row per (day, state). State is the customer's state, i.e.
-- the destination of the shipment.
--
-- Steps (CTEs):
--   1. valid_shipments     : filter orders that are real shipments and
--                            extract the shipment date (date part only)
--   2. raw_daily_counts    : aggregate count per (date, state)
--   3. date_spine          : generate every date between min and max
--   4. state_spine         : every state observed in the data
--   5. full_grid           : cartesian product (date_spine x state_spine)
--   6. final               : LEFT JOIN full_grid with raw_daily_counts,
--                            replacing NULL with 0
-- ============================================================================

DELETE FROM fact_daily_shipments_by_state;

WITH RECURSIVE
-- 1. Filter shipments and project relevant columns
valid_shipments AS (
    SELECT
        DATE(o.order_delivered_carrier_date) AS shipment_date,
        c.customer_state                     AS customer_state
    FROM fact_orders o
    JOIN dim_customers c
      ON c.customer_id = o.customer_id
    WHERE o.order_status IN ('delivered', 'shipped')
      AND o.order_delivered_carrier_date IS NOT NULL
),

-- 2. Aggregate to (date, state) granularity
raw_daily_counts AS (
    SELECT
        shipment_date,
        customer_state,
        COUNT(*) AS n_shipments
    FROM valid_shipments
    GROUP BY shipment_date, customer_state
),

-- 3. Build a dense date spine using a recursive CTE
date_bounds AS (
    SELECT
        MIN(shipment_date) AS min_date,
        MAX(shipment_date) AS max_date
    FROM raw_daily_counts
),
date_spine AS (
    SELECT min_date AS d FROM date_bounds
    UNION ALL
    SELECT DATE(d, '+1 day')
    FROM date_spine
    WHERE d < (SELECT max_date FROM date_bounds)
),

-- 4. Distinct states (only those that ever received a shipment)
state_spine AS (
    SELECT DISTINCT customer_state FROM raw_daily_counts
),

-- 5. Cartesian product = the full dense grid
full_grid AS (
    SELECT
        ds.d              AS shipment_date,
        ss.customer_state AS customer_state
    FROM date_spine  ds
    CROSS JOIN state_spine ss
)

-- 6. Final result: dense grid with zero-filled counts
INSERT INTO fact_daily_shipments_by_state (shipment_date, customer_state, n_shipments)
SELECT
    fg.shipment_date,
    fg.customer_state,
    COALESCE(rdc.n_shipments, 0) AS n_shipments
FROM full_grid fg
LEFT JOIN raw_daily_counts rdc
       ON rdc.shipment_date  = fg.shipment_date
      AND rdc.customer_state = fg.customer_state;

-- ============================================================================
-- Summary stats (printed to stdout when run with sqlite3 CLI)
-- ============================================================================
SELECT
    COUNT(*)                       AS total_rows,
    COUNT(DISTINCT shipment_date)  AS n_days,
    COUNT(DISTINCT customer_state) AS n_states,
    MIN(shipment_date)             AS first_date,
    MAX(shipment_date)             AS last_date,
    SUM(n_shipments)               AS total_shipments,
    SUM(CASE WHEN n_shipments = 0 THEN 1 ELSE 0 END) AS zero_days
FROM fact_daily_shipments_by_state;
