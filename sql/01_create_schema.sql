-- ============================================================================
-- Schema for Brazilian E-Commerce dataset (Olist) — analytical layer
-- ============================================================================
-- Design notes:
-- * Dimensional model: fact tables for events + dim tables for entities.
-- * Timestamps are stored as TEXT in ISO 8601 format (SQLite convention).
--   SQLite's date functions accept this format natively.
-- * All entity IDs are TEXT (32-char hex strings in the source data).
-- * Foreign keys are declared for documentation and integrity, even though
--   SQLite requires PRAGMA foreign_keys = ON to enforce them at runtime.
-- ============================================================================

-- Disable FKs during DDL so we can DROP tables in any order without errors.
-- Re-enabled at the bottom of the script.
PRAGMA foreign_keys = OFF;

-- ---------------------------------------------------------------------------
-- Dimension tables
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS dim_customers;
CREATE TABLE dim_customers (
    customer_id              TEXT PRIMARY KEY,
    customer_unique_id       TEXT NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city            TEXT NOT NULL,
    customer_state           TEXT NOT NULL CHECK (length(customer_state) = 2)
);

CREATE INDEX idx_dim_customers_state  ON dim_customers(customer_state);
CREATE INDEX idx_dim_customers_unique ON dim_customers(customer_unique_id);

DROP TABLE IF EXISTS dim_sellers;
CREATE TABLE dim_sellers (
    seller_id              TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city            TEXT NOT NULL,
    seller_state           TEXT NOT NULL CHECK (length(seller_state) = 2)
);

CREATE INDEX idx_dim_sellers_state ON dim_sellers(seller_state);

DROP TABLE IF EXISTS dim_products;
CREATE TABLE dim_products (
    product_id                  TEXT PRIMARY KEY,
    product_category_name       TEXT,
    product_category_name_en    TEXT,
    product_name_length         REAL,
    product_description_length  REAL,
    product_photos_qty          REAL,
    product_weight_g            REAL,
    product_length_cm           REAL,
    product_height_cm           REAL,
    product_width_cm            REAL
);

CREATE INDEX idx_dim_products_category ON dim_products(product_category_name);

-- ---------------------------------------------------------------------------
-- Fact tables
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS fact_orders;
CREATE TABLE fact_orders (
    order_id                       TEXT PRIMARY KEY,
    customer_id                    TEXT NOT NULL,
    order_status                   TEXT NOT NULL,
    order_purchase_timestamp       TEXT NOT NULL,
    order_approved_at              TEXT,
    order_delivered_carrier_date   TEXT,
    order_delivered_customer_date  TEXT,
    order_estimated_delivery_date  TEXT,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

CREATE INDEX idx_fact_orders_customer       ON fact_orders(customer_id);
CREATE INDEX idx_fact_orders_status         ON fact_orders(order_status);
CREATE INDEX idx_fact_orders_purchase_date  ON fact_orders(order_purchase_timestamp);
CREATE INDEX idx_fact_orders_carrier_date   ON fact_orders(order_delivered_carrier_date);

DROP TABLE IF EXISTS fact_order_items;
CREATE TABLE fact_order_items (
    order_id            TEXT NOT NULL,
    order_item_id       INTEGER NOT NULL,
    product_id          TEXT NOT NULL,
    seller_id           TEXT NOT NULL,
    shipping_limit_date TEXT NOT NULL,
    price               REAL NOT NULL,
    freight_value       REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id)   REFERENCES fact_orders(order_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (seller_id)  REFERENCES dim_sellers(seller_id)
);

CREATE INDEX idx_fact_items_order   ON fact_order_items(order_id);
CREATE INDEX idx_fact_items_product ON fact_order_items(product_id);
CREATE INDEX idx_fact_items_seller  ON fact_order_items(seller_id);

-- ---------------------------------------------------------------------------
-- Aggregated tables (populated in 02_build_daily_shipments.sql)
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS fact_daily_shipments_by_state;
CREATE TABLE fact_daily_shipments_by_state (
    shipment_date  TEXT NOT NULL,
    customer_state TEXT NOT NULL CHECK (length(customer_state) = 2),
    n_shipments    INTEGER NOT NULL CHECK (n_shipments >= 0),
    PRIMARY KEY (shipment_date, customer_state)
);

CREATE INDEX idx_fact_daily_state ON fact_daily_shipments_by_state(customer_state);
CREATE INDEX idx_fact_daily_date  ON fact_daily_shipments_by_state(shipment_date);

-- Re-enable foreign key enforcement for runtime queries
PRAGMA foreign_keys = ON;
