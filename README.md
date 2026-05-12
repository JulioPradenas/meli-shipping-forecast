# Shipping Demand Forecasting System

[![CI](https://github.com/JulioPradenas/meli-shipping-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/JulioPradenas/meli-shipping-forecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end shipping demand forecasting system for LATAM e-commerce, focused on operational capacity planning.

## Problem

LATAM e-commerce platforms operate continent-wide logistics networks. Operational capacity planning (warehouse staff, fleet, distribution-center space) depends critically on predicting how many packages will be processed per region and per day.

- **Under-forecasting** → operational collapse, delays, poor customer experience
- **Over-forecasting** → unnecessary operational cost (idle staff, underused fleet)

## Goal

Given the historical record of shipments per region, predict the daily package volume per state/region with a 7–30 day horizon, with enough precision to inform capacity decisions.

## Tech stack

- **Language**: Python 3.11
- **Dependency management**: uv
- **Data**: pandas, SQLAlchemy, pyarrow
- **ML**: scikit-learn, LightGBM, statsmodels, Prophet
- **Hyperparameter tuning**: Optuna
- **Explainability**: SHAP
- **Experiment tracking**: MLflow
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Infrastructure**: Docker, GitHub Actions
- **Quality**: pytest, ruff, mypy, pre-commit

## Project structure

\`\`\`
meli-shipping-forecast/
├── src/shipping_forecast/   # Main package
│   ├── config/              # Configuration (Pydantic settings)
│   ├── data/                # Data ingestion and SQL
│   ├── features/            # Feature engineering (OOP)
│   ├── models/              # Forecasting models
│   ├── evaluation/          # Metrics and business evaluation
│   ├── pipelines/           # End-to-end orchestration
│   └── utils/               # Logging and helpers
├── tests/                   # Unit and integration tests
├── notebooks/               # EDA and exploratory analysis
├── sql/                     # SQL queries
├── api/                     # FastAPI service
├── app/                     # Streamlit dashboard
├── docker/                  # Dockerfiles
└── .github/workflows/       # CI/CD
\`\`\`

## Setup

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (modern Python dependency manager)
- macOS users: \`brew install libomp\` (required by LightGBM)

### Installation

\`\`\`bash
git clone https://github.com/JulioPradenas/meli-shipping-forecast.git
cd meli-shipping-forecast
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-extras
\`\`\`

### Common commands

\`\`\`bash
make test        # Run tests with coverage
make lint        # Run ruff linter
make typecheck   # Run mypy
make check       # Run all quality checks
make format      # Auto-format code
\`\`\`

## Project status

🚧 **Work in progress**

- [x] **Phase 1**: Project setup, tooling, CI/CD
- [ ] **Phase 2**: Data ingestion and SQL feature engineering
- [ ] **Phase 3**: Exploratory data analysis
- [ ] **Phase 4**: Feature engineering (OOP)
- [ ] **Phase 5**: Baseline models
- [ ] **Phase 6**: Advanced ML models
- [ ] **Phase 7**: Business evaluation
- [ ] **Phase 8**: FastAPI service
- [ ] **Phase 9**: Streamlit dashboard and deployment

## License

MIT
