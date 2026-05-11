# Shipping Demand Forecasting System

End-to-end shipping demand forecasting system for LATAM e-commerce, focused on operational capacity planning.

## Status

🚧 **Work in progress** — Phase 1: Project setup

## Tech stack

- **Language**: Python 3.11
- **Data**: pandas, SQLAlchemy, pyarrow
- **ML**: scikit-learn, LightGBM, statsmodels, Prophet
- **Hyperparameter tuning**: Optuna
- **Explainability**: SHAP
- **Experiment tracking**: MLflow
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Infra**: Docker, GitHub Actions
- **Quality**: pytest, ruff, mypy, pre-commit

## Setup

\`\`\`bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-extras
\`\`\`

## License

MIT

## macOS prerequisites

LightGBM requires OpenMP runtime, which is not bundled with macOS Clang:

\`\`\`bash
brew install libomp
\`\`\`
