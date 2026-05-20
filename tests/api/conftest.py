"""Test fixtures for the API test suite.

Sets MELI_API_AUTO_TRAIN=true at import time so that when TestClient
spins up the lifespan, the app trains a fast-retrain model if no
joblib exists at the configured path. This makes the test suite
self-contained: no need to run `make train-model` first.
"""

from __future__ import annotations

import os

os.environ.setdefault("MELI_API_AUTO_TRAIN", "true")
