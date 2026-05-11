"""Sanity tests for the package itself."""

import shipping_forecast
from shipping_forecast import get_version


def test_package_has_version() -> None:
    """The package must expose a __version__ attribute."""
    assert hasattr(shipping_forecast, "__version__")
    assert isinstance(shipping_forecast.__version__, str)
    assert len(shipping_forecast.__version__) > 0


def test_version_follows_semver_format() -> None:
    """The version must follow MAJOR.MINOR.PATCH format."""
    parts = shipping_forecast.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_get_version_returns_string() -> None:
    """get_version() must return the version as a string."""
    version = get_version()
    assert isinstance(version, str)
    assert version == shipping_forecast.__version__
