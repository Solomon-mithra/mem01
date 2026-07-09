"""Smoke tests for package install / import."""

from __future__ import annotations


def test_import_mem01():
    import mem01

    assert mem01.__version__ == "0.1.0"


def test_version_is_semver_like():
    from mem01 import __version__

    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
