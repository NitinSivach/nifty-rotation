"""Tests for settings.py config loading, merge, and data-file resolution."""
import json

import pytest

from settings import (
    SETTINGS,
    _merge,
    app_config,
    backtest_config,
    data_file,
    load_config,
)


def test_load_config_returns_committed_config():
    cfg = load_config()
    assert "app" in cfg and "backtest" in cfg and "advisor" in cfg
    assert "data_files" in cfg


def test_load_config_respects_explicit_path(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"app": {"layout": "narrow"}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["app"]["layout"] == "narrow"


def test_merge_overrides_scalar_and_recurses_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"a": 2, "nested": {"y": 99}}
    merged = _merge(base, override)
    assert merged["a"] == 2
    assert merged["nested"] == {"x": 1, "y": 99}  # x preserved, y overridden


def test_merge_replaces_scalar_with_dict():
    base = {"a": {"x": 1}}
    override = {"a": 5}
    assert _merge(base, override)["a"] == 5


def test_config_helpers_expose_sections():
    assert isinstance(app_config(), dict)
    assert isinstance(backtest_config(), dict)
    assert "trading_days_year" in backtest_config()


def test_data_file_resolves_relative_to_project_root():
    path = data_file("nifty50_map")
    assert path.exists()
    assert path.name == "nifty50_map.json"


def test_data_file_unknown_name_raises():
    with pytest.raises(KeyError, match="No data_files entry"):
        data_file("does_not_exist")


def test_local_override_merges_when_present(tmp_path, monkeypatch):
    import settings as settings_mod

    override = tmp_path / "config.local.json"
    override.write_text(json.dumps({"app": {"layout": "narrow"}}), encoding="utf-8")
    monkeypatch.setattr(settings_mod, "_LOCAL_OVERRIDE_PATH", override)
    cfg = load_config()
    assert cfg["app"]["layout"] == "narrow"
    assert "backtest" in cfg  # still merged from base


def test_committed_settings_is_loadable_snapshot():
    # Guards against a config.json edit that silently breaks the app.
    assert SETTINGS["app"]["layout"] in ("wide", "narrow")
    assert callable(data_file)