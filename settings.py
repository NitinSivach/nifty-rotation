"""Central configuration loader for the Nifty sector-rotation project.

All tunable values live in ``config.json``. This module exposes them as a
plain ``SETTINGS`` dict plus small helpers. No secrets are stored here:
API keys come from environment variables or Streamlit Cloud secrets.
"""
import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_LOCAL_OVERRIDE_PATH = Path(__file__).resolve().parent / "config.local.json"


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` on top of ``base`` (dicts recurse)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None = None) -> dict:
    """Read the JSON config file (defaults to ./config.json next to this file).

    When ``config.local.json`` exists next to it, its values are deep-merged on
    top, letting you experiment locally without touching the committed file.
    """
    config_path = Path(path) if path else _CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        settings = json.load(fh)
    if path is None and _LOCAL_OVERRIDE_PATH.exists():
        with _LOCAL_OVERRIDE_PATH.open("r", encoding="utf-8") as fh:
            settings = _merge(settings, json.load(fh))
    return settings


SETTINGS = load_config()


def app_config() -> dict:
    return SETTINGS.get("app", {})


def backtest_config() -> dict:
    return SETTINGS.get("backtest", {})


def advisor_config() -> dict:
    return SETTINGS.get("advisor", {})


def data_file(name: str) -> Path:
    """Resolve a data-file path relative to the project root."""
    relative = SETTINGS.get("data_files", {}).get(name)
    if not relative:
        raise KeyError(f"No data_files entry named {name!r}")
    return Path(__file__).resolve().parent / relative