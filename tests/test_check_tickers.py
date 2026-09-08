"""Tests for check_tickers.py (offline-safe parts)."""
import json

import check_tickers as ct


def test_load_nifty50_map_returns_dict():
    mapping = ct.load_nifty50_map()
    assert "RELIANCE.NS" in mapping or len(mapping) >= 40
    assert all(isinstance(v, str) for v in mapping.values())


def test_load_nifty50_map_accepts_custom_path(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"A.NS": "Tech"}), encoding="utf-8")
    assert ct.load_nifty50_map(path) == {"A.NS": "Tech"}


def test_check_ticker_catches_yfinance_error(monkeypatch):
    class BrokenTicker:
        def history(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(ct.yf, "Ticker", lambda sym: BrokenTicker())
    rows, error = ct.check_ticker("DOES.NS")
    assert rows == 0
    assert "boom" in error


def test_check_ticker_returns_rows_on_success(monkeypatch):
    class OkTicker:
        def history(self, period="1mo"):
            return type("H", (), {"__len__": lambda self: 21})()

    monkeypatch.setattr(ct.yf, "Ticker", lambda sym: OkTicker())
    rows, error = ct.check_ticker("RELIANCE.NS")
    assert rows == 21
    assert error is None