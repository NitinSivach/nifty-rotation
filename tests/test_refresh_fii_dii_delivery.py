"""Tests for data/refresh_fii_dii.py delivery parsing and weekday helpers."""
import pandas as pd
import pytest

from data import refresh_fii_dii as rfd


class FakeResponse:
    def __init__(self, json_payload, status_code=200):
        self._payload = json_payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_delivery_builds_rows() -> None:
    session = type(
        "S",
        (),
        {
            "get": lambda self, url, params=None, timeout=None: FakeResponse(
                {
                    "data": [
                        {
                            "symbol": "RELIANCE",
                            "series": "EQ",
                            "security": "RELIANCE INDUSTRIES LTD",
                            "quantity": "100",
                            "deliverableQuantity": "75",
                        },
                        {
                            "symbol": "TCS",
                            "series": "EQ",
                            "security": "TATA CONSULTANCY SERVICES LTD",
                            "quantity": "200",
                            "deliverableQuantity": "40",
                        },
                    ]
                }
            )
        },
    )()
    df = rfd.fetch_delivery(session, pd.Timestamp("2026-08-10").date())
    assert list(df.columns) == [
        "date", "symbol", "series", "security",
        "quantity", "deliverable_quantity", "delivery_pct",
    ]
    assert len(df) == 2
    assert df["date"].iloc[0] == "2026-08-10"
    assert df["delivery_pct"].iloc[0] == 0.75
    assert df["delivery_pct"].iloc[1] == 0.20


def test_fetch_delivery_empty_payload_raises() -> None:
    session = type(
        "S", (), {"get": lambda self, url, params=None, timeout=None: FakeResponse({})}
    )()
    with pytest.raises(ValueError, match="contained no rows"):
        rfd.fetch_delivery(session, pd.Timestamp("2026-08-10").date())


def test_last_weekday_is_not_weekend() -> None:
    result = rfd.last_weekday()
    assert result.weekday() < 5


def test_last_weekday_handles_sunday_fallback() -> None:
    from datetime import date, timedelta
    # Force "today" to a Saturday and confirm we step back to Friday.
    saturday = date(2026, 8, 15)  # a Saturday
    original = rfd.date.today
    rfd.date = type("D", (), {"today": staticmethod(lambda: saturday)})
    try:
        result = rfd.last_weekday()
        assert result == saturday - timedelta(days=1)
    finally:
        rfd.date = original