"""Tests for the screener's indicator maths and its failure handling.

These cover the two failure modes fixed in the commits before this one: the
RSI division producing NaN on a flat window, and a run where every fetch
fails reaching rank() and dying there with a KeyError.

Run from the repo root:

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import screener  # noqa: E402  (import after the path insert)


# ---------------------------------------------------------------------------
# rsi()
# ---------------------------------------------------------------------------


def test_rsi_is_neutral_on_a_flat_window():
    """No gains and no losses is neither overbought nor oversold.

    up/dn is 0/0 here. That used to produce NaN, which Series.rank() then
    skips, so the ticker silently dropped out of the RSI component of the
    score rather than scoring as neutral.
    """
    flat = pd.Series([50.0] * 20)
    assert screener.rsi(flat).iloc[-1] == 50.0


def test_rsi_is_100_when_every_day_gains():
    rising = pd.Series([float(i) for i in range(1, 25)])
    assert screener.rsi(rising).iloc[-1] == 100.0


def test_rsi_is_0_when_every_day_loses():
    falling = pd.Series([float(25 - i) for i in range(24)])
    assert screener.rsi(falling).iloc[-1] == 0.0


def test_rsi_matches_the_hand_computed_value_on_a_mixed_window():
    series = pd.Series([10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18])
    assert screener.rsi(series).iloc[-1] == pytest.approx(66.6667, abs=1e-3)


def test_rsi_leaves_the_warmup_period_as_nan():
    """The warmup runs to index n, not n-1.

    diff() puts a NaN at position 0, so the first window that contains no NaN
    at all ends at index n rather than n-1.
    """
    series = pd.Series([float(i) for i in range(1, 25)])
    values = screener.rsi(series, n=14)
    assert values.first_valid_index() == 14
    assert values.iloc[:14].isna().all()
    assert not values.iloc[14:].isna().any()


# ---------------------------------------------------------------------------
# rank()
# ---------------------------------------------------------------------------


def test_rank_puts_the_best_composite_score_first():
    df = pd.DataFrame(
        [
            {"ticker": "GOOD", "momentum_12_1": 40.0, "vol_30d": 10.0,
             "rsi_14": 52.0, "dist_ma50": 8.0},
            {"ticker": "BAD", "momentum_12_1": -40.0, "vol_30d": 90.0,
             "rsi_14": 20.0, "dist_ma50": -8.0},
        ]
    )
    # GOOD wins every component: best momentum, lowest vol, RSI closest to 50,
    # furthest above its 50-day average.
    assert list(screener.rank(df)["ticker"]) == ["GOOD", "BAD"]


def test_rank_scores_a_single_ticker_as_itself():
    df = pd.DataFrame(
        [{"ticker": "ONLY", "momentum_12_1": 5.0, "vol_30d": 20.0,
          "rsi_14": 60.0, "dist_ma50": 1.0}]
    )
    ranked = screener.rank(df)
    assert ranked["score"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_aborts_when_no_ticker_produced_data(monkeypatch, tmp_path):
    """An all-failed run must stop before it blanks LATEST.md and README.md."""
    monkeypatch.setattr(screener, "analyze", lambda _ticker: None)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        screener.run()

    # nothing was written on the way out
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "LATEST.md").exists()
