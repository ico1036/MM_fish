"""Tests for Binance aggTrades parquet data loader."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path

from app.services.binance_data import load_trades, downsample_trades, get_price_path


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample parquet file mimicking Binance aggTrades."""
    n = 1000
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    quantities = rng.exponential(0.1, n)
    base = np.datetime64("2025-01-01T00:00:00", "ns")
    offsets = np.arange(n, dtype="int64") * np.timedelta64(1, "ms")
    timestamps = pa.array(base + offsets, type=pa.timestamp("ns"))

    table = pa.table(
        {
            "price": prices,
            "quantity": quantities,
            "timestamp": timestamps,
            "is_buyer_maker": rng.choice([True, False], n),
        }
    )

    symbol_dir = tmp_path / "TESTUSDT" / "2025"
    symbol_dir.mkdir(parents=True)
    pq.write_table(table, symbol_dir / "futures-TESTUSDT-aggTrades-2025-01.parquet")
    return tmp_path


class TestLoadTrades:
    def test_loads_all_trades(self, sample_parquet):
        result = load_trades("TESTUSDT", sample_parquet)
        assert len(result["prices"]) == 1000
        assert len(result["quantities"]) == 1000
        assert len(result["is_buyer_maker"]) == 1000

    def test_loads_with_max_rows(self, sample_parquet):
        result = load_trades("TESTUSDT", sample_parquet, max_rows=500)
        assert len(result["prices"]) == 500

    def test_raises_on_missing_symbol(self, sample_parquet):
        with pytest.raises(FileNotFoundError):
            load_trades("NONEXIST", sample_parquet)

    def test_raises_on_empty_dir(self, tmp_path):
        (tmp_path / "EMPTYUSDT").mkdir()
        with pytest.raises(FileNotFoundError):
            load_trades("EMPTYUSDT", tmp_path)

    def test_prices_are_floats(self, sample_parquet):
        result = load_trades("TESTUSDT", sample_parquet, max_rows=10)
        assert result["prices"].dtype in [np.float64, np.float32]


class TestDownsampleTrades:
    def test_downsample_reduces_length(self):
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, 1000))
        quantities = rng.exponential(0.1, 1000)
        is_buyer = rng.choice([True, False], 1000)

        result = downsample_trades(prices, quantities, is_buyer, every_n=100)
        assert len(result["prices"]) == 10
        assert len(result["buy_volume_ratio"]) == 10

    def test_downsample_preserves_first_price(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        qty = np.ones(6)
        buyer = np.array([True, False, True, False, True, False])

        result = downsample_trades(prices, qty, buyer, every_n=2)
        assert result["prices"][0] == 100.0
        assert result["prices"][1] == 102.0
        assert result["prices"][2] == 104.0

    def test_buy_volume_ratio_bounded(self):
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, 1000))
        quantities = rng.exponential(0.1, 1000)
        is_buyer = rng.choice([True, False], 1000)

        result = downsample_trades(prices, quantities, is_buyer, every_n=100)
        assert all(0 <= r <= 1 for r in result["buy_volume_ratio"])


class TestGetPricePath:
    def test_returns_correct_length(self, sample_parquet):
        path = get_price_path("TESTUSDT", sample_parquet, max_ticks=5, downsample_n=10)
        assert len(path) == 5

    def test_returns_numpy_array(self, sample_parquet):
        path = get_price_path("TESTUSDT", sample_parquet, max_ticks=5, downsample_n=10)
        assert isinstance(path, np.ndarray)
