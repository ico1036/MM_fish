"""Tests for DataReplayer — CSV tick data loading and replay."""

import csv
from pathlib import Path

import pytest

from app.services.data_replayer import DataReplayer


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small sample CSV file."""
    csv_path = tmp_path / "test_trades.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "price", "quantity", "side"])
        writer.writeheader()
        for i in range(100):
            writer.writerow({
                "timestamp": 1000000 + i * 100,
                "price": f"{100.0 + i * 0.01:.4f}",
                "quantity": f"{5.0:.2f}",
                "side": "bid" if i % 2 == 0 else "ask",
            })
    return csv_path


class TestDataReplayer:
    def test_loads_data(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        assert replayer.num_ticks == 100

    def test_initial_mid_price(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        assert replayer.get_initial_mid_price() == 100.0

    def test_price_path(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        path = replayer.get_price_path()
        assert len(path) == 100
        assert path[0] == 100.0
        assert path[-1] == pytest.approx(100.99, abs=0.01)

    def test_volatility_estimate(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        sigma = replayer.get_volatility_estimate()
        assert sigma > 0
        assert sigma < 0.01  # small for linear price path

    def test_snapshot_at_tick(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        snap = replayer.get_snapshot_at(0)
        assert snap is not None
        assert snap.mid_price == 100.0
        assert snap.spread > 0
        assert snap.best_bid < snap.mid_price
        assert snap.best_ask > snap.mid_price

    def test_snapshot_beyond_range(self, sample_csv: Path):
        replayer = DataReplayer(sample_csv)
        snap = replayer.get_snapshot_at(500)
        assert snap is None

    def test_empty_file_defaults(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "price", "quantity", "side"])
            writer.writeheader()

        replayer = DataReplayer(csv_path)
        assert replayer.num_ticks == 0
        assert replayer.get_initial_mid_price() == 100.0
        assert replayer.get_volatility_estimate() == 0.01
