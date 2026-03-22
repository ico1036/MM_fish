"""CSV tick data replayer for LOB simulation with real market data."""

import csv
from pathlib import Path

import numpy as np

from app.models.market import LOBSnapshot


class DataReplayer:
    """
    Replays historical tick data to drive market simulation.

    Reads CSV with columns: timestamp, price, quantity, side
    Produces LOBSnapshot-like data and a synthetic future price path.
    """

    def __init__(self, csv_path: Path, max_ticks: int = 1000) -> None:
        self.csv_path = csv_path
        self.max_ticks = max_ticks
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load CSV tick data."""
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._prices.append(float(row["price"]))
                self._volumes.append(float(row["quantity"]))

    @property
    def num_ticks(self) -> int:
        return len(self._prices)

    def get_price_path(self) -> list[float]:
        """Return the price path (for informed trader look-ahead)."""
        return self._prices[:]

    def get_initial_mid_price(self) -> float:
        """Return the first price as initial mid."""
        if not self._prices:
            return 100.0
        return self._prices[0]

    def get_volatility_estimate(self) -> float:
        """Estimate sigma from price returns."""
        if len(self._prices) < 2:
            return 0.01
        arr = np.array(self._prices[:min(500, len(self._prices))])
        returns = np.diff(np.log(arr))
        return float(np.std(returns)) if len(returns) > 0 else 0.01

    def get_snapshot_at(self, tick: int) -> LOBSnapshot | None:
        """Get a synthetic LOB snapshot at a given tick index."""
        if tick >= len(self._prices):
            return None
        price = self._prices[tick]
        spread = price * 0.001  # 0.1% synthetic spread
        return LOBSnapshot(
            timestamp=float(tick),
            best_bid=price - spread / 2,
            best_ask=price + spread / 2,
            mid_price=price,
            spread=spread,
        )
