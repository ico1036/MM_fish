"""Generate realistic RWA coin (ONDO-like) tick data for simulation."""

import csv
from pathlib import Path

import numpy as np


def generate_trades(
    n_trades: int = 2000,
    initial_price: float = 0.85,  # ONDO-like price
    volatility: float = 0.002,
    seed: int = 42,
) -> Path:
    """Generate synthetic RWA coin trades with realistic microstructure."""
    rng = np.random.default_rng(seed)
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ondousdt_trades.csv"

    price = initial_price
    base_ts = 1711100000000  # Some realistic epoch ms

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "price", "quantity", "side"])
        writer.writeheader()

        for i in range(n_trades):
            # GBM-like price evolution with mean reversion
            drift = -0.001 * (price - initial_price) / initial_price
            ret = drift + rng.normal(0, volatility)
            price *= (1 + ret)
            price = max(price * 0.5, min(price * 1.5, price))  # bounds

            qty = rng.exponential(50.0)  # ONDO-like qty
            side = "bid" if rng.random() < 0.5 else "ask"
            ts = base_ts + i * rng.integers(100, 2000)

            writer.writerow({
                "timestamp": ts,
                "price": f"{price:.4f}",
                "quantity": f"{qty:.2f}",
                "side": side,
            })

    print(f"Generated {n_trades} trades to {out_path}")
    print(f"Price range: {initial_price:.4f} (start)")
    return out_path


if __name__ == "__main__":
    generate_trades()
