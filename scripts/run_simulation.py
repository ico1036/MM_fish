"""
Helix MM Simulation — run with real or synthetic data.

Usage:
    uv run python scripts/run_simulation.py
    uv run python scripts/run_simulation.py --ticks 5000 --gamma 0.2
    uv run python scripts/run_simulation.py --data data/ondousdt_trades.csv
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.data_replayer import DataReplayer
from app.services.lob_engine import LOBEngine
from app.services.market_agents import Fundamentalist, InformedTrader, NoiseTrader
from app.services.metrics import generate_report
from app.services.mm_agent import HelixMMAgent
from app.services.simulation_runner import MarketSimulationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Helix MM Simulation")
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=None, help="Auto-estimated from data if not set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None, help="Path to CSV tick data")
    args = parser.parse_args()

    # Data-driven setup
    initial_mid = 100.0
    sigma = args.sigma or 0.3
    future_prices = None

    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            replayer = DataReplayer(data_path, max_ticks=args.ticks)
            initial_mid = replayer.get_initial_mid_price()
            sigma = args.sigma or replayer.get_volatility_estimate()
            future_prices = replayer.get_price_path()
            print(f"Data: {data_path.name} | Initial price: {initial_mid:.4f} | σ: {sigma:.6f}")
        else:
            print(f"Warning: {data_path} not found, using synthetic mode")

    # Auto-calibrate k so spread ≈ 10bps of price
    # From A-S: spread ≈ (2/gamma) * ln(1 + gamma/k) when sigma^2 term is small
    # Solve for k: k = gamma / (exp(gamma * target / 2) - 1)
    import numpy as np

    target_spread = initial_mid * 0.002  # target 20bps spread
    gamma = args.gamma
    k_auto = gamma / (np.exp(gamma * target_spread / 2.0) - 1.0)
    k = args.k if args.k != 1.5 else k_auto  # use auto unless user overrides
    print(f"Params: gamma={gamma}, k={k:.2f}, sigma={sigma:.6f}, target_spread={target_spread:.6f}")

    # Setup LOB
    tick_size = initial_mid * 0.0001  # 0.01% of price
    lob = LOBEngine(tick_size=tick_size)

    # Agents — scale quantities and spreads to price level
    agents = [
        NoiseTrader("noise_1", {"arrival_rate": 1.0, "max_spread": initial_mid * 0.005, "quantity": 1.0, "seed": args.seed}),
        NoiseTrader("noise_2", {"arrival_rate": 0.8, "max_spread": initial_mid * 0.005, "quantity": 1.0, "seed": args.seed + 1}),
        NoiseTrader("noise_3", {"arrival_rate": 0.6, "max_spread": initial_mid * 0.005, "quantity": 1.0, "seed": args.seed + 2}),
        Fundamentalist("fund_1", {
            "fundamental_value": initial_mid,
            "threshold": initial_mid * 0.003,
            "quantity": 2.0,
            "update_speed": initial_mid * 0.0001,
            "seed": args.seed + 3,
        }),
    ]

    informed = InformedTrader("informed_1", {
        "accuracy": 0.7,
        "arrival_rate": 0.3,
        "quantity": 3.0,
        "seed": args.seed + 4,
    })
    if future_prices:
        informed.set_future_prices(future_prices)
    else:
        # Generate synthetic path
        import numpy as np
        rng = np.random.default_rng(args.seed)
        prices = [initial_mid]
        for _ in range(args.ticks + 50):
            prices.append(prices[-1] * (1 + rng.normal(0, sigma)))
        informed.set_future_prices(prices)
    agents.append(informed)

    # MM Agent
    mm = HelixMMAgent("mm_helix", {
        "gamma": gamma,
        "k": k,
        "sigma": sigma,
        "quantity": 1.0,
        "max_inventory": 10,
    })

    # Run
    runner = MarketSimulationRunner(
        lob=lob, agents=agents, mm_agent=mm,
        max_ticks=args.ticks, initial_mid_price=initial_mid, seed=args.seed,
    )

    records = runner.run()
    report = generate_report(records)

    # Print results
    print("=" * 60)
    print("HELIX MM SIMULATION REPORT")
    print("=" * 60)
    print(f"  Total Ticks:       {report['total_ticks']}")
    print(f"  Total Trades:      {report['total_trades']}")
    print(f"  MM Final PnL:      {report['mm_final_pnl']:.4f}")
    print(f"  MM Final Inventory:{report['mm_final_inventory']:.1f}")
    print(f"  Sharpe Ratio:      {report['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:      {report['max_drawdown']:.2%}")
    inv = report['inventory_stats']
    print(f"  Inventory Mean:    {inv['mean']:.2f}")
    print(f"  Inventory Std:     {inv['std']:.2f}")
    print(f"  Inventory Max Abs: {inv['max_abs']:.1f}")
    spr = report['spread_stats']
    print(f"  Avg Spread:        {spr['mean']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
