"""Tests for MarketSimulationRunner — integration tests."""

import math

from app.services.lob_engine import LOBEngine
from app.services.market_agents import Fundamentalist, InformedTrader, NoiseTrader
from app.services.mm_agent import HelixMMAgent
from app.services.simulation_runner import MarketSimulationRunner


def _make_runner(max_ticks: int = 1000, seed: int = 42) -> MarketSimulationRunner:
    lob = LOBEngine(tick_size=0.01)
    agents = [
        NoiseTrader("noise_1", {"arrival_rate": 1.0, "seed": seed}),
        NoiseTrader("noise_2", {"arrival_rate": 0.8, "seed": seed + 1}),
        NoiseTrader("noise_3", {"arrival_rate": 0.6, "seed": seed + 2}),
        Fundamentalist("fund_1", {"fundamental_value": 100.0, "threshold": 0.5, "update_speed": 0.01, "seed": seed + 3}),
    ]
    informed = InformedTrader("informed_1", {"accuracy": 0.7, "arrival_rate": 0.3, "seed": seed + 4})
    # Generate a simple price path for informed trader
    import numpy as np
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(max_ticks + 50):
        prices.append(prices[-1] + rng.normal(0, 0.1))
    informed.set_future_prices(prices)
    agents.append(informed)

    mm = HelixMMAgent("mm_helix", {
        "gamma": 0.1,
        "k": 1.5,
        "sigma": 0.3,
        "quantity": 1.0,
        "max_inventory": 10,
    })

    return MarketSimulationRunner(
        lob=lob, agents=agents, mm_agent=mm,
        max_ticks=max_ticks, initial_mid_price=100.0, seed=seed,
    )


class TestSimulationCompletion:
    def test_simulation_runs_to_completion(self):
        """1000-tick simulation completes without error."""
        runner = _make_runner(max_ticks=1000)
        records = runner.run()
        assert len(records) == 1000

    def test_simulation_produces_tick_records(self):
        """run() returns max_ticks TickRecords."""
        runner = _make_runner(max_ticks=100)
        records = runner.run()
        assert len(records) == 100
        for r in records:
            assert r.tick >= 0
            assert r.mid_price > 0

    def test_simulation_has_trades(self):
        """At least 1 trade occurs during simulation."""
        runner = _make_runner(max_ticks=100)
        records = runner.run()
        total_trades = sum(r.num_trades for r in records)
        assert total_trades > 0

    def test_short_simulation_runs(self):
        """Very short simulation (10 ticks) works."""
        runner = _make_runner(max_ticks=10)
        records = runner.run()
        assert len(records) == 10


class TestSimulationQuality:
    def test_mm_pnl_is_finite(self):
        """MM PnL is a finite number (not NaN/Inf)."""
        runner = _make_runner(max_ticks=500)
        records = runner.run()
        for r in records:
            assert math.isfinite(r.mm_pnl)

    def test_mm_inventory_bounded(self):
        """MM inventory stays near max_inventory bounds."""
        runner = _make_runner(max_ticks=500)
        records = runner.run()
        for r in records:
            assert abs(r.mm_inventory) <= 15  # max_inventory=10 + some buffer

    def test_mid_price_series_continuous(self):
        """No >10% price jumps between consecutive ticks."""
        runner = _make_runner(max_ticks=500)
        records = runner.run()
        for i in range(1, len(records)):
            prev = records[i - 1].mid_price
            curr = records[i].mid_price
            if prev > 0:
                change = abs(curr - prev) / prev
                assert change < 0.10, f"Jump of {change:.1%} at tick {i}"

    def test_results_summary_contains_required_fields(self):
        """get_results_summary() has all required fields."""
        runner = _make_runner(max_ticks=100)
        runner.run()
        summary = runner.get_results_summary()
        required = [
            "total_ticks", "total_trades", "mm_final_pnl",
            "mm_final_inventory", "sharpe_ratio", "max_drawdown",
            "inventory_stats", "spread_stats",
        ]
        for field in required:
            assert field in summary, f"Missing field: {field}"
