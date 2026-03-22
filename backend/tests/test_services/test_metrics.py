"""Tests for performance metrics."""

from app.models.simulation import TickRecord
from app.services.metrics import (
    compute_inventory_stats,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_spread_stats,
    generate_report,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        """Positive returns → positive Sharpe."""
        pnl = [0.0, 0.5, 1.5, 2.0, 3.5, 5.0]  # varying positive returns
        assert compute_sharpe_ratio(pnl) > 0

    def test_zero_std(self):
        """Constant returns → Sharpe = 0 (no division by zero)."""
        pnl = [0.0, 1.0, 2.0, 3.0]  # constant return of 1.0
        assert compute_sharpe_ratio(pnl) == 0.0

    def test_negative_returns(self):
        """Negative returns → negative Sharpe."""
        pnl = [10.0, 8.0, 5.0, 2.0, 0.0]
        assert compute_sharpe_ratio(pnl) < 0

    def test_empty_series(self):
        """Empty or single element → 0."""
        assert compute_sharpe_ratio([]) == 0.0
        assert compute_sharpe_ratio([5.0]) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing → MDD = 0."""
        pnl = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_max_drawdown(pnl) == 0.0

    def test_known_case(self):
        """Known PnL [100, 80, 90, 70] → MDD = 30%."""
        pnl = [100.0, 80.0, 90.0, 70.0]
        mdd = compute_max_drawdown(pnl)
        assert abs(mdd - 0.3) < 1e-10

    def test_empty_series(self):
        assert compute_max_drawdown([]) == 0.0
        assert compute_max_drawdown([5.0]) == 0.0

    def test_all_negative(self):
        """Starting from zero, going negative."""
        pnl = [0.0, -5.0, -10.0]
        mdd = compute_max_drawdown(pnl)
        assert mdd == 1.0  # 100% drawdown from 0


class TestInventoryStats:
    def test_symmetric_inventory(self):
        """[-5, 5, -5, 5] → mean ≈ 0, max_abs = 5."""
        stats = compute_inventory_stats([-5.0, 5.0, -5.0, 5.0])
        assert abs(stats["mean"]) < 1e-10
        assert stats["max_abs"] == 5.0
        assert stats["zero_crossings"] == 3  # sign changes

    def test_empty_inventory(self):
        stats = compute_inventory_stats([])
        assert stats["mean"] == 0.0


class TestSpreadStats:
    def test_spread_stats(self):
        stats = compute_spread_stats([0.1, 0.2, 0.3])
        assert abs(stats["mean"] - 0.2) < 1e-10
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3

    def test_empty_spread(self):
        stats = compute_spread_stats([])
        assert stats["mean"] == 0.0


class TestGenerateReport:
    def test_report_from_tick_records(self):
        """Generate report from TickRecord list."""
        records = [
            TickRecord(tick=i, mid_price=100.0 + i * 0.1, spread=0.1, mm_inventory=float(i % 3 - 1), mm_pnl=float(i), num_trades=2)
            for i in range(100)
        ]
        report = generate_report(records)
        assert "total_ticks" in report
        assert report["total_ticks"] == 100
        assert "total_trades" in report
        assert report["total_trades"] == 200
        assert "mm_final_pnl" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "inventory_stats" in report
        assert "spread_stats" in report

    def test_empty_records(self):
        assert generate_report([]) == {}
