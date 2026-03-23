# backend/tests/test_services/test_liquidation_engine.py
import pytest
from app.models.perpetual import Position, LiquidationEvent
from app.services.liquidation_engine import LiquidationEngine


class TestLiquidationEngine:
    def test_no_liquidation_healthy(self):
        engine = LiquidationEngine(maintenance_margin_ratio=0.05)
        pos = Position(agent_id="a1", side="long", size=10, entry_price=100, leverage=5, margin=200)
        result = engine.check_liquidation(tick=1, position=pos, mark_price=100.0)
        assert result is None

    def test_liquidation_triggered(self):
        engine = LiquidationEngine(maintenance_margin_ratio=0.05)
        pos = Position(agent_id="a1", side="long", size=10, entry_price=100, leverage=5, margin=200)
        result = engine.check_liquidation(tick=50, position=pos, mark_price=82.0)
        assert result is not None
        assert isinstance(result, LiquidationEvent)
        assert result.agent_id == "a1"
        assert result.side == "long"
        assert result.loss > 0
        assert result.loss == pytest.approx(180.0)

    def test_short_liquidation(self):
        engine = LiquidationEngine(maintenance_margin_ratio=0.05)
        pos = Position(agent_id="a2", side="short", size=10, entry_price=100, leverage=5, margin=200)
        result = engine.check_liquidation(tick=50, position=pos, mark_price=118.0)
        assert result is not None
        assert result.side == "short"

    def test_batch_liquidations(self):
        engine = LiquidationEngine(maintenance_margin_ratio=0.05)
        positions = [
            Position(agent_id="healthy", side="long", size=10, entry_price=100, leverage=2, margin=500),
            Position(agent_id="rekt", side="long", size=10, entry_price=100, leverage=20, margin=50),
        ]
        events = engine.check_all(tick=10, positions=positions, mark_price=97.0)
        assert len(events) == 1
        assert events[0].agent_id == "rekt"
