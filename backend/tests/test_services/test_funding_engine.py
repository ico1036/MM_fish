# backend/tests/test_services/test_funding_engine.py
import pytest
from app.models.perpetual import Position, FundingRate
from app.services.funding_engine import FundingEngine


class TestFundingEngine:
    def test_compute_rate_balanced(self):
        engine = FundingEngine(base_rate=0.0001, funding_interval=100)
        positions = [
            Position(agent_id="a1", side="long", size=10, entry_price=100, leverage=5, margin=200),
            Position(agent_id="a2", side="short", size=10, entry_price=100, leverage=5, margin=200),
        ]
        fr = engine.compute_funding_rate(tick=100, positions=positions, mark_price=100.0, index_price=100.0)
        assert abs(fr.rate) < 0.001

    def test_compute_rate_long_heavy(self):
        engine = FundingEngine(base_rate=0.0001, funding_interval=100)
        positions = [
            Position(agent_id="a1", side="long", size=100, entry_price=100, leverage=5, margin=2000),
            Position(agent_id="a2", side="short", size=10, entry_price=100, leverage=5, margin=200),
        ]
        fr = engine.compute_funding_rate(tick=100, positions=positions, mark_price=101.0, index_price=100.0)
        assert fr.rate > 0
        assert fr.long_pays_short is True

    def test_is_funding_tick(self):
        engine = FundingEngine(base_rate=0.0001, funding_interval=100)
        assert engine.is_funding_tick(0) is False
        assert engine.is_funding_tick(100) is True
        assert engine.is_funding_tick(200) is True
        assert engine.is_funding_tick(150) is False

    def test_apply_funding_updates_margin(self):
        engine = FundingEngine(base_rate=0.0001, funding_interval=100)
        pos = Position(agent_id="a1", side="long", size=10, entry_price=100, leverage=5, margin=200)
        fr = FundingRate(tick=100, rate=0.001, long_pays_short=True)
        new_margin = engine.apply_funding(pos, fr)
        assert new_margin == pytest.approx(199.0)
