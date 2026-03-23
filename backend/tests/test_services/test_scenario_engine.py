# backend/tests/test_services/test_scenario_engine.py
import pytest
import numpy as np
from app.services.scenario_engine import ScenarioEngine, Scenario


class TestScenario:
    def test_normal_listing(self):
        s = Scenario.normal_listing(initial_price=5620.0, num_ticks=100, seed=42)
        assert len(s.index_prices) == 100
        assert s.index_prices[0] == pytest.approx(5620.0, abs=1.0)
        assert s.name == "normal_listing"

    def test_sp500_crash(self):
        s = Scenario.sp500_crash(initial_price=5620.0, num_ticks=100, crash_tick=50, crash_pct=0.03, seed=42)
        assert s.index_prices[49] > s.index_prices[55]
        assert s.name == "sp500_crash"

    def test_funding_spike(self):
        s = Scenario.funding_spike(initial_price=5620.0, num_ticks=100, spike_tick=50, seed=42)
        assert s.name == "funding_spike"
        assert s.events[50] == "funding_spike"


class TestScenarioEngine:
    def test_get_index_price(self):
        s = Scenario.normal_listing(initial_price=100.0, num_ticks=50, seed=42)
        engine = ScenarioEngine(s)
        assert engine.get_index_price(0) == pytest.approx(100.0, abs=1.0)

    def test_get_event(self):
        s = Scenario.sp500_crash(initial_price=100.0, num_ticks=50, crash_tick=25, crash_pct=0.03, seed=42)
        engine = ScenarioEngine(s)
        assert engine.get_event(25) == "sp500_crash"
        assert engine.get_event(10) is None
