"""Tests for LLM simulation runner."""

from unittest.mock import MagicMock
import pytest

from app.models.agent_profile import TraderProfile
from app.models.market import LOBSnapshot, Side
from app.services.lob_engine import LOBEngine
from app.services.llm_agents import LLMTrader
from app.services.llm_simulation_runner import LLMSimulationRunner
from app.services.market_agents import NoiseTrader
from app.services.mm_agent import HelixMMAgent


@pytest.fixture
def mm_agent():
    return HelixMMAgent(
        agent_id="mm",
        params={
            "gamma": 0.1,
            "k": 1.5,
            "sigma": 0.3,
            "T": 1.0,
            "quantity": 1.0,
            "max_inventory": 10,
        },
    )


@pytest.fixture
def noise_agents():
    return [
        NoiseTrader("noise_1", {"arrival_rate": 0.5, "market_order_pct": 0.3, "max_spread": 2.0, "quantity": 1.0, "seed": 42}),
        NoiseTrader("noise_2", {"arrival_rate": 0.5, "market_order_pct": 0.3, "max_spread": 2.0, "quantity": 1.0, "seed": 43}),
    ]


@pytest.fixture
def mock_llm_agents():
    """Create LLM agents with mocked LLM client."""
    client = MagicMock()
    client.chat_json.return_value = {
        "action": "BUY",
        "type": "LIMIT",
        "price": 99.5,
        "quantity": 1.0,
        "reason": "test",
    }
    profiles = [
        TraderProfile(
            agent_id=f"llm_{i}",
            trader_type="momentum",
            persona="Test trader",
            risk_appetite=0.5,
        )
        for i in range(3)
    ]
    agents = []
    for p in profiles:
        agent = LLMTrader(p, client, base_quantity=1.0, seed=42 + hash(p.agent_id) % 100)
        agent._arrival_rate = 0.3
        agents.append(agent)
    return agents


class TestSimulationCompletion:
    def test_runs_to_completion_as_mode(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=100,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        records = runner.run()
        assert len(records) == 100

    def test_runs_to_completion_llm_mode(self, mm_agent, mock_llm_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=mock_llm_agents,
            max_ticks=50,
            initial_mid_price=100.0,
            seed=42,
            mode="llm",
        )
        records = runner.run()
        assert len(records) == 50

    def test_produces_tick_records(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=100,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        records = runner.run()
        assert all(r.tick >= 0 for r in records)
        assert all(r.mid_price > 0 for r in records)


class TestSimulationQuality:
    def test_mm_pnl_is_finite(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=200,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        records = runner.run()
        assert all(abs(r.mm_pnl) < 1e6 for r in records)

    def test_mid_price_positive(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=200,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        records = runner.run()
        assert all(r.mid_price > 0 for r in records)


class TestResultsSummary:
    def test_summary_has_required_fields(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=100,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        runner.run()
        summary = runner.get_results_summary()
        assert "total_ticks" in summary
        assert "total_trades" in summary
        assert "mode" in summary
        assert summary["mode"] == "as"

    def test_llm_summary_has_agent_stats(self, mm_agent, mock_llm_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=mock_llm_agents,
            max_ticks=50,
            initial_mid_price=100.0,
            seed=42,
            mode="llm",
        )
        runner.run()
        summary = runner.get_results_summary()
        assert "llm_agent_stats" in summary
        assert summary["llm_agent_stats"]["total_llm_agents"] == 3


class TestPriceAndSpreadSeries:
    def test_get_price_series(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=100,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        runner.run()
        prices = runner.get_price_series()
        assert len(prices) == 100
        assert all(p > 0 for p in prices)

    def test_get_spread_series(self, mm_agent, noise_agents):
        lob = LOBEngine()
        runner = LLMSimulationRunner(
            lob=lob,
            mm_agent=mm_agent,
            agents=noise_agents,
            max_ticks=100,
            initial_mid_price=100.0,
            seed=42,
            mode="as",
        )
        runner.run()
        spreads = runner.get_spread_series()
        assert len(spreads) == 100
        assert all(s >= 0 for s in spreads)
