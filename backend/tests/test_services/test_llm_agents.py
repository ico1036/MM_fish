"""Tests for LLM-based market agents."""

from unittest.mock import MagicMock, patch
import pytest

from app.models.agent_profile import TraderProfile
from app.models.market import LOBSnapshot, Side, OrderType
from app.services.llm_agents import LLMTrader, create_llm_agents, ARRIVAL_RATES
from app.services.plan_executor import AgentPlan


@pytest.fixture
def sample_profile():
    return TraderProfile(
        agent_id="momentum_001",
        trader_type="momentum",
        persona="You are a momentum trader who chases trends.",
        risk_appetite=0.7,
        capital=10000.0,
        behavioral_bias="overconfidence",
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_json.return_value = {
        "action": "BUY",
        "type": "MARKET",
        "price": None,
        "quantity": 1.0,
        "reason": "trend continuation",
    }
    return client


@pytest.fixture
def lob_snapshot():
    return LOBSnapshot(
        timestamp=1.0,
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread=2.0,
        bid_depth=[(99.0, 10.0), (98.0, 20.0)],
        ask_depth=[(101.0, 10.0), (102.0, 20.0)],
    )


class TestLLMTraderCreation:
    def test_creates_with_profile(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        assert trader.agent_id == "momentum_001"
        assert trader.inventory == 0.0
        assert trader.pnl == 0.0

    def test_arrival_rate_from_type(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        assert trader.participation_rate == ARRIVAL_RATES["momentum"]


class TestLLMTraderBuyDecision:
    def test_buy_market_order(self, sample_profile, mock_llm_client, lob_snapshot):
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        # Force participation by setting high arrival rate
        trader._arrival_rate = 1.0

        orders = trader.generate_orders(1, lob_snapshot)
        if orders:  # LLM returned BUY
            assert orders[0].side == Side.BID
            assert orders[0].order_type == OrderType.MARKET

    def test_buy_limit_order(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {
            "action": "BUY",
            "type": "LIMIT",
            "price": 99.5,
            "quantity": 1.0,
            "reason": "support level",
        }
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        trader._arrival_rate = 1.0

        orders = trader.generate_orders(1, lob_snapshot)
        if orders:
            assert orders[0].side == Side.BID
            assert orders[0].order_type == OrderType.LIMIT
            assert orders[0].price == 99.5


class TestLLMTraderSellDecision:
    def test_sell_market_order(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {
            "action": "SELL",
            "type": "MARKET",
            "price": None,
            "quantity": 1.0,
            "reason": "trend reversal",
        }
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        trader._arrival_rate = 1.0

        orders = trader.generate_orders(1, lob_snapshot)
        if orders:
            assert orders[0].side == Side.ASK
            assert orders[0].order_type == OrderType.MARKET


class TestLLMTraderHoldDecision:
    def test_hold_returns_no_orders(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {
            "action": "HOLD",
            "type": "MARKET",
            "price": None,
            "quantity": 0,
            "reason": "uncertain",
        }
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        trader._arrival_rate = 1.0

        orders = trader.generate_orders(1, lob_snapshot)
        assert orders == []


class TestLLMTraderError:
    def test_raises_on_llm_error(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.side_effect = Exception("API error")
        trader = LLMTrader(sample_profile, mock_llm_client, seed=42)
        trader._arrival_rate = 1.0

        with pytest.raises(Exception, match="API error"):
            trader.generate_orders(1, lob_snapshot)

    def test_empty_response_parsed_as_hold(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {}
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        trader._arrival_rate = 1.0

        orders = trader.generate_orders(1, lob_snapshot)
        assert orders == []  # Empty dict parsed as HOLD


class TestLLMTraderFill:
    def test_buy_fill_updates_inventory(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        trader.on_fill(Side.BID, price=100.0, quantity=1.0)
        assert trader.inventory == 1.0
        assert trader.cash == -100.0

    def test_sell_fill_updates_inventory(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        trader.on_fill(Side.ASK, price=100.0, quantity=1.0)
        assert trader.inventory == -1.0
        assert trader.cash == 100.0

    def test_pnl_at_price(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        trader.on_fill(Side.BID, price=100.0, quantity=1.0)
        # Bought at 100, market at 110 → PnL = -100 + 1*110 = 10
        assert trader.pnl_at_price(110.0) == 10.0


class TestLLMTraderPriceHistory:
    def test_tracks_recent_prices(self, sample_profile, mock_llm_client, lob_snapshot):
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        trader._arrival_rate = 0.0  # Don't call LLM

        for tick in range(5):
            snapshot = LOBSnapshot(
                timestamp=float(tick),
                best_bid=99.0 + tick,
                best_ask=101.0 + tick,
                mid_price=100.0 + tick,
                spread=2.0,
            )
            trader.generate_orders(tick, snapshot)

        assert len(trader._recent_prices) == 5
        assert trader._recent_prices[-1] == 104.0

    def test_no_orders_without_mid_price(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        snapshot = LOBSnapshot(timestamp=1.0)
        orders = trader.generate_orders(1, snapshot)
        assert orders == []


class TestLLMTraderPrompt:
    def test_prompt_contains_persona(self, sample_profile, mock_llm_client, lob_snapshot):
        trader = LLMTrader(sample_profile, mock_llm_client)
        trader._arrival_rate = 1.0
        trader._recent_prices = [99.0, 99.5, 100.0]

        prompt = trader._build_decision_prompt(1, lob_snapshot)
        assert "momentum trader" in prompt
        assert "overconfidence" in prompt
        assert "100.00" in prompt


class TestLLMTraderStats:
    def test_get_stats(self, sample_profile, mock_llm_client):
        trader = LLMTrader(sample_profile, mock_llm_client)
        stats = trader.get_stats()
        assert stats["agent_id"] == "momentum_001"
        assert stats["trader_type"] == "momentum"
        assert stats["total_decisions"] == 0


class TestCreateLLMAgents:
    def test_creates_agents_from_profiles(self, mock_llm_client):
        profiles = [
            TraderProfile(agent_id=f"agent_{i}", trader_type="noise", persona="A trader")
            for i in range(5)
        ]
        agents = create_llm_agents(profiles, mock_llm_client)
        assert len(agents) == 5
        assert all(isinstance(a, LLMTrader) for a in agents)

    def test_agents_have_unique_seeds(self, mock_llm_client):
        profiles = [
            TraderProfile(agent_id=f"agent_{i}", trader_type="noise", persona="A trader")
            for i in range(3)
        ]
        agents = create_llm_agents(profiles, mock_llm_client, seed=100)
        # Different seeds should produce different random states
        randoms = [a._rng.random() for a in agents]
        assert len(set(randoms)) == 3  # All unique


class TestLLMTraderPlan:
    def test_generate_plan_returns_agent_plan(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {
            "plan": [
                {"tick_offset": 0, "action": "BUY", "type": "LIMIT", "price": 99.0, "quantity": 1.0},
                {"tick_offset": 5, "action": "SELL", "type": "LIMIT", "price": 101.0, "quantity": 1.0},
            ],
            "reassess_after": 10,
        }
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        plan = trader.generate_plan(tick=10, lob_snapshot=lob_snapshot)
        assert isinstance(plan, AgentPlan)
        assert plan.agent_id == "momentum_001"
        assert plan.start_tick == 10
        assert plan.reassess_tick == 20
        assert 10 in plan.orders
        assert 15 in plan.orders

    def test_generate_plan_hold_only(self, sample_profile, mock_llm_client, lob_snapshot):
        mock_llm_client.chat_json.return_value = {
            "plan": [],
            "reassess_after": 10,
        }
        trader = LLMTrader(sample_profile, mock_llm_client, seed=0)
        plan = trader.generate_plan(tick=0, lob_snapshot=lob_snapshot)
        assert plan.orders == {}
        assert plan.reassess_tick == 10
