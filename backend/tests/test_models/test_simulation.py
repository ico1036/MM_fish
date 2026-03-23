"""Tests for simulation state models."""

from app.models.simulation import (
    AgentConfig,
    MarketSimState,
    SimStatus,
    TickRecord,
)


class TestAgentConfig:
    def test_agent_config_creation(self):
        config = AgentConfig(agent_id="mm_1", agent_type="mm", params={"gamma": 0.1})
        assert config.agent_id == "mm_1"
        assert config.agent_type == "mm"
        assert config.params["gamma"] == 0.1

    def test_agent_config_default_params(self):
        config = AgentConfig(agent_id="noise_1", agent_type="noise")
        assert config.params == {}


class TestMarketSimState:
    def test_default_state(self):
        state = MarketSimState()
        assert state.status == SimStatus.CREATED
        assert state.current_tick == 0
        assert state.max_ticks == 1000
        assert state.initial_mid_price == 100.0
        assert state.tick_size == 0.01
        assert state.agents == []

    def test_custom_state(self):
        state = MarketSimState(
            max_ticks=5000,
            initial_mid_price=50000.0,
            tick_size=0.1,
        )
        assert state.max_ticks == 5000
        assert state.initial_mid_price == 50000.0

    def test_unique_simulation_ids(self):
        s1 = MarketSimState()
        s2 = MarketSimState()
        assert s1.simulation_id != s2.simulation_id


class TestTickRecord:
    def test_tick_record_creation(self):
        record = TickRecord(
            tick=42,
            mid_price=100.5,
            spread=0.1,
            mm_inventory=3.0,
            mm_pnl=15.2,
            mm_bid=100.45,
            mm_ask=100.55,
            num_trades=5,
        )
        assert record.tick == 42
        assert record.mid_price == 100.5
        assert record.mm_inventory == 3.0
        assert record.num_trades == 5

    def test_tick_record_defaults(self):
        record = TickRecord(
            tick=0, mid_price=100.0, spread=0.0, mm_inventory=0.0, mm_pnl=0.0
        )
        assert record.mm_bid is None
        assert record.mm_ask is None
        assert record.num_trades == 0


class TestTickRecordPerpetual:
    def test_tick_record_with_perp_fields(self):
        record = TickRecord(
            tick=1, mid_price=100.0, spread=1.0,
            mm_inventory=0.0, mm_pnl=0.0,
            funding_rate=0.0001, num_liquidations=2, index_price=5620.0,
        )
        assert record.funding_rate == 0.0001
        assert record.num_liquidations == 2
        assert record.index_price == 5620.0

    def test_tick_record_perp_fields_default_none(self):
        record = TickRecord(tick=1, mid_price=100.0, spread=1.0, mm_inventory=0.0, mm_pnl=0.0)
        assert record.funding_rate is None
        assert record.num_liquidations == 0
        assert record.index_price is None
