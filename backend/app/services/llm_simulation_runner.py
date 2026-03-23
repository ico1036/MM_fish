"""LLM-based market simulation runner with perpetual futures support."""

import logging

import numpy as np

from app.models.market import Order, OrderType, Side
from app.models.simulation import TickRecord
from app.models.perpetual import Position
from app.services.batch_auction import BatchAuction
from app.services.funding_engine import FundingEngine
from app.services.liquidation_engine import LiquidationEngine
from app.services.lob_engine import LOBEngine
from app.services.market_agents import MarketAgent
from app.services.mm_agent import HelixMMAgent
from app.services.llm_agents import LLMTrader
from app.services.plan_executor import PlanExecutor
from app.services.scenario_engine import ScenarioEngine

logger = logging.getLogger(__name__)


class LLMSimulationRunner:
    """
    Simulation runner supporting A-S (rule-based), LLM agent, and perpetual modes.

    When scenario is provided, uses batch auction matching, multi-tick plans,
    funding rate, and liquidation engines.
    When scenario is None, falls back to legacy sequential matching for A-S mode.
    """

    def __init__(
        self,
        lob: LOBEngine,
        mm_agent: HelixMMAgent,
        agents: list[MarketAgent],
        max_ticks: int = 1000,
        initial_mid_price: float = 100.0,
        seed: int = 42,
        mode: str = "llm",
        scenario: ScenarioEngine | None = None,
    ) -> None:
        self.lob = lob
        self.mm_agent = mm_agent
        self.agents = agents
        self.max_ticks = max_ticks
        self.initial_mid_price = initial_mid_price
        self.mode = mode
        self._rng = np.random.default_rng(seed)

        self.mm_agent.set_max_ticks(max_ticks)
        self._records: list[TickRecord] = []
        self._total_trades: int = 0
        self._agent_actions: list[dict] = []

        # Perpetual components (optional)
        self.scenario = scenario
        self._plan_executor = PlanExecutor() if scenario else None
        self._funding_engine = FundingEngine() if scenario else None
        self._liquidation_engine = LiquidationEngine() if scenario else None
        self._positions: dict[str, Position] = {}
        self._last_funding_rate: float | None = None

    def _seed_orderbook(self) -> None:
        """Place initial liquidity: 10 levels each side of mid with substantial depth."""
        mid = self.initial_mid_price
        level_step = mid * 0.0002
        for i in range(1, 11):
            bid_price = mid - i * level_step
            ask_price = mid + i * level_step
            qty = 20.0 / i
            self.lob.add_order(
                Order(
                    timestamp=0.0,
                    side=Side.BID,
                    price=bid_price,
                    quantity=qty,
                    order_type=OrderType.LIMIT,
                    agent_id="seed",
                )
            )
            self.lob.add_order(
                Order(
                    timestamp=0.0,
                    side=Side.ASK,
                    price=ask_price,
                    quantity=qty,
                    order_type=OrderType.LIMIT,
                    agent_id="seed",
                )
            )

    def run_tick(self, tick: int) -> TickRecord:
        """Execute one tick of the simulation."""
        if self.scenario:
            return self._run_tick_perpetual(tick)
        else:
            return self._run_tick_legacy(tick)

    def _run_tick_perpetual(self, tick: int) -> TickRecord:
        """Run a single tick in perpetual mode with batch auction, funding, liquidation."""
        tick_trades = 0
        snapshot = self.lob.get_depth()
        index_price = self.scenario.get_index_price(tick) if self.scenario else None

        # 1. Funding rate check
        funding_rate_val = None
        if self._funding_engine and self._funding_engine.is_funding_tick(tick):
            mid = snapshot.mid_price or self.initial_mid_price
            positions = list(self._positions.values())
            if positions and index_price:
                fr = self._funding_engine.compute_funding_rate(tick, positions, mid, index_price)
                funding_rate_val = fr.rate if fr.long_pays_short else -fr.rate
                self._last_funding_rate = funding_rate_val
                self.mm_agent.set_funding_rate(funding_rate_val)
            else:
                # Still record a zero funding rate on funding ticks
                funding_rate_val = 0.0

        # 2. Liquidation checks
        num_liquidations = 0
        if self._liquidation_engine and self._positions:
            mid = snapshot.mid_price or self.initial_mid_price
            events = self._liquidation_engine.check_all(tick, list(self._positions.values()), mid)
            num_liquidations = len(events)
            for event in events:
                self._positions.pop(event.agent_id, None)

        # 3. Cancel MM's previous quotes
        for cancel_id in self.mm_agent.get_cancel_ids():
            self.lob.cancel_order(cancel_id)

        # 4. MM generates new quotes
        mm_orders = self.mm_agent.generate_orders(tick, snapshot)

        # 5. Collect agent orders via plan executor
        agent_orders: list[Order] = []
        if self._plan_executor:
            agent_orders.extend(self._plan_executor.get_orders_for_tick(tick))

            needs_reassess = self._plan_executor.agents_needing_reassessment(tick)
            for agent in self.agents:
                if isinstance(agent, LLMTrader):
                    if agent.agent_id in needs_reassess or agent.agent_id not in self._plan_executor._plans:
                        plan = agent.generate_plan(tick, snapshot)
                        self._plan_executor.register_plan(plan)
                        agent_orders.extend(plan.get_orders(tick))
                else:
                    agent_orders.extend(agent.generate_orders(tick, snapshot))

        # 6. Batch auction
        auction = BatchAuction()
        auction.submit_many(mm_orders)
        auction.submit_many(agent_orders)
        trades, resting = auction.execute()

        # 7. Process trades
        mm_order_ids = {o.order_id for o in mm_orders}
        for trade in trades:
            tick_trades += 1
            self._process_trade(trade, mm_order_ids)

        # 8. Add resting orders to LOB
        for order in resting:
            self.lob.add_order(order)

        # 9. Build tick record
        mid = self.lob.get_mid_price() or self.initial_mid_price
        spread = self.lob.get_spread() or 0.0
        self._total_trades += tick_trades

        mm_bid = None
        mm_ask = None
        for o in mm_orders:
            if o.side == Side.BID:
                mm_bid = o.price
            elif o.side == Side.ASK:
                mm_ask = o.price

        return TickRecord(
            tick=tick,
            mid_price=mid,
            spread=spread,
            mm_inventory=self.mm_agent.inventory,
            mm_pnl=self.mm_agent.pnl_at_price(mid),
            mm_bid=mm_bid,
            mm_ask=mm_ask,
            num_trades=tick_trades,
            funding_rate=funding_rate_val,
            num_liquidations=num_liquidations,
            index_price=index_price,
        )

    def _run_tick_legacy(self, tick: int) -> TickRecord:
        """Run a single tick in legacy A-S mode with sequential matching."""
        tick_trades = 0
        tick_actions = {"tick": tick, "agent_orders": {}}

        # 1. Get LOB snapshot
        snapshot = self.lob.get_depth()

        # 2. Cancel MM's previous quotes
        for cancel_id in self.mm_agent.get_cancel_ids():
            self.lob.cancel_order(cancel_id)

        # 3. MM generates new quotes
        mm_orders = self.mm_agent.generate_orders(tick, snapshot)
        mm_order_ids = {o.order_id for o in mm_orders}

        # 4. Other agents generate orders
        all_orders: list[Order] = []
        for agent in self.agents:
            agent_orders = agent.generate_orders(tick, snapshot)
            all_orders.extend(agent_orders)
            if agent_orders:
                tick_actions["agent_orders"][agent.agent_id] = len(agent_orders)

        # 5. Submit MM orders first (provide liquidity)
        for order in mm_orders:
            trades = self.lob.add_order(order)
            for trade in trades:
                tick_trades += 1
                self._process_trade(trade, mm_order_ids)

        # 6. Submit other agent orders
        for order in all_orders:
            trades = self.lob.add_order(order)
            for trade in trades:
                tick_trades += 1
                self._process_trade(trade, mm_order_ids)

        # 7. Build tick record
        mid = self.lob.get_mid_price() or self.initial_mid_price
        spread = self.lob.get_spread() or 0.0
        self._total_trades += tick_trades

        mm_bid = None
        mm_ask = None
        for o in mm_orders:
            if o.side == Side.BID:
                mm_bid = o.price
            elif o.side == Side.ASK:
                mm_ask = o.price

        self._agent_actions.append(tick_actions)

        return TickRecord(
            tick=tick,
            mid_price=mid,
            spread=spread,
            mm_inventory=self.mm_agent.inventory,
            mm_pnl=self.mm_agent.pnl_at_price(mid),
            mm_bid=mm_bid,
            mm_ask=mm_ask,
            num_trades=tick_trades,
        )

    def _process_trade(self, trade, mm_order_ids: set[str]) -> None:
        """Process a trade: update MM and LLM agent states."""
        if trade.buyer_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.BID, trade.price, trade.quantity)
        if trade.seller_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.ASK, trade.price, trade.quantity)

        for agent in self.agents:
            if isinstance(agent, LLMTrader):
                if trade.buyer_id == agent.agent_id:
                    agent.on_fill(Side.BID, trade.price, trade.quantity)
                if trade.seller_id == agent.agent_id:
                    agent.on_fill(Side.ASK, trade.price, trade.quantity)

    def run(self) -> list[TickRecord]:
        """Run full simulation."""
        self._seed_orderbook()
        self._records = []

        for tick in range(self.max_ticks):
            record = self.run_tick(tick)
            self._records.append(record)

            if tick % 100 == 0:
                logger.info(
                    f"Tick {tick}/{self.max_ticks}: mid={record.mid_price:.2f}, "
                    f"spread={record.spread:.4f}, trades={record.num_trades}"
                )

        return self._records

    def get_results_summary(self) -> dict:
        """Summary statistics for the simulation."""
        from app.services.metrics import generate_report

        report = generate_report(self._records)
        report["mode"] = self.mode
        report["total_agents"] = len(self.agents)

        llm_agents = [a for a in self.agents if isinstance(a, LLMTrader)]
        if llm_agents:
            report["llm_agent_stats"] = {
                "total_llm_agents": len(llm_agents),
                "total_llm_decisions": sum(a._total_decisions for a in llm_agents),
                "total_decisions": sum(a._total_decisions for a in llm_agents),
                "avg_inventory": float(np.mean([a.inventory for a in llm_agents])),
            }

        return report

    def get_price_series(self) -> list[float]:
        """Extract mid price time series from records."""
        return [r.mid_price for r in self._records]

    def get_spread_series(self) -> list[float]:
        """Extract spread time series from records."""
        return [r.spread for r in self._records]
