"""LLM-based market simulation runner with A-S comparison mode."""

import logging

import numpy as np

from app.models.market import Order, OrderType, Side
from app.models.simulation import TickRecord
from app.services.lob_engine import LOBEngine
from app.services.market_agents import MarketAgent
from app.services.mm_agent import HelixMMAgent
from app.services.llm_agents import LLMTrader

logger = logging.getLogger(__name__)


class LLMSimulationRunner:
    """
    Simulation runner supporting both A-S (rule-based) and LLM agent modes.

    Both modes use the same LOB engine and can run on the same price path
    for fair comparison.
    """

    def __init__(
        self,
        lob: LOBEngine,
        mm_agent: HelixMMAgent,
        agents: list[MarketAgent],
        max_ticks: int = 1000,
        initial_mid_price: float = 100.0,
        seed: int = 42,
        mode: str = "llm",  # "llm" or "as" (Avellaneda-Stoikov)
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
        self._agent_actions: list[dict] = []  # Per-tick action log

    def _seed_orderbook(self) -> None:
        """Place initial liquidity: 5 levels each side of mid."""
        mid = self.initial_mid_price
        level_step = mid * 0.001
        for i in range(1, 6):
            bid_price = mid - i * level_step
            ask_price = mid + i * level_step
            self.lob.add_order(
                Order(
                    timestamp=0.0,
                    side=Side.BID,
                    price=bid_price,
                    quantity=10.0,
                    order_type=OrderType.LIMIT,
                    agent_id="seed",
                )
            )
            self.lob.add_order(
                Order(
                    timestamp=0.0,
                    side=Side.ASK,
                    price=ask_price,
                    quantity=10.0,
                    order_type=OrderType.LIMIT,
                    agent_id="seed",
                )
            )

    def run_tick(self, tick: int) -> TickRecord:
        """Execute one tick of the simulation."""
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
        # MM fill processing
        if trade.buyer_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.BID, trade.price, trade.quantity)
        if trade.seller_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.ASK, trade.price, trade.quantity)

        # LLM agent fill processing
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

        # Add LLM agent stats
        llm_agents = [a for a in self.agents if isinstance(a, LLMTrader)]
        if llm_agents:
            report["llm_agent_stats"] = {
                "total_llm_agents": len(llm_agents),
                "total_llm_decisions": sum(a._total_decisions for a in llm_agents),
                "total_fallbacks": sum(a._fallback_count for a in llm_agents),
                "avg_inventory": float(np.mean([a.inventory for a in llm_agents])),
            }

        return report

    def get_price_series(self) -> list[float]:
        """Extract mid price time series from records."""
        return [r.mid_price for r in self._records]

    def get_spread_series(self) -> list[float]:
        """Extract spread time series from records."""
        return [r.spread for r in self._records]
