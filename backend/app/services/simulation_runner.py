"""ABM Market Simulation Runner."""

import numpy as np

from app.models.market import Order, OrderType, Side
from app.models.simulation import TickRecord
from app.services.lob_engine import LOBEngine
from app.services.market_agents import MarketAgent
from app.services.mm_agent import HelixMMAgent


class MarketSimulationRunner:
    """
    Agent-Based Model market simulation.

    Runs tick-by-tick simulation with noise traders, fundamentalists,
    informed traders, and an A-S market maker.
    """

    def __init__(
        self,
        lob: LOBEngine,
        agents: list[MarketAgent],
        mm_agent: HelixMMAgent,
        max_ticks: int = 1000,
        initial_mid_price: float = 100.0,
        seed: int = 42,
    ) -> None:
        self.lob = lob
        self.agents = agents
        self.mm_agent = mm_agent
        self.max_ticks = max_ticks
        self.initial_mid_price = initial_mid_price
        self._rng = np.random.default_rng(seed)

        self.mm_agent.set_max_ticks(max_ticks)
        self._records: list[TickRecord] = []
        self._total_trades: int = 0

    def _seed_orderbook(self) -> None:
        """Place initial liquidity: 5 levels each side of mid."""
        mid = self.initial_mid_price
        # Use 0.1% of mid as level spacing (price-adaptive)
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
            all_orders.extend(agent.generate_orders(tick, snapshot))

        # 5. Submit MM orders first (they provide liquidity)
        for order in mm_orders:
            trades = self.lob.add_order(order)
            for trade in trades:
                tick_trades += 1
                self._process_mm_trade(trade, mm_order_ids)

        # 6. Submit other agent orders
        for order in all_orders:
            trades = self.lob.add_order(order)
            for trade in trades:
                tick_trades += 1
                self._process_mm_trade(trade, mm_order_ids)

        # 7. Build tick record
        mid = self.lob.get_mid_price() or self.initial_mid_price
        spread = self.lob.get_spread() or 0.0
        self._total_trades += tick_trades

        # Get MM quote prices
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
        )

    def _process_mm_trade(self, trade, mm_order_ids: set[str]) -> None:
        """Check if MM was part of a trade and update its state."""
        # Check if buyer side is MM (MM's bid got filled)
        if trade.buyer_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.BID, trade.price, trade.quantity)
        # Check if seller side is MM (MM's ask got filled)
        if trade.seller_id == self.mm_agent.agent_id:
            self.mm_agent.on_fill(Side.ASK, trade.price, trade.quantity)

    def run(self) -> list[TickRecord]:
        """Run full simulation."""
        self._seed_orderbook()
        self._records = []

        for tick in range(self.max_ticks):
            record = self.run_tick(tick)
            self._records.append(record)

        return self._records

    def get_results_summary(self) -> dict:
        """Summary statistics for the simulation."""
        from app.services.metrics import generate_report

        return generate_report(self._records)
