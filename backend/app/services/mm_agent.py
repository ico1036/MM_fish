"""Avellaneda-Stoikov based Market Maker agent."""

from app.models.market import LOBSnapshot, Order, OrderType, Side
from app.services.market_agents import MarketAgent
from app.utils.math_utils import compute_quotes


class HelixMMAgent(MarketAgent):
    """
    Avellaneda-Stoikov market maker.

    Each tick:
    1. Compute reservation price and optimal spread via A-S model
    2. Cancel previous quotes
    3. Submit new bid/ask (inventory-aware)
    4. Track fills → update inventory/cash
    """

    def __init__(self, agent_id: str, params: dict) -> None:
        super().__init__(agent_id, params)
        self.gamma: float = params.get("gamma", 0.1)
        self.k: float = params.get("k", 1.5)
        self.sigma: float = params.get("sigma", 0.3)
        self.T: float = params.get("T", 1.0)
        self.quantity: float = params.get("quantity", 1.0)
        self.max_inventory: int = params.get("max_inventory", 10)
        self.price_precision: int = params.get("price_precision", 4)

        # State
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.active_bid_id: str | None = None
        self.active_ask_id: str | None = None

        # Tracking
        self._max_ticks: int = 1000  # set by runner
        self._current_tick: int = 0

    def set_max_ticks(self, max_ticks: int) -> None:
        """Set total simulation ticks for T calculation."""
        self._max_ticks = max_ticks

    @property
    def remaining_time(self) -> float:
        """Remaining time fraction (1.0 → 0.0)."""
        if self._max_ticks <= 0:
            return 0.0
        return max(0.0, 1.0 - self._current_tick / self._max_ticks)

    @property
    def mark_to_market_pnl(self) -> float:
        """Cannot compute without mid_price — use pnl_at_price instead."""
        return self.cash

    def pnl_at_price(self, mid_price: float) -> float:
        """Mark-to-market PnL = cash + inventory * mid_price."""
        return self.cash + self.inventory * mid_price

    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        """Generate bid/ask quotes based on A-S model."""
        self._current_tick = tick
        mid = lob_snapshot.mid_price
        if mid is None:
            return []

        T_remaining = self.remaining_time
        bid_price, ask_price = compute_quotes(
            mid_price=mid,
            inventory=self.inventory,
            gamma=self.gamma,
            sigma=self.sigma,
            T=max(T_remaining, 0.001),  # avoid T=0 edge
            k=self.k,
        )

        orders: list[Order] = []

        # Inventory management: skip one side if at limit
        can_bid = self.inventory < self.max_inventory
        can_ask = self.inventory > -self.max_inventory

        if can_bid:
            bid = Order(
                timestamp=float(tick),
                side=Side.BID,
                price=round(bid_price, self.price_precision),
                quantity=self.quantity,
                order_type=OrderType.LIMIT,
                agent_id=self.agent_id,
            )
            orders.append(bid)
            self.active_bid_id = bid.order_id

        if can_ask:
            ask = Order(
                timestamp=float(tick),
                side=Side.ASK,
                price=round(ask_price, self.price_precision),
                quantity=self.quantity,
                order_type=OrderType.LIMIT,
                agent_id=self.agent_id,
            )
            orders.append(ask)
            self.active_ask_id = ask.order_id

        return orders

    def on_fill(self, side: Side, price: float, quantity: float) -> None:
        """Update state when an MM order is filled."""
        if side == Side.BID:
            # MM bought
            self.inventory += quantity
            self.cash -= price * quantity
        else:
            # MM sold
            self.inventory -= quantity
            self.cash += price * quantity

    def get_cancel_ids(self) -> list[str]:
        """Return IDs of active quotes to cancel before new quote cycle."""
        ids = []
        if self.active_bid_id is not None:
            ids.append(self.active_bid_id)
            self.active_bid_id = None
        if self.active_ask_id is not None:
            ids.append(self.active_ask_id)
            self.active_ask_id = None
        return ids
