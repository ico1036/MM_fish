"""Market participant agents for ABM simulation."""

from abc import ABC, abstractmethod

import numpy as np

from app.models.market import LOBSnapshot, Order, OrderType, Side


class MarketAgent(ABC):
    """Base class for all market participant agents."""

    def __init__(self, agent_id: str, params: dict) -> None:
        self.agent_id = agent_id
        self.params = params

    @abstractmethod
    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        """Generate orders based on current LOB state."""


class NoiseTrader(MarketAgent):
    """
    Random order generator.

    Each tick: Poisson(arrival_rate) orders.
    50/50 bid/ask, market_order_pct chance of market order.
    Limit prices: mid ± U(0, max_spread).
    """

    def __init__(self, agent_id: str, params: dict) -> None:
        super().__init__(agent_id, params)
        self.arrival_rate: float = params.get("arrival_rate", 1.0)
        self.market_order_pct: float = params.get("market_order_pct", 0.3)
        self.max_spread: float = params.get("max_spread", 2.0)
        self.quantity: float = params.get("quantity", 1.0)
        self._rng = np.random.default_rng(params.get("seed", None))

    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        mid = lob_snapshot.mid_price
        if mid is None:
            return []

        n_orders = self._rng.poisson(self.arrival_rate)
        orders: list[Order] = []

        for _ in range(n_orders):
            side = Side.BID if self._rng.random() < 0.5 else Side.ASK
            is_market = self._rng.random() < self.market_order_pct

            if is_market:
                price = float("inf") if side == Side.BID else 0.0
                order_type = OrderType.MARKET
            else:
                offset = self._rng.uniform(0, self.max_spread)
                price = mid - offset if side == Side.BID else mid + offset
                order_type = OrderType.LIMIT

            orders.append(
                Order(
                    timestamp=float(tick),
                    side=side,
                    price=price,
                    quantity=self.quantity,
                    order_type=order_type,
                    agent_id=self.agent_id,
                )
            )

        return orders


class Fundamentalist(MarketAgent):
    """
    Mean-reverting agent that trades toward fundamental value.

    Buys when mid < fundamental - threshold, sells when mid > fundamental + threshold.
    """

    def __init__(self, agent_id: str, params: dict) -> None:
        super().__init__(agent_id, params)
        self.fundamental_value: float = params.get("fundamental_value", 100.0)
        self.threshold: float = params.get("threshold", 0.5)
        self.aggression: float = params.get("aggression", 0.5)
        self.quantity: float = params.get("quantity", 2.0)
        self.update_speed: float = params.get("update_speed", 0.01)
        self._rng = np.random.default_rng(params.get("seed", None))

    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        mid = lob_snapshot.mid_price
        if mid is None:
            return []

        # Random walk on fundamental value
        self.fundamental_value += self._rng.normal(0, self.update_speed)

        deviation = mid - self.fundamental_value
        if abs(deviation) < self.threshold:
            return []

        if deviation > 0:
            # Price above fundamental → sell
            side = Side.ASK
        else:
            # Price below fundamental → buy
            side = Side.BID

        is_market = self._rng.random() < self.aggression
        if is_market:
            price = 0.0 if side == Side.ASK else float("inf")
            order_type = OrderType.MARKET
        else:
            price = self.fundamental_value if side == Side.BID else self.fundamental_value
            order_type = OrderType.LIMIT

        return [
            Order(
                timestamp=float(tick),
                side=side,
                price=price,
                quantity=self.quantity,
                order_type=order_type,
                agent_id=self.agent_id,
            )
        ]


class InformedTrader(MarketAgent):
    """
    Trader with partial knowledge of future prices.

    Uses market orders aggressively, imposing adverse selection on MM.
    """

    def __init__(self, agent_id: str, params: dict) -> None:
        super().__init__(agent_id, params)
        self.look_ahead: int = params.get("look_ahead", 10)
        self.accuracy: float = params.get("accuracy", 0.7)
        self.arrival_rate: float = params.get("arrival_rate", 0.3)
        self.quantity: float = params.get("quantity", 3.0)
        self._rng = np.random.default_rng(params.get("seed", None))
        self._future_prices: list[float] = []

    def set_future_prices(self, prices: list[float]) -> None:
        """Set the future price sequence for the informed trader."""
        self._future_prices = prices

    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        mid = lob_snapshot.mid_price
        if mid is None:
            return []

        # Check participation probability
        if self._rng.random() > self.arrival_rate:
            return []

        # Look ahead
        future_tick = tick + self.look_ahead
        if future_tick >= len(self._future_prices):
            return []

        future_price = self._future_prices[future_tick]

        # Apply accuracy: with (1-accuracy) probability, flip the signal
        if self._rng.random() > self.accuracy:
            # Wrong signal
            future_price = 2 * mid - future_price

        if future_price > mid:
            side = Side.BID
            price = float("inf")
        elif future_price < mid:
            side = Side.ASK
            price = 0.0
        else:
            return []

        return [
            Order(
                timestamp=float(tick),
                side=side,
                price=price,
                quantity=self.quantity,
                order_type=OrderType.MARKET,
                agent_id=self.agent_id,
            )
        ]
