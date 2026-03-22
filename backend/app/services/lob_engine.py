"""Limit Order Book Engine with FIFO price-time priority matching."""

from collections import defaultdict, deque
from dataclasses import dataclass

from app.models.market import LOBSnapshot, Order, OrderType, Side, Trade


@dataclass
class _MutableOrder:
    """Mutable copy of an Order for partial fill tracking."""

    order_id: str
    timestamp: float
    side: Side
    price: float
    remaining_qty: float
    order_type: OrderType
    agent_id: str


class LOBEngine:
    """FIFO price-time priority Limit Order Book."""

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size
        # price → deque of _MutableOrder
        self._bids: dict[float, deque[_MutableOrder]] = defaultdict(deque)
        self._asks: dict[float, deque[_MutableOrder]] = defaultdict(deque)
        # order_id → (side, price) for fast cancel lookup
        self._order_index: dict[str, tuple[Side, float]] = {}
        self._timestamp: float = 0.0

    def add_order(self, order: Order) -> list[Trade]:
        """Add an order to the book. Returns list of trades if any matching occurs."""
        self._timestamp = order.timestamp
        morder = _MutableOrder(
            order_id=order.order_id,
            timestamp=order.timestamp,
            side=order.side,
            price=order.price,
            remaining_qty=order.quantity,
            order_type=order.order_type,
            agent_id=order.agent_id,
        )

        trades: list[Trade] = []

        if order.side == Side.BID:
            trades = self._match_bid(morder)
        else:
            trades = self._match_ask(morder)

        # Place residual on book (only for limit orders)
        if morder.remaining_qty > 0 and morder.order_type == OrderType.LIMIT:
            book = self._bids if morder.side == Side.BID else self._asks
            book[morder.price].append(morder)
            self._order_index[morder.order_id] = (morder.side, morder.price)

        return trades

    def _match_bid(self, bid: _MutableOrder) -> list[Trade]:
        """Match incoming bid against asks (lowest ask first)."""
        trades: list[Trade] = []
        sorted_ask_prices = sorted(self._asks.keys())

        for ask_price in sorted_ask_prices:
            if bid.remaining_qty <= 0:
                break
            if bid.price < ask_price and bid.order_type == OrderType.LIMIT:
                break  # No more matchable asks

            queue = self._asks[ask_price]
            while queue and bid.remaining_qty > 0:
                ask_order = queue[0]
                fill_qty = min(bid.remaining_qty, ask_order.remaining_qty)
                trade = Trade(
                    timestamp=bid.timestamp,
                    price=ask_order.price,  # passive order price
                    quantity=fill_qty,
                    buyer_id=bid.agent_id,
                    seller_id=ask_order.agent_id,
                    aggressor_side=Side.BID,
                )
                trades.append(trade)

                bid.remaining_qty -= fill_qty
                ask_order.remaining_qty -= fill_qty

                if ask_order.remaining_qty <= 0:
                    queue.popleft()
                    self._order_index.pop(ask_order.order_id, None)

            if not queue:
                del self._asks[ask_price]

        return trades

    def _match_ask(self, ask: _MutableOrder) -> list[Trade]:
        """Match incoming ask against bids (highest bid first)."""
        trades: list[Trade] = []
        sorted_bid_prices = sorted(self._bids.keys(), reverse=True)

        for bid_price in sorted_bid_prices:
            if ask.remaining_qty <= 0:
                break
            if ask.price > bid_price and ask.order_type == OrderType.LIMIT:
                break  # No more matchable bids

            queue = self._bids[bid_price]
            while queue and ask.remaining_qty > 0:
                bid_order = queue[0]
                fill_qty = min(ask.remaining_qty, bid_order.remaining_qty)
                trade = Trade(
                    timestamp=ask.timestamp,
                    price=bid_order.price,  # passive order price
                    quantity=fill_qty,
                    buyer_id=bid_order.agent_id,
                    seller_id=ask.agent_id,
                    aggressor_side=Side.ASK,
                )
                trades.append(trade)

                ask.remaining_qty -= fill_qty
                bid_order.remaining_qty -= fill_qty

                if bid_order.remaining_qty <= 0:
                    queue.popleft()
                    self._order_index.pop(bid_order.order_id, None)

            if not queue:
                del self._bids[bid_price]

        return trades

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if found and cancelled."""
        if order_id not in self._order_index:
            return False

        side, price = self._order_index.pop(order_id)
        book = self._bids if side == Side.BID else self._asks
        queue = book.get(price)
        if queue is None:
            return False

        # Find and remove the order from the deque
        for i, order in enumerate(queue):
            if order.order_id == order_id:
                del queue[i]
                if not queue:
                    del book[price]
                return True

        return False

    def get_best_bid(self) -> float | None:
        """Highest bid price, or None if no bids."""
        if not self._bids:
            return None
        return max(self._bids.keys())

    def get_best_ask(self) -> float | None:
        """Lowest ask price, or None if no asks."""
        if not self._asks:
            return None
        return min(self._asks.keys())

    def get_mid_price(self) -> float | None:
        """(best_bid + best_ask) / 2, or None if either side is empty."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def get_spread(self) -> float | None:
        """best_ask - best_bid, or None if either side is empty."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def get_depth(self, levels: int = 5) -> LOBSnapshot:
        """Return LOB snapshot with top N levels of depth."""
        sorted_bids = sorted(self._bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self._asks.keys())[:levels]

        bid_depth = []
        for price in sorted_bids:
            total_qty = sum(o.remaining_qty for o in self._bids[price])
            bid_depth.append((price, total_qty))

        ask_depth = []
        for price in sorted_asks:
            total_qty = sum(o.remaining_qty for o in self._asks[price])
            ask_depth.append((price, total_qty))

        return LOBSnapshot(
            timestamp=self._timestamp,
            best_bid=self.get_best_bid(),
            best_ask=self.get_best_ask(),
            mid_price=self.get_mid_price(),
            spread=self.get_spread(),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

    def get_order_count(self) -> int:
        """Total number of resting orders in the book."""
        count = 0
        for queue in self._bids.values():
            count += len(queue)
        for queue in self._asks.values():
            count += len(queue)
        return count

    def clear(self) -> None:
        """Reset the order book."""
        self._bids.clear()
        self._asks.clear()
        self._order_index.clear()
