"""Batch auction matching engine — collects orders, matches in one pass."""

from app.models.market import Order, OrderType, Side, Trade


class BatchAuction:
    """
    Collects orders during a tick, then matches them in a single batch.
    Market orders have priority. Pro-rata allocation for excess demand/supply.
    """

    def __init__(self) -> None:
        self._bids: list[Order] = []
        self._asks: list[Order] = []

    def submit(self, order: Order) -> None:
        if order.side == Side.BID:
            self._bids.append(order)
        else:
            self._asks.append(order)

    def submit_many(self, orders: list[Order]) -> None:
        for order in orders:
            self.submit(order)

    def execute(self) -> tuple[list[Trade], list[Order]]:
        trades: list[Trade] = []
        market_bids = [o for o in self._bids if o.order_type == OrderType.MARKET]
        limit_bids = sorted([o for o in self._bids if o.order_type == OrderType.LIMIT], key=lambda o: -o.price)
        market_asks = [o for o in self._asks if o.order_type == OrderType.MARKET]
        limit_asks = sorted([o for o in self._asks if o.order_type == OrderType.LIMIT], key=lambda o: o.price)
        ask_pool = [[o, o.quantity] for o in limit_asks]
        bid_pool = [[o, o.quantity] for o in limit_bids]

        # Phase 1: Market buys vs limit asks
        trades.extend(self._match_market_vs_limit(market_bids, ask_pool, side="buy"))
        # Phase 2: Market sells vs limit bids
        trades.extend(self._match_market_vs_limit(market_asks, bid_pool, side="sell"))
        # Phase 3: Limit bids vs remaining limit asks
        trades.extend(self._match_limits(bid_pool, ask_pool))

        resting = []
        for order, remaining in bid_pool:
            if remaining > 0.001:
                resting.append(Order(timestamp=order.timestamp, side=order.side, price=order.price,
                    quantity=remaining, order_type=order.order_type, agent_id=order.agent_id))
        for order, remaining in ask_pool:
            if remaining > 0.001:
                resting.append(Order(timestamp=order.timestamp, side=order.side, price=order.price,
                    quantity=remaining, order_type=order.order_type, agent_id=order.agent_id))

        self._bids.clear()
        self._asks.clear()
        return trades, resting

    def _match_market_vs_limit(self, market_orders, limit_pool, side):
        trades = []
        total_market_qty = sum(o.quantity for o in market_orders)
        if total_market_qty == 0:
            return trades
        total_limit_qty = sum(qty for _, qty in limit_pool)
        if total_limit_qty == 0:
            return trades
        remaining_market = {o.agent_id: o.quantity for o in market_orders}
        for i in range(len(limit_pool)):
            limit_order, limit_remaining = limit_pool[i]
            if limit_remaining <= 0.001:
                continue
            total_demand = sum(remaining_market.values())
            if total_demand <= 0.001:
                break
            fillable = min(limit_remaining, total_demand)
            for mkt_order in market_orders:
                agent_remaining = remaining_market.get(mkt_order.agent_id, 0)
                if agent_remaining <= 0.001:
                    continue
                share = agent_remaining / total_demand
                fill_qty = min(share * fillable, agent_remaining)
                if fill_qty > 0.001:
                    buyer_id = mkt_order.agent_id if side == "buy" else limit_order.agent_id
                    seller_id = limit_order.agent_id if side == "buy" else mkt_order.agent_id
                    trades.append(Trade(timestamp=mkt_order.timestamp, price=limit_order.price,
                        quantity=round(fill_qty, 6), buyer_id=buyer_id, seller_id=seller_id,
                        aggressor_side=Side.BID if side == "buy" else Side.ASK))
                    remaining_market[mkt_order.agent_id] = agent_remaining - fill_qty
                    limit_pool[i] = [limit_order, limit_remaining - fill_qty]
                    limit_remaining -= fill_qty
        return trades

    def _match_limits(self, bid_pool, ask_pool):
        trades = []
        bi, ai = 0, 0
        while bi < len(bid_pool) and ai < len(ask_pool):
            bid_order, bid_remaining = bid_pool[bi]
            ask_order, ask_remaining = ask_pool[ai]
            if bid_remaining <= 0.001:
                bi += 1
                continue
            if ask_remaining <= 0.001:
                ai += 1
                continue
            if bid_order.price < ask_order.price:
                break
            fill_qty = min(bid_remaining, ask_remaining)
            trades.append(Trade(timestamp=bid_order.timestamp, price=ask_order.price,
                quantity=round(fill_qty, 6), buyer_id=bid_order.agent_id,
                seller_id=ask_order.agent_id, aggressor_side=Side.BID))
            bid_pool[bi] = [bid_order, bid_remaining - fill_qty]
            ask_pool[ai] = [ask_order, ask_remaining - fill_qty]
        return trades
