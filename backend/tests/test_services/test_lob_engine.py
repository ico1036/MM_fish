"""Tests for LOB Engine — FIFO price-time priority matching."""

from app.models.market import Order, OrderType, Side
from app.services.lob_engine import LOBEngine


def _make_order(
    side: Side,
    price: float,
    quantity: float,
    timestamp: float = 1.0,
    order_type: OrderType = OrderType.LIMIT,
    agent_id: str = "",
) -> Order:
    return Order(
        timestamp=timestamp,
        side=side,
        price=price,
        quantity=quantity,
        order_type=order_type,
        agent_id=agent_id,
    )


class TestEmptyBook:
    def test_empty_lob_returns_none_for_prices(self):
        """Empty book has None for best_bid, best_ask, mid_price."""
        lob = LOBEngine()
        assert lob.get_best_bid() is None
        assert lob.get_best_ask() is None
        assert lob.get_mid_price() is None
        assert lob.get_spread() is None

    def test_empty_lob_order_count_zero(self):
        lob = LOBEngine()
        assert lob.get_order_count() == 0


class TestLimitOrders:
    def test_add_limit_bid_appears_in_book(self):
        """Limit bid order rests in the book."""
        lob = LOBEngine()
        order = _make_order(Side.BID, 99.0, 10.0)
        trades = lob.add_order(order)
        assert trades == []
        assert lob.get_best_bid() == 99.0
        assert lob.get_order_count() == 1

    def test_add_limit_ask_appears_in_book(self):
        """Limit ask order rests in the book."""
        lob = LOBEngine()
        order = _make_order(Side.ASK, 101.0, 5.0)
        trades = lob.add_order(order)
        assert trades == []
        assert lob.get_best_ask() == 101.0
        assert lob.get_order_count() == 1


class TestMatching:
    def test_crossing_orders_produce_trade(self):
        """Bid price >= ask price triggers a trade."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 5.0, timestamp=1.0, agent_id="seller"))
        trades = lob.add_order(_make_order(Side.BID, 100.0, 5.0, timestamp=2.0, agent_id="buyer"))
        assert len(trades) == 1
        assert trades[0].quantity == 5.0
        assert trades[0].buyer_id == "buyer"
        assert trades[0].seller_id == "seller"

    def test_trade_price_is_passive_order_price(self):
        """Trade executes at the passive (resting) order's price."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 99.0, 5.0, timestamp=1.0))
        trades = lob.add_order(_make_order(Side.BID, 101.0, 5.0, timestamp=2.0))
        assert len(trades) == 1
        assert trades[0].price == 99.0  # passive ask price

    def test_fifo_priority_at_same_price(self):
        """FIFO: earlier orders fill first at the same price."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 3.0, timestamp=1.0, agent_id="first"))
        lob.add_order(_make_order(Side.ASK, 100.0, 3.0, timestamp=2.0, agent_id="second"))
        trades = lob.add_order(_make_order(Side.BID, 100.0, 4.0, timestamp=3.0, agent_id="buyer"))
        assert len(trades) == 2
        assert trades[0].seller_id == "first"
        assert trades[0].quantity == 3.0
        assert trades[1].seller_id == "second"
        assert trades[1].quantity == 1.0

    def test_partial_fill(self):
        """Partial fill: residual stays on book."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 10.0, timestamp=1.0))
        trades = lob.add_order(_make_order(Side.BID, 100.0, 3.0, timestamp=2.0))
        assert len(trades) == 1
        assert trades[0].quantity == 3.0
        # 7 remaining on ask side
        assert lob.get_order_count() == 1
        assert lob.get_best_ask() == 100.0

    def test_multiple_trades_from_large_order(self):
        """Large order crosses multiple price levels producing multiple trades."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 5.0, timestamp=1.0, agent_id="s1"))
        lob.add_order(_make_order(Side.ASK, 101.0, 5.0, timestamp=2.0, agent_id="s2"))
        lob.add_order(_make_order(Side.ASK, 102.0, 5.0, timestamp=3.0, agent_id="s3"))

        trades = lob.add_order(_make_order(Side.BID, 102.0, 12.0, timestamp=4.0, agent_id="buyer"))
        assert len(trades) == 3
        assert trades[0].price == 100.0
        assert trades[0].quantity == 5.0
        assert trades[1].price == 101.0
        assert trades[1].quantity == 5.0
        assert trades[2].price == 102.0
        assert trades[2].quantity == 2.0
        # 3 remaining at 102.0
        assert lob.get_order_count() == 1


class TestMarketOrders:
    def test_market_order_fills_immediately(self):
        """Market order fills against best opposite side."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 5.0, timestamp=1.0))
        trades = lob.add_order(
            _make_order(Side.BID, float("inf"), 3.0, timestamp=2.0, order_type=OrderType.MARKET)
        )
        assert len(trades) == 1
        assert trades[0].quantity == 3.0
        assert trades[0].price == 100.0

    def test_market_order_no_residual(self):
        """Market order residual does NOT rest in the book (IOC)."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.ASK, 100.0, 3.0, timestamp=1.0))
        trades = lob.add_order(
            _make_order(Side.BID, float("inf"), 10.0, timestamp=2.0, order_type=OrderType.MARKET)
        )
        assert len(trades) == 1
        assert trades[0].quantity == 3.0
        assert lob.get_order_count() == 0  # no residual

    def test_sell_market_order_fills_against_bids(self):
        """Sell market order matches against highest bids."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.BID, 99.0, 5.0, timestamp=1.0))
        trades = lob.add_order(
            _make_order(Side.ASK, 0.0, 3.0, timestamp=2.0, order_type=OrderType.MARKET)
        )
        assert len(trades) == 1
        assert trades[0].price == 99.0


class TestCancel:
    def test_cancel_order_removes_from_book(self):
        """Cancelled order disappears from the book."""
        lob = LOBEngine()
        order = _make_order(Side.BID, 99.0, 10.0)
        lob.add_order(order)
        assert lob.get_order_count() == 1
        result = lob.cancel_order(order.order_id)
        assert result is True
        assert lob.get_order_count() == 0
        assert lob.get_best_bid() is None

    def test_cancel_nonexistent_returns_false(self):
        """Cancelling non-existent order returns False."""
        lob = LOBEngine()
        assert lob.cancel_order("nonexistent") is False


class TestDepthAndSpread:
    def test_get_depth_returns_correct_levels(self):
        """get_depth returns accurate price levels."""
        lob = LOBEngine()
        for i in range(5):
            lob.add_order(_make_order(Side.BID, 99.0 - i * 0.1, 10.0, timestamp=float(i)))
            lob.add_order(_make_order(Side.ASK, 101.0 + i * 0.1, 10.0, timestamp=float(i)))

        snap = lob.get_depth(levels=3)
        assert len(snap.bid_depth) == 3
        assert len(snap.ask_depth) == 3
        # Bids descending
        assert snap.bid_depth[0][0] == 99.0
        assert snap.bid_depth[1][0] == 98.9
        # Asks ascending
        assert snap.ask_depth[0][0] == 101.0
        assert snap.ask_depth[1][0] == 101.1

    def test_spread_calculation(self):
        """Spread = best_ask - best_bid."""
        lob = LOBEngine()
        lob.add_order(_make_order(Side.BID, 99.0, 10.0))
        lob.add_order(_make_order(Side.ASK, 101.0, 10.0))
        assert lob.get_spread() == 2.0
        assert lob.get_mid_price() == 100.0


class TestClear:
    def test_clear_resets_book(self):
        lob = LOBEngine()
        lob.add_order(_make_order(Side.BID, 99.0, 10.0))
        lob.add_order(_make_order(Side.ASK, 101.0, 10.0))
        lob.clear()
        assert lob.get_order_count() == 0
        assert lob.get_best_bid() is None
        assert lob.get_best_ask() is None
