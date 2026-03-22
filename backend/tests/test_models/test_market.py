"""Tests for market data models."""

from app.models.market import Order, Trade, LOBSnapshot, Side, OrderType


class TestOrder:
    def test_order_creation_with_defaults(self):
        order = Order(timestamp=1.0, side=Side.BID, price=100.0, quantity=1.0)
        assert order.order_id  # auto-generated
        assert order.timestamp == 1.0
        assert order.side == Side.BID
        assert order.price == 100.0
        assert order.quantity == 1.0
        assert order.order_type == OrderType.LIMIT
        assert order.agent_id == ""

    def test_order_is_immutable(self):
        order = Order(timestamp=1.0, side=Side.BID, price=100.0, quantity=1.0)
        try:
            order.price = 200.0
            assert False, "Should have raised"
        except Exception:
            pass

    def test_market_order_creation(self):
        order = Order(
            timestamp=1.0,
            side=Side.ASK,
            price=0.0,
            quantity=5.0,
            order_type=OrderType.MARKET,
            agent_id="trader_1",
        )
        assert order.order_type == OrderType.MARKET
        assert order.agent_id == "trader_1"

    def test_unique_order_ids(self):
        o1 = Order(timestamp=1.0, side=Side.BID, price=100.0, quantity=1.0)
        o2 = Order(timestamp=1.0, side=Side.BID, price=100.0, quantity=1.0)
        assert o1.order_id != o2.order_id


class TestTrade:
    def test_trade_creation(self):
        trade = Trade(
            timestamp=1.0,
            price=100.0,
            quantity=2.0,
            buyer_id="buyer",
            seller_id="seller",
            aggressor_side=Side.BID,
        )
        assert trade.trade_id
        assert trade.price == 100.0
        assert trade.quantity == 2.0
        assert trade.buyer_id == "buyer"
        assert trade.seller_id == "seller"
        assert trade.aggressor_side == Side.BID


class TestLOBSnapshot:
    def test_empty_snapshot(self):
        snap = LOBSnapshot(timestamp=0.0)
        assert snap.best_bid is None
        assert snap.best_ask is None
        assert snap.mid_price is None
        assert snap.spread is None
        assert snap.bid_depth == []
        assert snap.ask_depth == []

    def test_snapshot_with_data(self):
        snap = LOBSnapshot(
            timestamp=1.0,
            best_bid=99.0,
            best_ask=101.0,
            mid_price=100.0,
            spread=2.0,
            bid_depth=[(99.0, 10.0), (98.0, 20.0)],
            ask_depth=[(101.0, 10.0), (102.0, 20.0)],
        )
        assert snap.best_bid == 99.0
        assert snap.spread == 2.0
        assert len(snap.bid_depth) == 2
