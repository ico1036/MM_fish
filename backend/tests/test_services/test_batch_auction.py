# backend/tests/test_services/test_batch_auction.py
import pytest
from app.models.market import Order, OrderType, Side
from app.services.batch_auction import BatchAuction


class TestBatchAuctionBasic:
    def test_single_trade(self):
        auction = BatchAuction()
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=float("inf"), quantity=5.0, order_type=OrderType.MARKET, agent_id="buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.ASK, price=100.0, quantity=10.0, order_type=OrderType.LIMIT, agent_id="seller"))
        trades, resting = auction.execute()
        assert len(trades) == 1
        assert trades[0].quantity == 5.0
        assert trades[0].price == 100.0

    def test_no_match(self):
        auction = BatchAuction()
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=99.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.ASK, price=101.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="seller"))
        trades, resting = auction.execute()
        assert len(trades) == 0
        assert len(resting) == 2


class TestBatchAuctionProRata:
    def test_excess_demand_pro_rata(self):
        auction = BatchAuction()
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=float("inf"), quantity=10.0, order_type=OrderType.MARKET, agent_id="big_buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=float("inf"), quantity=5.0, order_type=OrderType.MARKET, agent_id="small_buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.ASK, price=100.0, quantity=10.0, order_type=OrderType.LIMIT, agent_id="seller"))
        trades, _ = auction.execute()
        buyer_fills = {}
        for t in trades:
            buyer_fills[t.buyer_id] = buyer_fills.get(t.buyer_id, 0) + t.quantity
        assert buyer_fills["big_buyer"] == pytest.approx(6.667, abs=0.01)
        assert buyer_fills["small_buyer"] == pytest.approx(3.333, abs=0.01)


class TestBatchAuctionMarketPriority:
    def test_market_orders_before_limit(self):
        auction = BatchAuction()
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="limit_buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.BID, price=float("inf"), quantity=5.0, order_type=OrderType.MARKET, agent_id="market_buyer"))
        auction.submit(Order(timestamp=1.0, side=Side.ASK, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="seller"))
        trades, resting = auction.execute()
        assert len(trades) == 1
        assert trades[0].buyer_id == "market_buyer"
        assert any(o.agent_id == "limit_buyer" for o in resting)
