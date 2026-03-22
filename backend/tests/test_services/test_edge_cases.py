"""Edge case tests to reach 100% coverage."""

from app.models.market import LOBSnapshot, Order, OrderType, Side
from app.services.lob_engine import LOBEngine
from app.services.market_agents import Fundamentalist, InformedTrader
from app.services.mm_agent import HelixMMAgent
from app.services.simulation_runner import MarketSimulationRunner


class TestLOBEngineEdgeCases:
    def test_cancel_order_queue_gone(self):
        """Cancel when index exists but queue was somehow emptied."""
        lob = LOBEngine()
        order = Order(timestamp=1.0, side=Side.BID, price=99.0, quantity=5.0, agent_id="a")
        lob.add_order(order)
        # Manually manipulate: remove queue but leave index
        lob._bids.pop(99.0, None)
        result = lob.cancel_order(order.order_id)
        assert result is False

    def test_cancel_order_id_mismatch_in_queue(self):
        """Cancel when order_id in index but not found in queue (defensive)."""
        lob = LOBEngine()
        o1 = Order(timestamp=1.0, side=Side.BID, price=99.0, quantity=5.0, agent_id="a")
        lob.add_order(o1)
        # Replace the queue contents (simulating corruption)
        from collections import deque
        from app.services.lob_engine import _MutableOrder
        lob._bids[99.0] = deque([_MutableOrder(
            order_id="different", timestamp=1.0, side=Side.BID,
            price=99.0, remaining_qty=5.0, order_type=OrderType.LIMIT, agent_id="b",
        )])
        result = lob.cancel_order(o1.order_id)
        assert result is False


class TestMMAgentEdgeCases:
    def test_zero_max_ticks(self):
        """max_ticks=0 → remaining_time = 0."""
        mm = HelixMMAgent("mm", {"gamma": 0.1, "k": 1.5, "sigma": 0.3})
        mm.set_max_ticks(0)
        assert mm.remaining_time == 0.0

    def test_mark_to_market_property(self):
        """mark_to_market_pnl returns cash."""
        mm = HelixMMAgent("mm", {"gamma": 0.1, "k": 1.5, "sigma": 0.3})
        mm.cash = 42.0
        assert mm.mark_to_market_pnl == 42.0


class TestFundamentalistEdgeCases:
    def test_no_mid_price(self):
        """Fundamentalist returns empty if no mid_price."""
        fund = Fundamentalist("f1", {"fundamental_value": 100.0, "threshold": 0.5, "update_speed": 0.0, "seed": 42})
        orders = fund.generate_orders(0, LOBSnapshot(timestamp=0.0))
        assert orders == []


class TestInformedTraderEdgeCases:
    def test_no_mid_price(self):
        """InformedTrader returns empty if no mid_price."""
        trader = InformedTrader("i1", {"look_ahead": 5, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42})
        trader.set_future_prices([100.0] * 20)
        orders = trader.generate_orders(0, LOBSnapshot(timestamp=0.0))
        assert orders == []

    def test_future_price_equals_mid(self):
        """No order when future_price == mid_price."""
        trader = InformedTrader("i1", {"look_ahead": 1, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42})
        trader.set_future_prices([100.0, 100.0, 100.0])
        snap = LOBSnapshot(timestamp=0.0, best_bid=99.9, best_ask=100.1, mid_price=100.0, spread=0.2)
        orders = trader.generate_orders(0, snap)
        assert orders == []


class TestSimulationRunnerEdgeCases:
    def test_mm_order_fills_immediately_against_resting(self):
        """When MM has large short inventory, bid shifts above best_ask → immediate fill."""
        lob = LOBEngine(tick_size=0.01)
        # Tight book: bid=99.9, ask=100.1 → mid=100.0
        lob.add_order(Order(
            timestamp=0.0, side=Side.BID, price=99.9, quantity=10.0,
            order_type=OrderType.LIMIT, agent_id="resting_buyer",
        ))
        lob.add_order(Order(
            timestamp=0.0, side=Side.ASK, price=100.1, quantity=1.0,
            order_type=OrderType.LIMIT, agent_id="resting_seller",
        ))
        # mid = 100.0
        # With gamma=1.0, sigma=0.3, inventory=-10:
        #   r = 100 - (-10)*1.0*0.09*1.0 = 100 + 0.9 = 100.9
        #   delta = 1.0*0.09 + 2*ln(1+1/1.5) = 0.09 + 0.811 = 0.901
        #   bid = 100.9 - 0.45 = 100.45 → crosses best_ask at 100.1!
        mm = HelixMMAgent("mm_helix", {
            "gamma": 1.0, "k": 1.5, "sigma": 0.3,
            "quantity": 1.0, "max_inventory": 20, "price_precision": 2,
        })
        mm.set_max_ticks(100)
        mm.inventory = -10.0  # Large short → bid shifts way up

        runner = MarketSimulationRunner(
            lob=lob, agents=[], mm_agent=mm,
            max_ticks=1, initial_mid_price=100.0,
        )
        record = runner.run_tick(0)

        # MM bid at ~100.45 crosses resting ask at 100.1 → trade!
        assert record.num_trades >= 1
        # MM bought (inventory increased from -10)
        assert mm.inventory > -10.0
