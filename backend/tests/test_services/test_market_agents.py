"""Tests for market participant agents."""

from app.models.market import LOBSnapshot, OrderType, Side
from app.services.market_agents import Fundamentalist, InformedTrader, NoiseTrader


def _make_snapshot(mid_price: float = 100.0) -> LOBSnapshot:
    spread = 0.1
    return LOBSnapshot(
        timestamp=0.0,
        best_bid=mid_price - spread / 2,
        best_ask=mid_price + spread / 2,
        mid_price=mid_price,
        spread=spread,
    )


class TestNoiseTrader:
    def test_generates_orders(self):
        """NoiseTrader generates orders with deterministic seed."""
        trader = NoiseTrader("noise_1", {"arrival_rate": 2.0, "seed": 42})
        orders = trader.generate_orders(tick=1, lob_snapshot=_make_snapshot())
        assert len(orders) > 0
        for o in orders:
            assert o.agent_id == "noise_1"

    def test_respects_arrival_rate_zero(self):
        """arrival_rate=0 → no orders."""
        trader = NoiseTrader("noise_1", {"arrival_rate": 0.0, "seed": 42})
        orders = trader.generate_orders(tick=1, lob_snapshot=_make_snapshot())
        assert len(orders) == 0

    def test_high_arrival_rate_many_orders(self):
        """High arrival_rate → many orders on average."""
        trader = NoiseTrader("noise_1", {"arrival_rate": 10.0, "seed": 42})
        total = 0
        for t in range(100):
            total += len(trader.generate_orders(tick=t, lob_snapshot=_make_snapshot()))
        assert total > 500  # ~1000 expected

    def test_no_orders_without_mid_price(self):
        """No orders if LOB has no mid_price."""
        trader = NoiseTrader("noise_1", {"arrival_rate": 5.0, "seed": 42})
        empty_snap = LOBSnapshot(timestamp=0.0)
        orders = trader.generate_orders(tick=1, lob_snapshot=empty_snap)
        assert len(orders) == 0

    def test_generates_both_sides(self):
        """Over many ticks, generates both BID and ASK orders."""
        trader = NoiseTrader("noise_1", {"arrival_rate": 3.0, "seed": 42})
        sides = set()
        for t in range(50):
            for o in trader.generate_orders(tick=t, lob_snapshot=_make_snapshot()):
                sides.add(o.side)
        assert Side.BID in sides
        assert Side.ASK in sides


class TestFundamentalist:
    def test_buys_when_price_below_value(self):
        """mid_price < fundamental_value → BID order."""
        fund = Fundamentalist(
            "fund_1",
            {"fundamental_value": 102.0, "threshold": 0.5, "aggression": 1.0, "update_speed": 0.0, "seed": 42},
        )
        orders = fund.generate_orders(tick=1, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 1
        assert orders[0].side == Side.BID

    def test_sells_when_price_above_value(self):
        """mid_price > fundamental_value → ASK order."""
        fund = Fundamentalist(
            "fund_1",
            {"fundamental_value": 98.0, "threshold": 0.5, "aggression": 1.0, "update_speed": 0.0, "seed": 42},
        )
        orders = fund.generate_orders(tick=1, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 1
        assert orders[0].side == Side.ASK

    def test_no_order_within_threshold(self):
        """No order when deviation < threshold."""
        fund = Fundamentalist(
            "fund_1",
            {"fundamental_value": 100.2, "threshold": 0.5, "aggression": 0.5, "update_speed": 0.0, "seed": 42},
        )
        orders = fund.generate_orders(tick=1, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 0


class TestInformedTrader:
    def test_buys_before_price_rise(self):
        """Buys when future price is higher."""
        trader = InformedTrader(
            "informed_1", {"look_ahead": 5, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42}
        )
        # Future prices: rising after tick 5
        future_prices = [100.0] * 5 + [105.0] * 20
        trader.set_future_prices(future_prices)

        orders = trader.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 1
        assert orders[0].side == Side.BID

    def test_sells_before_price_drop(self):
        """Sells when future price is lower."""
        trader = InformedTrader(
            "informed_1", {"look_ahead": 5, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42}
        )
        future_prices = [100.0] * 5 + [95.0] * 20
        trader.set_future_prices(future_prices)

        orders = trader.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 1
        assert orders[0].side == Side.ASK

    def test_uses_market_orders(self):
        """InformedTrader always uses market orders."""
        trader = InformedTrader(
            "informed_1", {"look_ahead": 5, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42}
        )
        future_prices = [100.0] * 5 + [110.0] * 20
        trader.set_future_prices(future_prices)

        orders = trader.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 1
        assert orders[0].order_type == OrderType.MARKET

    def test_no_order_without_future_data(self):
        """No order if future_prices not set or tick beyond range."""
        trader = InformedTrader(
            "informed_1", {"look_ahead": 5, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42}
        )
        orders = trader.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        assert len(orders) == 0


class TestAllAgentsReturnValidOrders:
    def test_all_agents_return_valid_orders(self):
        """All agents produce valid Order objects."""
        snap = _make_snapshot()
        agents = [
            NoiseTrader("n1", {"arrival_rate": 2.0, "seed": 42}),
            Fundamentalist("f1", {"fundamental_value": 105.0, "threshold": 0.5, "aggression": 1.0, "update_speed": 0.0, "seed": 42}),
        ]
        informed = InformedTrader("i1", {"look_ahead": 2, "accuracy": 1.0, "arrival_rate": 1.0, "seed": 42})
        informed.set_future_prices([100.0, 100.0, 110.0] + [110.0] * 20)
        agents.append(informed)

        for agent in agents:
            orders = agent.generate_orders(tick=0, lob_snapshot=snap)
            for o in orders:
                assert isinstance(o.side, Side)
                assert o.quantity > 0
                assert isinstance(o.order_type, OrderType)
