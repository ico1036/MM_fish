"""Tests for HelixMMAgent — Avellaneda-Stoikov market maker."""

from app.models.market import LOBSnapshot, Side
from app.services.mm_agent import HelixMMAgent


def _make_snapshot(mid: float = 100.0, spread: float = 0.1) -> LOBSnapshot:
    return LOBSnapshot(
        timestamp=0.0,
        best_bid=mid - spread / 2,
        best_ask=mid + spread / 2,
        mid_price=mid,
        spread=spread,
    )


def _make_mm(**overrides) -> HelixMMAgent:
    defaults = {
        "gamma": 0.1,
        "k": 1.5,
        "sigma": 0.3,
        "quantity": 1.0,
        "max_inventory": 10,
    }
    defaults.update(overrides)
    mm = HelixMMAgent("mm_helix", defaults)
    mm.set_max_ticks(1000)
    return mm


class TestQuoteGeneration:
    def test_mm_generates_two_sided_quotes(self):
        """At zero inventory, MM generates both bid and ask."""
        mm = _make_mm()
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot())
        assert len(orders) == 2
        sides = {o.side for o in orders}
        assert Side.BID in sides
        assert Side.ASK in sides

    def test_mm_quotes_straddle_mid_price(self):
        """bid < mid_price < ask."""
        mm = _make_mm()
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        bid = next(o for o in orders if o.side == Side.BID)
        ask = next(o for o in orders if o.side == Side.ASK)
        assert bid.price < 100.0
        assert ask.price > 100.0

    def test_mm_skews_quotes_with_long_inventory(self):
        """Long inventory → ask closer to mid (wants to sell)."""
        mm = _make_mm()
        mm.inventory = 5.0
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        bid = next(o for o in orders if o.side == Side.BID)
        ask = next(o for o in orders if o.side == Side.ASK)
        # Both shifted down due to long position
        mid = (bid.price + ask.price) / 2.0
        assert mid < 100.0

    def test_mm_skews_quotes_with_short_inventory(self):
        """Short inventory → bid closer to mid (wants to buy)."""
        mm = _make_mm()
        mm.inventory = -5.0
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot(100.0))
        bid = next(o for o in orders if o.side == Side.BID)
        ask = next(o for o in orders if o.side == Side.ASK)
        mid = (bid.price + ask.price) / 2.0
        assert mid > 100.0


class TestInventoryLimits:
    def test_only_asks_at_max_long_inventory(self):
        """At max_inventory, only ask is submitted."""
        mm = _make_mm(max_inventory=10)
        mm.inventory = 10.0
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot())
        assert len(orders) == 1
        assert orders[0].side == Side.ASK

    def test_only_bids_at_max_short_inventory(self):
        """At -max_inventory, only bid is submitted."""
        mm = _make_mm(max_inventory=10)
        mm.inventory = -10.0
        orders = mm.generate_orders(tick=0, lob_snapshot=_make_snapshot())
        assert len(orders) == 1
        assert orders[0].side == Side.BID


class TestCancelAndPnL:
    def test_cancels_previous_quotes(self):
        """get_cancel_ids returns previous quote IDs."""
        mm = _make_mm()
        mm.generate_orders(tick=0, lob_snapshot=_make_snapshot())
        assert mm.active_bid_id is not None
        assert mm.active_ask_id is not None

        cancel_ids = mm.get_cancel_ids()
        assert len(cancel_ids) == 2
        assert mm.active_bid_id is None
        assert mm.active_ask_id is None

    def test_pnl_tracks_correctly(self):
        """Cash and inventory update correctly on fills."""
        mm = _make_mm()
        # MM buys 1 at 99.0
        mm.on_fill(Side.BID, price=99.0, quantity=1.0)
        assert mm.inventory == 1.0
        assert mm.cash == -99.0

        # MM sells 1 at 101.0
        mm.on_fill(Side.ASK, price=101.0, quantity=1.0)
        assert mm.inventory == 0.0
        assert mm.cash == -99.0 + 101.0  # profit of 2.0

    def test_mark_to_market_pnl(self):
        """PnL at price includes unrealized inventory value."""
        mm = _make_mm()
        mm.on_fill(Side.BID, price=99.0, quantity=2.0)
        # cash = -198, inventory = 2, mid = 100
        pnl = mm.pnl_at_price(100.0)
        assert pnl == -198.0 + 2.0 * 100.0  # = 2.0


class TestRemainingTime:
    def test_remaining_time_decreases(self):
        """T decreases as ticks progress."""
        mm = _make_mm()
        mm.set_max_ticks(100)

        mm.generate_orders(tick=0, lob_snapshot=_make_snapshot())
        t0 = mm.remaining_time

        mm.generate_orders(tick=50, lob_snapshot=_make_snapshot())
        t50 = mm.remaining_time

        mm.generate_orders(tick=99, lob_snapshot=_make_snapshot())
        t99 = mm.remaining_time

        assert t0 > t50 > t99
        assert t0 == 1.0
        assert t99 < 0.02

    def test_no_orders_without_mid_price(self):
        """No quotes if LOB has no mid_price."""
        mm = _make_mm()
        empty = LOBSnapshot(timestamp=0.0)
        orders = mm.generate_orders(tick=0, lob_snapshot=empty)
        assert len(orders) == 0
