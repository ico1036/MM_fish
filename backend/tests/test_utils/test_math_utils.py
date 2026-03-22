"""Tests for Avellaneda-Stoikov math utilities."""

from app.utils.math_utils import compute_quotes, optimal_spread, reservation_price


class TestReservationPrice:
    def test_zero_inventory_equals_mid_price(self):
        """Reservation price == mid_price when inventory is zero."""
        r = reservation_price(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.3, T=1.0)
        assert r == 100.0

    def test_long_inventory_lowers_price(self):
        """Long inventory (q > 0) → reservation price < mid_price."""
        r = reservation_price(mid_price=100.0, inventory=5.0, gamma=0.1, sigma=0.3, T=1.0)
        assert r < 100.0

    def test_short_inventory_raises_price(self):
        """Short inventory (q < 0) → reservation price > mid_price."""
        r = reservation_price(mid_price=100.0, inventory=-5.0, gamma=0.1, sigma=0.3, T=1.0)
        assert r > 100.0

    def test_higher_gamma_larger_adjustment(self):
        """Higher gamma → larger inventory adjustment."""
        r_low = reservation_price(mid_price=100.0, inventory=5.0, gamma=0.1, sigma=0.3, T=1.0)
        r_high = reservation_price(mid_price=100.0, inventory=5.0, gamma=0.5, sigma=0.3, T=1.0)
        # Both below mid, but higher gamma pushes further down
        assert r_high < r_low

    def test_zero_time_equals_mid_price(self):
        """At T=0, reservation price equals mid_price regardless of inventory."""
        r = reservation_price(mid_price=100.0, inventory=10.0, gamma=0.5, sigma=1.0, T=0.0)
        assert r == 100.0


class TestOptimalSpread:
    def test_spread_is_positive(self):
        """Spread is always positive."""
        s = optimal_spread(gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        assert s > 0

    def test_higher_gamma_wider_spread(self):
        """Higher risk aversion → wider spread (when sigma is large enough for linear term to dominate)."""
        s_low = optimal_spread(gamma=0.1, sigma=2.0, T=1.0, k=1.5)
        s_high = optimal_spread(gamma=0.5, sigma=2.0, T=1.0, k=1.5)
        assert s_high > s_low

    def test_higher_volatility_wider_spread(self):
        """Higher volatility → wider spread."""
        s_low = optimal_spread(gamma=0.1, sigma=0.1, T=1.0, k=1.5)
        s_high = optimal_spread(gamma=0.1, sigma=0.5, T=1.0, k=1.5)
        assert s_high > s_low

    def test_higher_k_narrower_spread(self):
        """Higher liquidity (k) → narrower spread."""
        s_low_k = optimal_spread(gamma=0.1, sigma=0.3, T=1.0, k=0.5)
        s_high_k = optimal_spread(gamma=0.1, sigma=0.3, T=1.0, k=5.0)
        assert s_high_k < s_low_k

    def test_zero_time_spread_still_positive(self):
        """Even at T=0, spread has the log component."""
        s = optimal_spread(gamma=0.1, sigma=0.3, T=0.0, k=1.5)
        assert s > 0


class TestComputeQuotes:
    def test_symmetric_at_zero_inventory(self):
        """At zero inventory, bid/ask are symmetric around mid_price."""
        bid, ask = compute_quotes(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        mid = (bid + ask) / 2.0
        assert abs(mid - 100.0) < 1e-10

    def test_bid_less_than_ask(self):
        """Bid is always less than ask."""
        bid, ask = compute_quotes(mid_price=100.0, inventory=3.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        assert bid < ask

    def test_long_inventory_skews_quotes_down(self):
        """Long inventory shifts both quotes lower (wants to sell)."""
        bid_zero, ask_zero = compute_quotes(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        bid_long, ask_long = compute_quotes(mid_price=100.0, inventory=5.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        assert bid_long < bid_zero
        assert ask_long < ask_zero

    def test_short_inventory_skews_quotes_up(self):
        """Short inventory shifts both quotes higher (wants to buy)."""
        bid_zero, ask_zero = compute_quotes(mid_price=100.0, inventory=0.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        bid_short, ask_short = compute_quotes(mid_price=100.0, inventory=-5.0, gamma=0.1, sigma=0.3, T=1.0, k=1.5)
        assert bid_short > bid_zero
        assert ask_short > ask_zero
