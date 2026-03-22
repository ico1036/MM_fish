"""LLM-based market agents using Claude Haiku for per-tick trading decisions."""

import logging

import numpy as np

from app.models.agent_profile import TraderProfile
from app.models.market import LOBSnapshot, Order, OrderType, Side
from app.services.llm_client import LLMClient, HAIKU_MODEL
from app.services.market_agents import MarketAgent

logger = logging.getLogger(__name__)

# Default Poisson arrival rates by trader type
ARRIVAL_RATES = {
    "informed": 0.2,
    "noise": 0.15,
    "momentum": 0.25,
    "mean_reversion": 0.2,
    "fundamental": 0.1,
    "hft": 0.6,
    "institutional": 0.05,
}

# Default quantity multipliers by trader type
QUANTITY_MULTIPLIERS = {
    "informed": 2.0,
    "noise": 0.5,
    "momentum": 1.0,
    "mean_reversion": 1.0,
    "fundamental": 1.5,
    "hft": 0.3,
    "institutional": 5.0,
}


class LLMTrader(MarketAgent):
    """
    Agent that uses Claude Haiku for per-tick trading decisions.

    Each tick (with Poisson arrival probability), the agent receives
    a market state context and returns a single trading decision as JSON.
    """

    def __init__(
        self,
        profile: TraderProfile,
        llm_client: LLMClient,
        base_quantity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id=profile.agent_id, params={})
        self.profile = profile
        self.llm = llm_client
        self.base_quantity = base_quantity
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.pnl: float = 0.0
        self._rng = np.random.default_rng(seed)
        self._arrival_rate = ARRIVAL_RATES.get(profile.trader_type, 0.15)
        self._qty_mult = QUANTITY_MULTIPLIERS.get(profile.trader_type, 1.0)
        self._recent_prices: list[float] = []
        self._max_recent = 20
        self._total_decisions = 0
        self._fallback_count = 0

    @property
    def participation_rate(self) -> float:
        return self._arrival_rate

    def generate_orders(self, tick: int, lob_snapshot: LOBSnapshot) -> list[Order]:
        """Generate orders using LLM decision. Uses Poisson arrival to limit calls."""
        mid = lob_snapshot.mid_price
        if mid is None:
            return []

        # Track price history
        self._recent_prices.append(mid)
        if len(self._recent_prices) > self._max_recent:
            self._recent_prices = self._recent_prices[-self._max_recent:]

        # Poisson arrival check
        if self._rng.random() > self._arrival_rate:
            return []

        # Build prompt and call LLM
        prompt = self._build_decision_prompt(tick, lob_snapshot)
        try:
            response = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            self._total_decisions += 1
            return self._parse_decision(response, tick, mid)
        except Exception as e:
            logger.debug(f"LLM call failed for {self.agent_id}: {e}")
            self._fallback_count += 1
            return self._fallback_decision(tick, mid)

    def _build_decision_prompt(self, tick: int, snapshot: LOBSnapshot) -> str:
        """Build the market context prompt for the LLM."""
        mid = snapshot.mid_price or 0
        spread = snapshot.spread or 0
        best_bid = snapshot.best_bid or mid
        best_ask = snapshot.best_ask or mid

        # Compute price changes
        pct_5 = 0.0
        pct_20 = 0.0
        if len(self._recent_prices) >= 5:
            old = self._recent_prices[-5]
            if old > 0:
                pct_5 = ((mid - old) / old) * 100
        if len(self._recent_prices) >= 20:
            old = self._recent_prices[-20]
            if old > 0:
                pct_20 = ((mid - old) / old) * 100

        # Bid/ask depth
        bid_qty = sum(q for _, q in snapshot.bid_depth[:3]) if snapshot.bid_depth else 0
        ask_qty = sum(q for _, q in snapshot.ask_depth[:3]) if snapshot.ask_depth else 0

        return f"""You are {self.profile.persona}

Your behavioral bias: {self.profile.behavioral_bias}
Your risk appetite: {self.profile.risk_appetite:.1f}/1.0

MARKET STATE:
- Mid price: {mid:.2f}, Spread: {spread:.4f}
- Best bid: {best_bid:.2f} (depth: {bid_qty:.1f}), Best ask: {best_ask:.2f} (depth: {ask_qty:.1f})
- Price change (5 ticks): {pct_5:+.3f}%, Price change (20 ticks): {pct_20:+.3f}%
- Your inventory: {self.inventory:.4f}, Your PnL: {self.pnl:.2f}

Decide ONE action. Respond with ONLY a JSON object:
{{"action": "BUY"|"SELL"|"HOLD", "type": "MARKET"|"LIMIT", "price": null, "quantity": {self.base_quantity * self._qty_mult:.4f}, "reason": "brief"}}

For LIMIT orders, set price. For MARKET orders, price is null."""

    def _parse_decision(self, response: dict, tick: int, mid: float) -> list[Order]:
        """Parse LLM JSON response into Order objects."""
        if not response:
            return []

        action = response.get("action", "HOLD").upper()
        if action == "HOLD":
            return []

        order_type_str = response.get("type", "MARKET").upper()
        quantity = float(response.get("quantity", self.base_quantity * self._qty_mult))
        price = response.get("price")

        if action == "BUY":
            side = Side.BID
            if order_type_str == "MARKET":
                price = float("inf")
                order_type = OrderType.MARKET
            else:
                price = float(price) if price else mid
                order_type = OrderType.LIMIT
        elif action == "SELL":
            side = Side.ASK
            if order_type_str == "MARKET":
                price = 0.0
                order_type = OrderType.MARKET
            else:
                price = float(price) if price else mid
                order_type = OrderType.LIMIT
        else:
            return []

        return [
            Order(
                timestamp=float(tick),
                side=side,
                price=price,
                quantity=quantity,
                order_type=order_type,
                agent_id=self.agent_id,
            )
        ]

    def _fallback_decision(self, tick: int, mid: float) -> list[Order]:
        """Rule-based fallback when LLM is unavailable."""
        # Simple noise: 50/50 buy/sell with small quantity
        if self._rng.random() < 0.5:
            return []  # Most of the time, just hold on fallback

        side = Side.BID if self._rng.random() < 0.5 else Side.ASK
        offset = self._rng.uniform(0, mid * 0.002)
        price = mid - offset if side == Side.BID else mid + offset

        return [
            Order(
                timestamp=float(tick),
                side=side,
                price=price,
                quantity=self.base_quantity * self._qty_mult * 0.5,
                order_type=OrderType.LIMIT,
                agent_id=self.agent_id,
            )
        ]

    def on_fill(self, side: Side, price: float, quantity: float) -> None:
        """Update state when an order is filled."""
        if side == Side.BID:
            self.inventory += quantity
            self.cash -= price * quantity
        else:
            self.inventory -= quantity
            self.cash += price * quantity

    def pnl_at_price(self, mid_price: float) -> float:
        """Mark-to-market PnL."""
        return self.cash + self.inventory * mid_price

    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "agent_id": self.agent_id,
            "trader_type": self.profile.trader_type,
            "total_decisions": self._total_decisions,
            "fallback_count": self._fallback_count,
            "inventory": self.inventory,
            "cash": self.cash,
        }


def create_llm_agents(
    profiles: list[TraderProfile],
    llm_client: LLMClient,
    base_quantity: float = 1.0,
    seed: int = 42,
) -> list[LLMTrader]:
    """
    Create LLMTrader instances from profiles.

    Args:
        profiles: List of trader profiles.
        llm_client: Shared LLM client.
        base_quantity: Base order quantity.
        seed: Random seed base.

    Returns:
        List of LLMTrader instances.
    """
    agents = []
    for i, profile in enumerate(profiles):
        agent = LLMTrader(
            profile=profile,
            llm_client=llm_client,
            base_quantity=base_quantity,
            seed=seed + i,
        )
        agents.append(agent)
    return agents
