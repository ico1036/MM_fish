"""Trader profile models for LLM-based market agents."""

from pydantic import BaseModel, Field


class TraderProfile(BaseModel):
    """Profile defining a unique market participant persona."""

    agent_id: str
    trader_type: str  # informed, noise, momentum, mean_reversion, fundamental, hft, institutional
    persona: str  # Detailed personality/strategy description
    risk_appetite: float = Field(ge=0.0, le=1.0, default=0.5)
    capital: float = Field(gt=0, default=10000.0)
    information_edge: str = ""  # What unique info this trader has
    time_horizon: str = "intraday"  # scalper, intraday, swing
    behavioral_bias: str = ""  # overconfidence, loss_aversion, herd, contrarian

    @property
    def short_description(self) -> str:
        """One-line summary for logging."""
        return f"{self.agent_id}:{self.trader_type}({self.time_horizon}, risk={self.risk_appetite:.1f})"


# Default agent composition percentages
AGENT_COMPOSITION = {
    "informed": 0.08,
    "noise": 0.30,
    "momentum": 0.15,
    "mean_reversion": 0.15,
    "fundamental": 0.12,
    "hft": 0.15,
    "institutional": 0.05,
}

# Archetype persona templates (used as defaults when LLM generation is unavailable)
ARCHETYPE_PERSONAS = {
    "informed": (
        "You are an informed trader with access to proprietary research and order flow data. "
        "You trade aggressively when you believe the market is mispriced based on your edge. "
        "You prefer market orders for speed and are willing to pay the spread."
    ),
    "noise": (
        "You are a retail trader who trades for liquidity needs, portfolio rebalancing, or entertainment. "
        "You have no edge and make decisions based on gut feeling, news headlines, or random timing. "
        "You trade small sizes and don't have strong convictions."
    ),
    "momentum": (
        "You are a momentum trader who follows trends. When price is rising, you buy more. "
        "When price is falling, you sell. You use recent price changes as your primary signal. "
        "You tend to chase breakouts and cut losses quickly."
    ),
    "mean_reversion": (
        "You are a mean-reversion trader who fades extreme moves. When price moves too far too fast, "
        "you take the opposite side expecting a pullback. You use deviation from recent average as your signal. "
        "You are patient and willing to hold positions through short-term pain."
    ),
    "fundamental": (
        "You are a fundamental trader who anchors to a fair value estimate. "
        "You buy when the market price is below your fair value and sell when above. "
        "You update your fair value slowly based on macroeconomic data and on-chain metrics."
    ),
    "hft": (
        "You are a high-frequency scalper focused on capturing tiny spreads. "
        "You trade very frequently with small size, always using limit orders. "
        "You avoid holding positions for long and quickly flatten inventory."
    ),
    "institutional": (
        "You are an institutional block trader executing large orders for a fund. "
        "You split orders over time to minimize market impact. "
        "You are patient and use limit orders, often waiting for liquidity."
    ),
}
