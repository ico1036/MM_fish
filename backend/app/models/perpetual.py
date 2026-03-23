"""Perpetual futures position and funding models."""

from pydantic import BaseModel, computed_field


class Position(BaseModel):
    """Leveraged perpetual position."""
    agent_id: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    leverage: float
    margin: float

    @computed_field
    @property
    def notional(self) -> float:
        return self.size * self.entry_price

    @computed_field
    @property
    def liquidation_price(self) -> float:
        if self.side == "long":
            return self.entry_price - self.margin / self.size
        else:
            return self.entry_price + self.margin / self.size

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == "long":
            return (mark_price - self.entry_price) * self.size
        else:
            return (self.entry_price - mark_price) * self.size

    def margin_ratio(self, mark_price: float) -> float:
        effective_margin = self.margin + self.unrealized_pnl(mark_price)
        return effective_margin / (self.size * mark_price)


class FundingRate(BaseModel):
    """Funding rate snapshot at a given tick."""
    tick: int
    rate: float
    long_pays_short: bool

    def payment(self, notional: float, is_long: bool) -> float:
        if self.long_pays_short:
            return -self.rate * notional if is_long else self.rate * notional
        else:
            return self.rate * notional if is_long else -self.rate * notional


class LiquidationEvent(BaseModel):
    """Record of a forced liquidation."""
    tick: int
    agent_id: str
    side: str
    size: float
    price: float
    loss: float
