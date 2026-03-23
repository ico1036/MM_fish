"""Perpetual funding rate engine."""

from app.models.perpetual import FundingRate, Position


class FundingEngine:
    def __init__(self, base_rate: float = 0.0001, funding_interval: int = 100) -> None:
        self.base_rate = base_rate
        self.funding_interval = funding_interval
        self._history: list[FundingRate] = []

    def is_funding_tick(self, tick: int) -> bool:
        return tick > 0 and tick % self.funding_interval == 0

    def compute_funding_rate(
        self, tick: int, positions: list[Position], mark_price: float, index_price: float,
    ) -> FundingRate:
        premium = (mark_price - index_price) / index_price if index_price > 0 else 0.0
        long_oi = sum(p.size * p.entry_price for p in positions if p.side == "long")
        short_oi = sum(p.size * p.entry_price for p in positions if p.side == "short")
        total_oi = long_oi + short_oi
        imbalance = (long_oi - short_oi) / total_oi if total_oi > 0 else 0.0
        rate = self.base_rate * 0.1 + premium * 0.5 + imbalance * 0.001
        long_pays_short = rate >= 0
        fr = FundingRate(tick=tick, rate=abs(rate), long_pays_short=long_pays_short)
        self._history.append(fr)
        return fr

    def apply_funding(self, position: Position, funding_rate: FundingRate) -> float:
        payment = funding_rate.payment(position.notional, is_long=(position.side == "long"))
        return position.margin + payment
