"""Liquidation engine for perpetual positions."""

from app.models.perpetual import LiquidationEvent, Position


class LiquidationEngine:
    def __init__(self, maintenance_margin_ratio: float = 0.05) -> None:
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self._events: list[LiquidationEvent] = []

    def check_liquidation(self, tick: int, position: Position, mark_price: float) -> LiquidationEvent | None:
        ratio = position.margin_ratio(mark_price)
        if ratio < self.maintenance_margin_ratio:
            remaining_margin = max(0.0, position.margin + position.unrealized_pnl(mark_price))
            loss = position.margin - remaining_margin
            event = LiquidationEvent(
                tick=tick, agent_id=position.agent_id, side=position.side,
                size=position.size, price=mark_price, loss=loss,
            )
            self._events.append(event)
            return event
        return None

    def check_all(self, tick: int, positions: list[Position], mark_price: float) -> list[LiquidationEvent]:
        events = []
        for pos in positions:
            event = self.check_liquidation(tick, pos, mark_price)
            if event is not None:
                events.append(event)
        return events

    @property
    def history(self) -> list[LiquidationEvent]:
        return list(self._events)
