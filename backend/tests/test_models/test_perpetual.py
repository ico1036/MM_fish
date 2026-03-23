import pytest
from app.models.perpetual import Position, FundingRate, LiquidationEvent


class TestPosition:
    def test_create_long(self):
        pos = Position(agent_id="a1", side="long", size=10.0, entry_price=5620.0, leverage=5.0, margin=11240.0)
        assert pos.notional == 56200.0
        assert pos.liquidation_price == pytest.approx(4496.0)

    def test_create_short(self):
        pos = Position(agent_id="a1", side="short", size=10.0, entry_price=5620.0, leverage=5.0, margin=11240.0)
        assert pos.liquidation_price == pytest.approx(6744.0)

    def test_unrealized_pnl_long(self):
        pos = Position(agent_id="a1", side="long", size=10.0, entry_price=5620.0, leverage=5.0, margin=11240.0)
        assert pos.unrealized_pnl(5630.0) == 100.0

    def test_unrealized_pnl_short(self):
        pos = Position(agent_id="a1", side="short", size=10.0, entry_price=5620.0, leverage=5.0, margin=11240.0)
        assert pos.unrealized_pnl(5610.0) == 100.0

    def test_margin_ratio(self):
        pos = Position(agent_id="a1", side="long", size=10.0, entry_price=5620.0, leverage=5.0, margin=11240.0)
        assert pos.margin_ratio(5620.0) == pytest.approx(0.2)


class TestFundingRate:
    def test_create(self):
        fr = FundingRate(tick=100, rate=0.0001, long_pays_short=True)
        assert fr.rate == 0.0001

    def test_payment_long(self):
        fr = FundingRate(tick=100, rate=0.0001, long_pays_short=True)
        assert fr.payment(notional=56200.0, is_long=True) == pytest.approx(-5.62)
        assert fr.payment(notional=56200.0, is_long=False) == pytest.approx(5.62)


class TestLiquidationEvent:
    def test_create(self):
        le = LiquidationEvent(tick=200, agent_id="a1", side="long", size=10.0, price=4500.0, loss=11200.0)
        assert le.agent_id == "a1"
