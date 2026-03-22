"""Market data models for LOB simulation."""

from enum import Enum

import uuid

from pydantic import BaseModel, Field


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class Order(BaseModel):
    """A single order in the LOB."""

    model_config = {"frozen": True}

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float
    side: Side
    price: float
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    agent_id: str = ""


class Trade(BaseModel):
    """A single executed trade."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float
    price: float
    quantity: float
    buyer_id: str
    seller_id: str
    aggressor_side: Side


class LOBSnapshot(BaseModel):
    """Snapshot of the LOB at a point in time."""

    timestamp: float
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    bid_depth: list[tuple[float, float]] = []
    ask_depth: list[tuple[float, float]] = []
