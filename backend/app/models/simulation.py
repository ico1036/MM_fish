"""Market simulation state models."""

from enum import Enum
from datetime import datetime

import uuid

from pydantic import BaseModel, Field


class SimStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentConfig(BaseModel):
    """Configuration for a market participant agent."""

    agent_id: str
    agent_type: str  # "noise", "fundamentalist", "informed", "mm"
    params: dict = {}


class MarketSimState(BaseModel):
    """State of a market simulation."""

    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: SimStatus = SimStatus.CREATED
    agents: list[AgentConfig] = []
    current_tick: int = 0
    max_ticks: int = 1000
    initial_mid_price: float = 100.0
    tick_size: float = 0.01
    created_at: datetime = Field(default_factory=datetime.now)


class TickRecord(BaseModel):
    """Record of a single simulation tick."""

    tick: int
    mid_price: float
    spread: float
    mm_inventory: float
    mm_pnl: float
    mm_bid: float | None = None
    mm_ask: float | None = None
    num_trades: int = 0
    funding_rate: float | None = None
    num_liquidations: int = 0
    index_price: float | None = None
