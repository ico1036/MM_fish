"""Multi-tick plan storage and execution."""

from dataclasses import dataclass, field
from app.models.market import Order


@dataclass
class AgentPlan:
    agent_id: str
    start_tick: int
    reassess_tick: int
    orders: dict[int, list[Order]] = field(default_factory=dict)
    cancel_condition: str | None = None

    def is_active(self, tick: int) -> bool:
        return self.start_tick <= tick < self.reassess_tick

    def get_orders(self, tick: int) -> list[Order]:
        if not self.is_active(tick):
            return []
        return self.orders.get(tick, [])


class PlanExecutor:
    def __init__(self) -> None:
        self._plans: dict[str, AgentPlan] = {}

    def register_plan(self, plan: AgentPlan) -> None:
        self._plans[plan.agent_id] = plan

    def get_orders_for_tick(self, tick: int) -> list[Order]:
        orders = []
        for plan in self._plans.values():
            orders.extend(plan.get_orders(tick))
        return orders

    def agents_needing_reassessment(self, tick: int) -> set[str]:
        needs = set()
        for agent_id, plan in self._plans.items():
            if tick >= plan.reassess_tick:
                needs.add(agent_id)
        return needs

    def remove_plan(self, agent_id: str) -> None:
        self._plans.pop(agent_id, None)
