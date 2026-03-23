# backend/tests/test_services/test_plan_executor.py
import pytest
from app.models.market import Order, OrderType, Side
from app.services.plan_executor import AgentPlan, PlanExecutor


class TestAgentPlan:
    def test_create(self):
        plan = AgentPlan(
            agent_id="a1",
            start_tick=10,
            reassess_tick=20,
            orders={
                10: [Order(timestamp=10.0, side=Side.BID, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="a1")],
                15: [Order(timestamp=15.0, side=Side.ASK, price=105.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="a1")],
            },
            cancel_condition="price < 95",
        )
        assert plan.is_active(tick=12)
        assert not plan.is_active(tick=25)

    def test_get_orders_for_tick(self):
        order = Order(timestamp=10.0, side=Side.BID, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="a1")
        plan = AgentPlan(agent_id="a1", start_tick=10, reassess_tick=20, orders={10: [order]})
        assert plan.get_orders(10) == [order]
        assert plan.get_orders(11) == []


class TestPlanExecutor:
    def test_register_and_get_orders(self):
        executor = PlanExecutor()
        order = Order(timestamp=10.0, side=Side.BID, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="a1")
        plan = AgentPlan(agent_id="a1", start_tick=10, reassess_tick=20, orders={10: [order]})
        executor.register_plan(plan)
        orders = executor.get_orders_for_tick(10)
        assert len(orders) == 1

    def test_agents_needing_reassessment(self):
        executor = PlanExecutor()
        plan = AgentPlan(agent_id="a1", start_tick=0, reassess_tick=10, orders={})
        executor.register_plan(plan)
        assert "a1" in executor.agents_needing_reassessment(10)
        assert "a1" not in executor.agents_needing_reassessment(5)

    def test_expired_plan_not_returned(self):
        executor = PlanExecutor()
        order = Order(timestamp=10.0, side=Side.BID, price=100.0, quantity=5.0, order_type=OrderType.LIMIT, agent_id="a1")
        plan = AgentPlan(agent_id="a1", start_tick=10, reassess_tick=15, orders={10: [order]})
        executor.register_plan(plan)
        assert executor.get_orders_for_tick(20) == []
