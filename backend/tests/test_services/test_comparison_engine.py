"""Tests for comparison engine."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.services.comparison_engine import (
    ComparisonResult,
    compute_scorecard,
    run_as_simulation,
    run_llm_simulation,
)
from app.services.stylized_facts import compute_all_stylized_facts, compute_log_returns


@pytest.fixture
def gbm_prices():
    """GBM price path for testing."""
    rng = np.random.default_rng(42)
    n = 1000
    returns = rng.normal(0, 0.005, n)
    return 100.0 * np.exp(np.cumsum(returns))


class TestComparisonResult:
    def test_to_dict(self):
        result = ComparisonResult()
        result.real_prices = np.array([100.0, 101.0])
        d = result.to_dict()
        assert "real_facts" in d
        assert "scorecard" in d
        assert "real_prices" in d


class TestRunASSimulation:
    def test_runs_and_produces_records(self, gbm_prices):
        runner = run_as_simulation(
            initial_mid_price=float(gbm_prices[0]),
            price_path=gbm_prices,
            max_ticks=200,
            seed=42,
        )
        assert len(runner._records) == 200
        assert runner.mode == "as"

    def test_results_summary(self, gbm_prices):
        runner = run_as_simulation(
            initial_mid_price=float(gbm_prices[0]),
            price_path=gbm_prices,
            max_ticks=100,
            seed=42,
        )
        summary = runner.get_results_summary()
        assert summary["total_ticks"] == 100
        assert summary["mode"] == "as"


class TestRunLLMSimulation:
    def test_runs_with_mock_llm(self):
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "action": "HOLD",
            "type": "MARKET",
            "price": None,
            "quantity": 0,
            "reason": "test",
        }

        runner = run_llm_simulation(
            initial_mid_price=100.0,
            max_ticks=50,
            num_agents=5,
            seed=42,
            llm_client=mock_client,
        )
        assert len(runner._records) == 50
        assert runner.mode == "llm"

    def test_llm_summary_has_stats(self):
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {"action": "HOLD"}

        runner = run_llm_simulation(
            initial_mid_price=100.0,
            max_ticks=50,
            num_agents=10,
            seed=42,
            llm_client=mock_client,
        )
        summary = runner.get_results_summary()
        assert "llm_agent_stats" in summary
        assert summary["llm_agent_stats"]["total_llm_agents"] == 10


class TestRunLLMSimulationWithScenario:
    def test_runs_with_scenario(self):
        from app.services.scenario_engine import Scenario, ScenarioEngine
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {"plan": [], "reassess_after": 10}

        scenario = Scenario.normal_listing(initial_price=100.0, num_ticks=50, seed=42)

        runner = run_llm_simulation(
            initial_mid_price=100.0,
            llm_client=mock_client,
            max_ticks=50,
            num_agents=5,
            seed=42,
            scenario=ScenarioEngine(scenario),
        )
        assert len(runner._records) == 50
        assert runner._records[0].index_price is not None


class TestComputeScorecard:
    def test_scorecard_has_required_metrics(self, gbm_prices):
        real_facts = compute_all_stylized_facts(gbm_prices)

        # Create slightly different prices for sim
        rng = np.random.default_rng(99)
        sim_prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 1000)))
        sim_facts = compute_all_stylized_facts(sim_prices)

        real_ret = compute_log_returns(gbm_prices)
        sim_ret = compute_log_returns(sim_prices)

        scorecard = compute_scorecard(real_facts, sim_facts, real_ret, sim_ret)

        assert "kurtosis_distance" in scorecard
        assert "skewness_distance" in scorecard
        assert "acf_abs_returns_rmse" in scorecard
        assert "hurst_distance" in scorecard
        assert "ks_statistic" in scorecard
        assert "wasserstein_distance" in scorecard
        assert "hill_distance" in scorecard
        assert "total_score" in scorecard

    def test_identical_data_has_zero_score(self, gbm_prices):
        facts = compute_all_stylized_facts(gbm_prices)
        returns = compute_log_returns(gbm_prices)

        scorecard = compute_scorecard(facts, facts, returns, returns)
        assert scorecard["kurtosis_distance"] == 0.0
        assert scorecard["ks_statistic"] == 0.0
        assert scorecard["total_score"] < 0.01

    def test_different_data_has_positive_score(self, gbm_prices):
        real_facts = compute_all_stylized_facts(gbm_prices)

        # Fat-tailed prices should score differently
        rng = np.random.default_rng(99)
        fat_prices = 100.0 * np.exp(np.cumsum(rng.standard_t(3, 1000) * 0.005))
        fat_facts = compute_all_stylized_facts(fat_prices)

        real_ret = compute_log_returns(gbm_prices)
        fat_ret = compute_log_returns(fat_prices)

        scorecard = compute_scorecard(real_facts, fat_facts, real_ret, fat_ret)
        assert scorecard["total_score"] > 0

    def test_total_score_is_weighted_sum(self, gbm_prices):
        real_facts = compute_all_stylized_facts(gbm_prices)
        rng = np.random.default_rng(99)
        sim_prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 1000)))
        sim_facts = compute_all_stylized_facts(sim_prices)
        real_ret = compute_log_returns(gbm_prices)
        sim_ret = compute_log_returns(sim_prices)

        scorecard = compute_scorecard(real_facts, sim_facts, real_ret, sim_ret)

        # Manually compute weighted total
        weights = {
            "kurtosis_distance": 0.15,
            "skewness_distance": 0.10,
            "acf_abs_returns_rmse": 0.20,
            "hurst_distance": 0.10,
            "ks_statistic": 0.15,
            "wasserstein_distance": 0.20,
            "hill_distance": 0.10,
        }
        expected = sum(scorecard[k] * weights[k] for k in weights)
        assert abs(scorecard["total_score"] - expected) < 1e-10
