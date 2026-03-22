"""Tests for stylized facts statistical analysis."""

import numpy as np
import pytest

from app.services.stylized_facts import (
    compute_log_returns,
    compute_return_stats,
    compute_volatility_clustering,
    compute_autocorrelation,
    compute_fat_tails,
    compute_hurst_exponent,
    compare_distributions,
    compute_all_stylized_facts,
)


@pytest.fixture
def gbm_prices():
    """Generate GBM prices with known properties."""
    rng = np.random.default_rng(42)
    n = 5000
    returns = rng.normal(0, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def fat_tail_prices():
    """Generate prices with fat-tailed returns (t-distribution)."""
    rng = np.random.default_rng(42)
    n = 5000
    returns = rng.standard_t(df=3, size=n) * 0.01
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


class TestLogReturns:
    def test_correct_length(self, gbm_prices):
        returns = compute_log_returns(gbm_prices)
        assert len(returns) == len(gbm_prices) - 1

    def test_returns_are_finite(self, gbm_prices):
        returns = compute_log_returns(gbm_prices)
        assert np.all(np.isfinite(returns))


class TestReturnStats:
    def test_normal_returns_low_kurtosis(self, gbm_prices):
        result = compute_return_stats(gbm_prices)
        # GBM should have near-zero excess kurtosis
        assert abs(result["kurtosis"]) < 1.0

    def test_fat_tail_returns_high_kurtosis(self, fat_tail_prices):
        result = compute_return_stats(fat_tail_prices)
        # t(3) distribution should have high kurtosis
        assert result["kurtosis"] > 2.0

    def test_jarque_bera_rejects_fat_tails(self, fat_tail_prices):
        result = compute_return_stats(fat_tail_prices)
        # Should reject normality for fat-tailed data
        assert result["jarque_bera_pvalue"] < 0.05

    def test_short_series_returns_defaults(self):
        result = compute_return_stats(np.array([100.0, 101.0]))
        assert result["kurtosis"] == 0.0


class TestVolatilityClustering:
    def test_produces_acf_values(self, gbm_prices):
        result = compute_volatility_clustering(gbm_prices)
        assert len(result["acf_abs_returns"]) == 21  # lag 0 to 20

    def test_first_acf_is_one(self, gbm_prices):
        result = compute_volatility_clustering(gbm_prices)
        assert abs(result["acf_abs_returns"][0] - 1.0) < 1e-10

    def test_short_series_returns_empty(self):
        result = compute_volatility_clustering(np.array([100.0, 101.0, 102.0]))
        assert result["acf_abs_returns"] == []


class TestAutocorrelation:
    def test_produces_both_acfs(self, gbm_prices):
        result = compute_autocorrelation(gbm_prices)
        assert len(result["acf_returns"]) == 21
        assert len(result["acf_abs_returns"]) == 21

    def test_gbm_returns_near_zero_acf(self, gbm_prices):
        result = compute_autocorrelation(gbm_prices)
        # GBM return ACF at lag 1 should be near zero
        assert abs(result["acf_returns"][1]) < 0.1


class TestFatTails:
    def test_hill_estimator_positive(self, fat_tail_prices):
        result = compute_fat_tails(fat_tail_prices)
        assert result["hill_estimator"] > 0

    def test_qq_data_produced(self, gbm_prices):
        result = compute_fat_tails(gbm_prices)
        assert len(result["qq_theoretical"]) > 0
        assert len(result["qq_empirical"]) == len(result["qq_theoretical"])

    def test_short_series_returns_defaults(self):
        result = compute_fat_tails(np.array([100.0, 101.0]))
        assert result["hill_estimator"] == 0.0


class TestHurstExponent:
    def test_gbm_near_half(self, gbm_prices):
        h = compute_hurst_exponent(gbm_prices)
        # GBM should have H ≈ 0.5 (random walk)
        assert 0.3 < h < 0.7

    def test_bounded_zero_one(self, gbm_prices):
        h = compute_hurst_exponent(gbm_prices)
        assert 0.0 <= h <= 1.0

    def test_short_series_returns_half(self):
        h = compute_hurst_exponent(np.array([100.0, 101.0, 102.0]))
        assert h == 0.5


class TestCompareDistributions:
    def test_identical_distributions(self, gbm_prices):
        returns = compute_log_returns(gbm_prices)
        result = compare_distributions(returns, returns)
        assert result["ks_statistic"] == 0.0
        assert result["wasserstein_distance"] < 1e-10

    def test_different_distributions(self, gbm_prices, fat_tail_prices):
        r1 = compute_log_returns(gbm_prices)
        r2 = compute_log_returns(fat_tail_prices)
        result = compare_distributions(r1, r2)
        assert result["ks_statistic"] > 0
        assert result["wasserstein_distance"] > 0

    def test_ks_rejects_different(self, gbm_prices, fat_tail_prices):
        r1 = compute_log_returns(gbm_prices)
        r2 = compute_log_returns(fat_tail_prices)
        result = compare_distributions(r1, r2)
        assert result["ks_pvalue"] < 0.05

    def test_short_series_returns_defaults(self):
        result = compare_distributions(np.array([0.01]), np.array([0.01]))
        assert result["ks_statistic"] == 1.0


class TestComputeAll:
    def test_all_keys_present(self, gbm_prices):
        result = compute_all_stylized_facts(gbm_prices)
        assert "return_stats" in result
        assert "volatility_clustering" in result
        assert "autocorrelation" in result
        assert "fat_tails" in result
        assert "hurst_exponent" in result
