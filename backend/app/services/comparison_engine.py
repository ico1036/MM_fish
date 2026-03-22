"""Comparison engine: A-S baseline vs LLM agent simulation."""

import logging

import numpy as np

from app.models.market import LOBSnapshot
from app.services.binance_data import get_price_path
from app.services.lob_engine import LOBEngine
from app.services.llm_agents import LLMTrader, create_llm_agents
from app.services.llm_client import LLMClient
from app.services.llm_simulation_runner import LLMSimulationRunner
from app.services.market_agents import NoiseTrader, Fundamentalist, InformedTrader
from app.services.mm_agent import HelixMMAgent
from app.services.persona_generator import generate_profiles_deterministic
from app.services.stylized_facts import (
    compute_all_stylized_facts,
    compute_log_returns,
    compare_distributions,
)

logger = logging.getLogger(__name__)


class ComparisonResult:
    """Container for A-S vs LLM comparison results."""

    def __init__(self):
        self.real_prices: np.ndarray = np.array([])
        self.as_records: list = []
        self.llm_records: list = []
        self.real_facts: dict = {}
        self.as_facts: dict = {}
        self.llm_facts: dict = {}
        self.as_summary: dict = {}
        self.llm_summary: dict = {}
        self.scorecard: dict = {}

    def to_dict(self) -> dict:
        return {
            "real_facts": self.real_facts,
            "as_facts": self.as_facts,
            "llm_facts": self.llm_facts,
            "as_summary": self.as_summary,
            "llm_summary": self.llm_summary,
            "scorecard": self.scorecard,
            "real_prices": self.real_prices.tolist(),
            "as_prices": [r.mid_price for r in self.as_records],
            "llm_prices": [r.mid_price for r in self.llm_records],
        }


def run_as_simulation(
    initial_mid_price: float,
    price_path: np.ndarray,
    max_ticks: int = 1000,
    seed: int = 42,
    gamma: float = 0.1,
    sigma: float | None = None,
) -> LLMSimulationRunner:
    """
    Run A-S baseline simulation with rule-based agents.

    Args:
        initial_mid_price: Starting mid price.
        price_path: Real price path for informed trader.
        max_ticks: Number of ticks.
        seed: Random seed.
        gamma: MM risk aversion.
        sigma: Volatility estimate. Auto-computed if None.

    Returns:
        Completed LLMSimulationRunner in "as" mode.
    """
    if sigma is None:
        if len(price_path) > 1:
            returns = np.diff(np.log(price_path))
            sigma = float(np.std(returns))
        else:
            sigma = 0.01

    # Calibrate k to target ~20bps spread
    target_spread = initial_mid_price * 0.002
    k = max(0.1, 2.0 * gamma / (np.exp(gamma * target_spread / 2.0) - 1.0))

    lob = LOBEngine(tick_size=0.01)
    mm = HelixMMAgent(
        agent_id="mm_as",
        params={
            "gamma": gamma,
            "k": k,
            "sigma": sigma,
            "T": 1.0,
            "quantity": 1.0,
            "max_inventory": 10,
        },
    )

    agents = [
        NoiseTrader("noise_1", {"arrival_rate": 1.0, "market_order_pct": 0.3, "max_spread": initial_mid_price * 0.005, "quantity": 1.0, "seed": seed}),
        NoiseTrader("noise_2", {"arrival_rate": 0.8, "market_order_pct": 0.2, "max_spread": initial_mid_price * 0.003, "quantity": 0.5, "seed": seed + 1}),
        NoiseTrader("noise_3", {"arrival_rate": 0.5, "market_order_pct": 0.4, "max_spread": initial_mid_price * 0.004, "quantity": 1.5, "seed": seed + 2}),
        Fundamentalist("fund_1", {"fundamental_value": initial_mid_price, "threshold": initial_mid_price * 0.003, "aggression": 0.5, "quantity": 2.0, "seed": seed + 3}),
        InformedTrader("informed_1", {"look_ahead": 10, "accuracy": 0.7, "arrival_rate": 0.3, "quantity": 3.0, "seed": seed + 4}),
    ]

    # Set price path for informed trader
    agents[-1].set_future_prices(price_path.tolist())

    runner = LLMSimulationRunner(
        lob=lob,
        mm_agent=mm,
        agents=agents,
        max_ticks=min(max_ticks, len(price_path)),
        initial_mid_price=initial_mid_price,
        seed=seed,
        mode="as",
    )

    runner.run()
    return runner


def run_llm_simulation(
    initial_mid_price: float,
    max_ticks: int = 1000,
    num_agents: int = 50,
    seed: int = 42,
    gamma: float = 0.1,
    sigma: float = 0.01,
    llm_client: LLMClient | None = None,
    price_path: np.ndarray | None = None,
) -> LLMSimulationRunner:
    """
    Run LLM-based heterogeneous agent simulation.

    Args:
        initial_mid_price: Starting mid price.
        max_ticks: Number of ticks.
        num_agents: Number of LLM agents.
        seed: Random seed.
        gamma: MM risk aversion.
        sigma: Volatility estimate.
        llm_client: LLM client (creates new if None).
        price_path: Real price path for informed/fundamental agents.

    Returns:
        Completed LLMSimulationRunner in "llm" mode.
    """
    # Calibrate k
    target_spread = initial_mid_price * 0.002
    k = max(0.1, 2.0 * gamma / (np.exp(gamma * target_spread / 2.0) - 1.0))

    lob = LOBEngine(tick_size=0.01)
    mm = HelixMMAgent(
        agent_id="mm_llm",
        params={
            "gamma": gamma,
            "k": k,
            "sigma": sigma,
            "T": 1.0,
            "quantity": 1.0,
            "max_inventory": 10,
        },
    )

    # Generate LLM agent profiles
    profiles = generate_profiles_deterministic(total_agents=num_agents, seed=seed)

    # Create LLM agents
    client = llm_client or LLMClient()
    llm_agents = create_llm_agents(profiles, client, base_quantity=1.0, seed=seed)

    # Add rule-based anchoring agents for price discovery and liquidity
    # These ensure the LOB stays deep and prices stay realistic
    rule_agents: list = [
        NoiseTrader("llm_noise_bg_1", {"arrival_rate": 1.0, "market_order_pct": 0.3, "max_spread": initial_mid_price * 0.003, "quantity": 0.5, "seed": seed + 100}),
        NoiseTrader("llm_noise_bg_2", {"arrival_rate": 0.8, "market_order_pct": 0.2, "max_spread": initial_mid_price * 0.002, "quantity": 0.5, "seed": seed + 103}),
        NoiseTrader("llm_noise_bg_3", {"arrival_rate": 0.5, "market_order_pct": 0.4, "max_spread": initial_mid_price * 0.004, "quantity": 1.0, "seed": seed + 104}),
        Fundamentalist("llm_fund_bg_1", {"fundamental_value": initial_mid_price, "threshold": initial_mid_price * 0.003, "aggression": 0.5, "quantity": 1.5, "seed": seed + 101}),
    ]

    # Add informed trader with real price path if available
    if price_path is not None:
        informed = InformedTrader("llm_informed_bg_1", {
            "look_ahead": 10, "accuracy": 0.7, "arrival_rate": 0.3,
            "quantity": 3.0, "seed": seed + 102,
        })
        informed.set_future_prices(price_path.tolist())
        rule_agents.append(informed)

    all_agents = llm_agents + rule_agents

    runner = LLMSimulationRunner(
        lob=lob,
        mm_agent=mm,
        agents=all_agents,
        max_ticks=max_ticks,
        initial_mid_price=initial_mid_price,
        seed=seed,
        mode="llm",
    )

    runner.run()
    return runner


def compute_scorecard(
    real_facts: dict,
    sim_facts: dict,
    real_returns: np.ndarray,
    sim_returns: np.ndarray,
) -> dict:
    """
    Compute a scorecard comparing simulation to real data.

    Lower total_score = better match to real data.
    """
    scores = {}

    # 1. Kurtosis match
    real_kurt = real_facts["return_stats"]["kurtosis"]
    sim_kurt = sim_facts["return_stats"]["kurtosis"]
    scores["kurtosis_distance"] = abs(real_kurt - sim_kurt)

    # 2. Skewness match
    real_skew = real_facts["return_stats"]["skewness"]
    sim_skew = sim_facts["return_stats"]["skewness"]
    scores["skewness_distance"] = abs(real_skew - sim_skew)

    # 3. Volatility clustering (ACF of |returns| match)
    real_acf = real_facts["volatility_clustering"].get("acf_abs_returns", [])
    sim_acf = sim_facts["volatility_clustering"].get("acf_abs_returns", [])
    if real_acf and sim_acf:
        min_len = min(len(real_acf), len(sim_acf))
        acf_diff = np.array(real_acf[:min_len]) - np.array(sim_acf[:min_len])
        scores["acf_abs_returns_rmse"] = float(np.sqrt(np.mean(acf_diff**2)))
    else:
        scores["acf_abs_returns_rmse"] = 1.0

    # 4. Hurst exponent match
    real_hurst = real_facts["hurst_exponent"]
    sim_hurst = sim_facts["hurst_exponent"]
    scores["hurst_distance"] = abs(real_hurst - sim_hurst)

    # 5. Distribution comparison (KS test, Wasserstein)
    dist_comparison = compare_distributions(real_returns, sim_returns)
    scores["ks_statistic"] = dist_comparison["ks_statistic"]
    scores["wasserstein_distance"] = dist_comparison["wasserstein_distance"]

    # 6. Fat tail index match
    real_hill = real_facts["fat_tails"]["hill_estimator"]
    sim_hill = sim_facts["fat_tails"]["hill_estimator"]
    if real_hill > 0 and sim_hill > 0:
        scores["hill_distance"] = abs(real_hill - sim_hill) / max(real_hill, sim_hill)
    else:
        scores["hill_distance"] = 1.0

    # Weighted total score (lower = better)
    weights = {
        "kurtosis_distance": 0.15,
        "skewness_distance": 0.10,
        "acf_abs_returns_rmse": 0.20,
        "hurst_distance": 0.10,
        "ks_statistic": 0.15,
        "wasserstein_distance": 0.20,
        "hill_distance": 0.10,
    }
    total = sum(scores[k] * weights[k] for k in weights)
    scores["total_score"] = total

    return scores


def run_comparison(
    symbol: str = "BTCUSDT",
    data_dir: str = "~/intraday_trading/data/futures_ticks",
    max_ticks: int = 5000,
    downsample_n: int = 100,
    num_llm_agents: int = 50,
    seed: int = 42,
    llm_client: LLMClient | None = None,
) -> ComparisonResult:
    """
    Run full A-S vs LLM comparison on real Binance data.

    Args:
        symbol: Trading pair.
        data_dir: Path to futures tick data.
        max_ticks: Number of simulation ticks.
        downsample_n: Downsample factor for real data.
        num_llm_agents: Number of LLM agents.
        seed: Random seed.
        llm_client: LLM client instance.

    Returns:
        ComparisonResult with all metrics.
    """
    result = ComparisonResult()

    # 1. Load real data
    logger.info(f"Loading {symbol} data from {data_dir}...")
    real_prices = get_price_path(symbol, data_dir, max_ticks, downsample_n)
    result.real_prices = real_prices
    initial_mid = float(real_prices[0])

    # Compute real sigma
    real_returns_arr = np.diff(np.log(real_prices))
    sigma = float(np.std(real_returns_arr))

    # 2. Real data stylized facts
    logger.info("Computing real data stylized facts...")
    result.real_facts = compute_all_stylized_facts(real_prices)

    # 3. Run A-S simulation
    logger.info("Running A-S baseline simulation...")
    as_runner = run_as_simulation(
        initial_mid_price=initial_mid,
        price_path=real_prices,
        max_ticks=max_ticks,
        seed=seed,
        sigma=sigma,
    )
    result.as_records = as_runner._records
    result.as_summary = as_runner.get_results_summary()
    as_prices = np.array(as_runner.get_price_series())
    result.as_facts = compute_all_stylized_facts(as_prices)

    # 4. Run LLM simulation
    logger.info(f"Running LLM simulation with {num_llm_agents} agents...")
    llm_runner = run_llm_simulation(
        initial_mid_price=initial_mid,
        max_ticks=max_ticks,
        num_agents=num_llm_agents,
        seed=seed,
        sigma=sigma,
        llm_client=llm_client,
        price_path=real_prices,
    )
    result.llm_records = llm_runner._records
    result.llm_summary = llm_runner.get_results_summary()
    llm_prices = np.array(llm_runner.get_price_series())
    result.llm_facts = compute_all_stylized_facts(llm_prices)

    # 5. Compute scorecards
    real_returns = compute_log_returns(real_prices)
    as_returns = compute_log_returns(as_prices)
    llm_returns = compute_log_returns(llm_prices)

    result.scorecard = {
        "as": compute_scorecard(result.real_facts, result.as_facts, real_returns, as_returns),
        "llm": compute_scorecard(result.real_facts, result.llm_facts, real_returns, llm_returns),
    }

    # Log summary
    as_score = result.scorecard["as"]["total_score"]
    llm_score = result.scorecard["llm"]["total_score"]
    winner = "LLM" if llm_score < as_score else "A-S"
    logger.info(f"Scorecard: A-S={as_score:.4f}, LLM={llm_score:.4f} → Winner: {winner}")

    return result
