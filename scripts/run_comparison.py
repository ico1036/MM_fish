"""CLI tool to run A-S vs LLM comparison simulation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.comparison_engine import run_comparison, run_as_simulation, run_llm_simulation
from app.services.llm_client import LLMClient
from app.services.binance_data import get_price_path
from app.services.stylized_facts import compute_all_stylized_facts, compute_log_returns
from app.services.comparison_engine import compute_scorecard, ComparisonResult

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run A-S vs LLM market simulation comparison")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of simulation ticks")
    parser.add_argument("--agents", type=int, default=50, help="Number of LLM agents")
    parser.add_argument("--downsample", type=int, default=100, help="Downsample factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default="~/intraday_trading/data/futures_ticks", help="Data directory")
    parser.add_argument("--output", default="results/comparison.json", help="Output JSON file")
    parser.add_argument("--mode", choices=["both", "as", "llm"], default="both", help="Which simulations to run")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (use fallback agents)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load real data
    logger.info(f"Loading {args.symbol} data...")
    real_prices = get_price_path(args.symbol, args.data_dir, args.ticks, args.downsample)
    initial_mid = float(real_prices[0])
    real_returns = np.diff(np.log(real_prices))
    sigma = float(np.std(real_returns))
    logger.info(f"Loaded {len(real_prices)} ticks, initial_mid={initial_mid:.2f}, sigma={sigma:.6f}")

    # Real data facts
    real_facts = compute_all_stylized_facts(real_prices)
    real_log_returns = compute_log_returns(real_prices)

    result = ComparisonResult()
    result.real_prices = real_prices
    result.real_facts = real_facts

    # A-S simulation
    if args.mode in ("both", "as"):
        logger.info("Running A-S baseline simulation...")
        as_runner = run_as_simulation(
            initial_mid_price=initial_mid,
            price_path=real_prices,
            max_ticks=args.ticks,
            seed=args.seed,
            sigma=sigma,
        )
        result.as_records = as_runner._records
        result.as_summary = as_runner.get_results_summary()
        as_prices = np.array(as_runner.get_price_series())
        result.as_facts = compute_all_stylized_facts(as_prices)

        as_returns = compute_log_returns(as_prices)
        as_score = compute_scorecard(real_facts, result.as_facts, real_log_returns, as_returns)
        logger.info(f"A-S Score: {as_score['total_score']:.4f}")
        _print_scorecard("A-S Baseline", as_score)

    # LLM simulation
    if args.mode in ("both", "llm"):
        logger.info(f"Running LLM simulation with {args.agents} agents...")
        llm_client = None
        if not args.no_llm:
            llm_client = LLMClient()

        llm_runner = run_llm_simulation(
            initial_mid_price=initial_mid,
            max_ticks=args.ticks,
            num_agents=args.agents,
            seed=args.seed,
            sigma=sigma,
            llm_client=llm_client,
            price_path=real_prices,
        )
        result.llm_records = llm_runner._records
        result.llm_summary = llm_runner.get_results_summary()
        llm_prices = np.array(llm_runner.get_price_series())
        result.llm_facts = compute_all_stylized_facts(llm_prices)

        llm_returns = compute_log_returns(llm_prices)
        llm_score = compute_scorecard(real_facts, result.llm_facts, real_log_returns, llm_returns)
        logger.info(f"LLM Score: {llm_score['total_score']:.4f}")
        _print_scorecard("LLM Agents", llm_score)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build serializable output
    output = {
        "config": {
            "symbol": args.symbol,
            "ticks": args.ticks,
            "agents": args.agents,
            "downsample": args.downsample,
            "seed": args.seed,
        },
        "real_facts": real_facts,
    }

    if result.as_facts:
        output["as_facts"] = result.as_facts
        output["as_summary"] = result.as_summary
        output["as_prices"] = [r.mid_price for r in result.as_records]
        output["as_spreads"] = [r.spread for r in result.as_records]

    if result.llm_facts:
        output["llm_facts"] = result.llm_facts
        output["llm_summary"] = result.llm_summary
        output["llm_prices"] = [r.mid_price for r in result.llm_records]
        output["llm_spreads"] = [r.spread for r in result.llm_records]

    output["real_prices"] = real_prices.tolist()

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print comparison
    if args.mode == "both" and result.as_facts and result.llm_facts:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'Real':>10} {'A-S':>10} {'LLM':>10}")
        print("-" * 60)
        print(f"{'Kurtosis':<30} {real_facts['return_stats']['kurtosis']:>10.2f} {result.as_facts['return_stats']['kurtosis']:>10.2f} {result.llm_facts['return_stats']['kurtosis']:>10.2f}")
        print(f"{'Skewness':<30} {real_facts['return_stats']['skewness']:>10.2f} {result.as_facts['return_stats']['skewness']:>10.2f} {result.llm_facts['return_stats']['skewness']:>10.2f}")
        print(f"{'Hurst Exponent':<30} {real_facts['hurst_exponent']:>10.3f} {result.as_facts['hurst_exponent']:>10.3f} {result.llm_facts['hurst_exponent']:>10.3f}")
        print(f"{'Hill Estimator':<30} {real_facts['fat_tails']['hill_estimator']:>10.2f} {result.as_facts['fat_tails']['hill_estimator']:>10.2f} {result.llm_facts['fat_tails']['hill_estimator']:>10.2f}")
        print("=" * 60)


def _print_scorecard(name: str, scorecard: dict):
    """Pretty-print a scorecard."""
    print(f"\n--- {name} Scorecard ---")
    for key, value in scorecard.items():
        if key != "total_score":
            print(f"  {key:<30}: {value:.4f}")
    print(f"  {'TOTAL SCORE':<30}: {scorecard['total_score']:.4f}")


if __name__ == "__main__":
    main()
