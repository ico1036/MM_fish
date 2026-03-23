"""
Hands-on: RWA S&P500 Perpetual MM Backtest PoC

Runs the MM algorithm against LLM agents in two scenarios:
1. Normal listing day
2. S&P500 -3% crash

Usage:
    uv run --no-env-file python scripts/hands_on.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.services.comparison_engine import run_as_simulation, run_llm_simulation, compute_scorecard
from app.services.llm_client import LLMClient
from app.services.scenario_engine import Scenario, ScenarioEngine
from app.services.stylized_facts import compute_all_stylized_facts, compute_log_returns

INITIAL_PRICE = 5620.0
MAX_TICKS = 500
NUM_AGENTS = 50
SEED = 42

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("RWA S&P500 Perpetual MM Backtest PoC")
print("=" * 60)

# --- Scenario 1: Normal Listing ---
print("\n[1/2] Normal listing scenario...")
scenario_normal = Scenario.normal_listing(INITIAL_PRICE, MAX_TICKS, seed=SEED)
llm_client = LLMClient()

runner_normal = run_llm_simulation(
    initial_mid_price=INITIAL_PRICE,
    llm_client=llm_client,
    max_ticks=MAX_TICKS,
    num_agents=NUM_AGENTS,
    seed=SEED,
    scenario=ScenarioEngine(scenario_normal),
)
normal_prices = np.array(runner_normal.get_price_series())
normal_records = runner_normal._records
print(f"  Ticks: {len(normal_records)}, Trades: {sum(r.num_trades for r in normal_records)}")
print(f"  MM PnL: ${normal_records[-1].mm_pnl:.2f}, Final inventory: {normal_records[-1].mm_inventory:.1f}")

# --- Scenario 2: S&P500 Crash ---
print("\n[2/2] S&P500 crash scenario (-3% at tick 250)...")
scenario_crash = Scenario.sp500_crash(INITIAL_PRICE, MAX_TICKS, crash_tick=250, crash_pct=0.03, seed=SEED)

runner_crash = run_llm_simulation(
    initial_mid_price=INITIAL_PRICE,
    llm_client=llm_client,
    max_ticks=MAX_TICKS,
    num_agents=NUM_AGENTS,
    seed=SEED + 1,
    scenario=ScenarioEngine(scenario_crash),
)
crash_prices = np.array(runner_crash.get_price_series())
crash_records = runner_crash._records
crash_liqs = sum(r.num_liquidations for r in crash_records)
print(f"  Ticks: {len(crash_records)}, Trades: {sum(r.num_trades for r in crash_records)}")
print(f"  MM PnL: ${crash_records[-1].mm_pnl:.2f}, Final inventory: {crash_records[-1].mm_inventory:.1f}")
print(f"  Liquidations: {crash_liqs}")

# --- Plots ---
print("\nGenerating plots...")

# Plot 1: Price comparison
fig1 = make_subplots(rows=2, cols=1, subplot_titles=["Normal Listing", "S&P500 Crash"])
fig1.add_trace(go.Scatter(y=scenario_normal.index_prices, name="S&P Index", line=dict(color="blue")), row=1, col=1)
fig1.add_trace(go.Scatter(y=normal_prices, name="XYZ Perp", line=dict(color="green")), row=1, col=1)
fig1.add_trace(go.Scatter(y=scenario_crash.index_prices, name="S&P Index", line=dict(color="blue")), row=2, col=1)
fig1.add_trace(go.Scatter(y=crash_prices, name="XYZ Perp", line=dict(color="red")), row=2, col=1)
fig1.update_layout(height=700, title="XYZ/S&P500 Perp vs Index", template="plotly_white")
fig1.write_html(str(output_dir / "1_price_scenarios.html"))

# Plot 2: MM inventory & PnL
fig2 = make_subplots(rows=2, cols=2,
    subplot_titles=["Normal: MM Inventory", "Normal: MM PnL", "Crash: MM Inventory", "Crash: MM PnL"])
fig2.add_trace(go.Scatter(y=[r.mm_inventory for r in normal_records], name="Inventory"), row=1, col=1)
fig2.add_trace(go.Scatter(y=[r.mm_pnl for r in normal_records], name="PnL"), row=1, col=2)
fig2.add_trace(go.Scatter(y=[r.mm_inventory for r in crash_records], name="Inventory", line=dict(color="red")), row=2, col=1)
fig2.add_trace(go.Scatter(y=[r.mm_pnl for r in crash_records], name="PnL", line=dict(color="red")), row=2, col=2)
fig2.update_layout(height=600, title="MM Agent Performance", template="plotly_white")
fig2.write_html(str(output_dir / "2_mm_performance.html"))

# Plot 3: Spreads
fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=[r.spread for r in normal_records], name="Normal", line=dict(color="green")))
fig3.add_trace(go.Scatter(y=[r.spread for r in crash_records], name="Crash", line=dict(color="red")))
fig3.update_layout(title="Bid-Ask Spread", xaxis_title="Tick", yaxis_title="Spread ($)", template="plotly_white")
fig3.write_html(str(output_dir / "3_spreads.html"))

# Plot 4: Funding rates
funding_ticks_crash = [(r.tick, r.funding_rate) for r in crash_records if r.funding_rate is not None]
if funding_ticks_crash:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=[t for t, _ in funding_ticks_crash], y=[f for _, f in funding_ticks_crash], name="Funding Rate"))
    fig4.update_layout(title="Funding Rate (Crash Scenario)", xaxis_title="Tick", yaxis_title="Rate", template="plotly_white")
    fig4.write_html(str(output_dir / "4_funding_rates.html"))

print(f"\nPlots saved to {output_dir}/")
for f in sorted(output_dir.glob("*.html")):
    print(f"  open {f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Metric':<30} {'Normal':>12} {'Crash':>12}")
print("  " + "-" * 54)
print(f"  {'MM Final PnL':<30} {'$' + f'{normal_records[-1].mm_pnl:.2f}':>12} {'$' + f'{crash_records[-1].mm_pnl:.2f}':>12}")
print(f"  {'MM Max Inventory':<30} {max(abs(r.mm_inventory) for r in normal_records):>12.1f} {max(abs(r.mm_inventory) for r in crash_records):>12.1f}")
print(f"  {'Total Trades':<30} {sum(r.num_trades for r in normal_records):>12} {sum(r.num_trades for r in crash_records):>12}")
print(f"  {'Liquidations':<30} {sum(r.num_liquidations for r in normal_records):>12} {crash_liqs:>12}")
print(f"  {'LLM Calls':<30} {llm_client.call_count:>12}")
print("=" * 60)
