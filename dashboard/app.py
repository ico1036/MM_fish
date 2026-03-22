"""Streamlit dashboard for A-S vs LLM market simulation comparison."""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

st.set_page_config(page_title="Helix MM Fish - A-S vs LLM Comparison", layout="wide")
st.title("Helix MM Fish: A-S vs LLM Market Simulator")


def load_results(path: str) -> dict:
    """Load comparison results from JSON."""
    with open(path) as f:
        return json.load(f)


# Sidebar: Configuration
st.sidebar.header("Configuration")
results_file = st.sidebar.text_input("Results file", value="results/comparison.json")

# Check if results file exists
results_path = Path(__file__).parent.parent / results_file
if not results_path.exists():
    st.warning(f"Results file not found: {results_path}")
    st.info("Run the comparison first:")
    st.code("uv run python scripts/run_comparison.py --symbol BTCUSDT --ticks 1000", language="bash")
    st.stop()

data = load_results(results_path)
config = data.get("config", {})

st.sidebar.markdown(f"""
**Symbol**: {config.get('symbol', 'N/A')}
**Ticks**: {config.get('ticks', 'N/A')}
**LLM Agents**: {config.get('agents', 'N/A')}
**Downsample**: {config.get('downsample', 'N/A')}
""")

# Extract data
real_prices = np.array(data.get("real_prices", []))
as_prices = np.array(data.get("as_prices", []))
llm_prices = np.array(data.get("llm_prices", []))
as_spreads = np.array(data.get("as_spreads", []))
llm_spreads = np.array(data.get("llm_spreads", []))

real_facts = data.get("real_facts", {})
as_facts = data.get("as_facts", {})
llm_facts = data.get("llm_facts", {})

has_as = len(as_prices) > 0
has_llm = len(llm_prices) > 0

# ============================================================
# Tab 1: Price & Spread
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Price & Spread",
    "Stylized Facts",
    "Market Microstructure",
    "Scorecard",
])

with tab1:
    st.header("Price Comparison")

    # Price overlay
    fig_price = go.Figure()
    if len(real_prices) > 0:
        fig_price.add_trace(go.Scatter(
            y=real_prices, name="Real", line=dict(color="blue", width=1),
            opacity=0.8,
        ))
    if has_as:
        fig_price.add_trace(go.Scatter(
            y=as_prices, name="A-S Baseline", line=dict(color="red", width=1),
            opacity=0.7,
        ))
    if has_llm:
        fig_price.add_trace(go.Scatter(
            y=llm_prices, name="LLM Agents", line=dict(color="green", width=1),
            opacity=0.7,
        ))
    fig_price.update_layout(
        title="Mid Price: Real vs A-S vs LLM",
        xaxis_title="Tick",
        yaxis_title="Price",
        height=500,
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Spread comparison
    col1, col2 = st.columns(2)
    with col1:
        if has_as and len(as_spreads) > 0:
            fig_as_spread = go.Figure()
            fig_as_spread.add_trace(go.Histogram(x=as_spreads, name="A-S Spread", nbinsx=50, marker_color="red", opacity=0.6))
            fig_as_spread.update_layout(title="A-S Spread Distribution", xaxis_title="Spread", yaxis_title="Count", height=350)
            st.plotly_chart(fig_as_spread, use_container_width=True)

    with col2:
        if has_llm and len(llm_spreads) > 0:
            fig_llm_spread = go.Figure()
            fig_llm_spread.add_trace(go.Histogram(x=llm_spreads, name="LLM Spread", nbinsx=50, marker_color="green", opacity=0.6))
            fig_llm_spread.update_layout(title="LLM Spread Distribution", xaxis_title="Spread", yaxis_title="Count", height=350)
            st.plotly_chart(fig_llm_spread, use_container_width=True)

# ============================================================
# Tab 2: Stylized Facts
# ============================================================
with tab2:
    st.header("Stylized Facts Comparison")

    # Return distribution QQ plots
    col1, col2, col3 = st.columns(3)

    def make_qq_plot(facts: dict, name: str, color: str):
        """Create QQ plot from fat_tails data."""
        qq_t = facts.get("fat_tails", {}).get("qq_theoretical", [])
        qq_e = facts.get("fat_tails", {}).get("qq_empirical", [])
        if qq_t and qq_e:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=qq_t, y=qq_e, mode="markers",
                marker=dict(size=3, color=color), name=name,
            ))
            # 45-degree line
            min_val = min(min(qq_t), min(qq_e))
            max_val = max(max(qq_t), max(qq_e))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(color="gray", dash="dash"),
                name="Normal",
            ))
            fig.update_layout(
                title=f"QQ Plot: {name}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Empirical Quantiles",
                height=350,
                showlegend=False,
            )
            return fig
        return None

    with col1:
        fig = make_qq_plot(real_facts, "Real Data", "blue")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_as:
            fig = make_qq_plot(as_facts, "A-S Baseline", "red")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col3:
        if has_llm:
            fig = make_qq_plot(llm_facts, "LLM Agents", "green")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # ACF comparison
    st.subheader("Autocorrelation of |Returns|")
    fig_acf = go.Figure()

    def add_acf_trace(facts, name, color):
        acf_vals = facts.get("volatility_clustering", {}).get("acf_abs_returns", [])
        if acf_vals:
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name=name,
                line=dict(color=color, width=2),
            ))

    add_acf_trace(real_facts, "Real", "blue")
    if has_as:
        add_acf_trace(as_facts, "A-S", "red")
    if has_llm:
        add_acf_trace(llm_facts, "LLM", "green")

    fig_acf.update_layout(
        title="ACF of Absolute Returns (Volatility Clustering)",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        height=400,
    )
    st.plotly_chart(fig_acf, use_container_width=True)

    # Kurtosis/Skewness bar chart
    st.subheader("Distribution Shape")
    col1, col2 = st.columns(2)

    with col1:
        labels = []
        kurt_values = []
        colors = []

        if real_facts:
            labels.append("Real")
            kurt_values.append(real_facts.get("return_stats", {}).get("kurtosis", 0))
            colors.append("blue")
        if has_as and as_facts:
            labels.append("A-S")
            kurt_values.append(as_facts.get("return_stats", {}).get("kurtosis", 0))
            colors.append("red")
        if has_llm and llm_facts:
            labels.append("LLM")
            kurt_values.append(llm_facts.get("return_stats", {}).get("kurtosis", 0))
            colors.append("green")

        fig_kurt = go.Figure(go.Bar(x=labels, y=kurt_values, marker_color=colors))
        fig_kurt.update_layout(title="Excess Kurtosis (higher = fatter tails)", height=300)
        fig_kurt.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Normal")
        st.plotly_chart(fig_kurt, use_container_width=True)

    with col2:
        labels2 = []
        skew_values = []
        colors2 = []

        if real_facts:
            labels2.append("Real")
            skew_values.append(real_facts.get("return_stats", {}).get("skewness", 0))
            colors2.append("blue")
        if has_as and as_facts:
            labels2.append("A-S")
            skew_values.append(as_facts.get("return_stats", {}).get("skewness", 0))
            colors2.append("red")
        if has_llm and llm_facts:
            labels2.append("LLM")
            skew_values.append(llm_facts.get("return_stats", {}).get("skewness", 0))
            colors2.append("green")

        fig_skew = go.Figure(go.Bar(x=labels2, y=skew_values, marker_color=colors2))
        fig_skew.update_layout(title="Skewness (0 = symmetric)", height=300)
        fig_skew.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_skew, use_container_width=True)

# ============================================================
# Tab 3: Market Microstructure
# ============================================================
with tab3:
    st.header("Market Microstructure")

    # Return distribution histograms
    st.subheader("Return Distribution")
    fig_returns = go.Figure()

    if len(real_prices) > 1:
        real_ret = np.diff(np.log(real_prices))
        fig_returns.add_trace(go.Histogram(
            x=real_ret, name="Real", nbinsx=100,
            marker_color="blue", opacity=0.5,
        ))

    if has_as and len(as_prices) > 1:
        as_ret = np.diff(np.log(as_prices))
        fig_returns.add_trace(go.Histogram(
            x=as_ret, name="A-S", nbinsx=100,
            marker_color="red", opacity=0.5,
        ))

    if has_llm and len(llm_prices) > 1:
        llm_ret = np.diff(np.log(llm_prices))
        fig_returns.add_trace(go.Histogram(
            x=llm_ret, name="LLM", nbinsx=100,
            marker_color="green", opacity=0.5,
        ))

    fig_returns.update_layout(
        title="Log Return Distribution",
        xaxis_title="Log Return",
        yaxis_title="Count",
        barmode="overlay",
        height=450,
    )
    st.plotly_chart(fig_returns, use_container_width=True)

    # Simulation summaries
    col1, col2 = st.columns(2)
    with col1:
        if has_as:
            st.subheader("A-S Simulation Summary")
            as_summary = data.get("as_summary", {})
            st.json(as_summary)

    with col2:
        if has_llm:
            st.subheader("LLM Simulation Summary")
            llm_summary = data.get("llm_summary", {})
            st.json(llm_summary)

    # Hurst exponent comparison
    st.subheader("Long Memory (Hurst Exponent)")
    col1, col2, col3 = st.columns(3)
    with col1:
        h = real_facts.get("hurst_exponent", 0.5)
        st.metric("Real", f"{h:.3f}", delta=None)
        st.caption("H=0.5: random walk, H>0.5: trending, H<0.5: mean-reverting")
    with col2:
        if has_as:
            h = as_facts.get("hurst_exponent", 0.5)
            st.metric("A-S", f"{h:.3f}")
    with col3:
        if has_llm:
            h = llm_facts.get("hurst_exponent", 0.5)
            st.metric("LLM", f"{h:.3f}")

# ============================================================
# Tab 4: Scorecard
# ============================================================
with tab4:
    st.header("Market Realism Scorecard")
    st.caption("Lower score = closer to real market data")

    # Build scorecard table
    from app.services.stylized_facts import compute_log_returns as clr
    from app.services.comparison_engine import compute_scorecard

    metrics = [
        "kurtosis_distance",
        "skewness_distance",
        "acf_abs_returns_rmse",
        "hurst_distance",
        "ks_statistic",
        "wasserstein_distance",
        "hill_distance",
        "total_score",
    ]

    weights = {
        "kurtosis_distance": 0.15,
        "skewness_distance": 0.10,
        "acf_abs_returns_rmse": 0.20,
        "hurst_distance": 0.10,
        "ks_statistic": 0.15,
        "wasserstein_distance": 0.20,
        "hill_distance": 0.10,
        "total_score": 1.0,
    }

    # Compute scorecards if we have the data
    real_ret = clr(real_prices) if len(real_prices) > 1 else np.array([])

    as_scorecard = {}
    llm_scorecard = {}

    if has_as and len(as_prices) > 1 and as_facts and real_facts:
        as_ret = clr(as_prices)
        as_scorecard = compute_scorecard(real_facts, as_facts, real_ret, as_ret)

    if has_llm and len(llm_prices) > 1 and llm_facts and real_facts:
        llm_ret = clr(llm_prices)
        llm_scorecard = compute_scorecard(real_facts, llm_facts, real_ret, llm_ret)

    # Display as table
    if as_scorecard or llm_scorecard:
        table_data = []
        for m in metrics:
            row = {"Metric": m, "Weight": f"{weights.get(m, 0):.0%}"}
            if as_scorecard:
                row["A-S Score"] = f"{as_scorecard.get(m, 0):.4f}"
            if llm_scorecard:
                row["LLM Score"] = f"{llm_scorecard.get(m, 0):.4f}"
            if as_scorecard and llm_scorecard:
                as_val = as_scorecard.get(m, 0)
                llm_val = llm_scorecard.get(m, 0)
                row["Winner"] = "LLM" if llm_val < as_val else ("A-S" if as_val < llm_val else "Tie")
            table_data.append(row)

        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Overall winner
        if as_scorecard and llm_scorecard:
            as_total = as_scorecard.get("total_score", 0)
            llm_total = llm_scorecard.get("total_score", 0)
            winner = "LLM Agents" if llm_total < as_total else "A-S Baseline"
            margin = abs(as_total - llm_total)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("A-S Total Score", f"{as_total:.4f}")
            with col2:
                st.metric("LLM Total Score", f"{llm_total:.4f}")
            with col3:
                st.metric("Winner", winner, delta=f"by {margin:.4f}")
    else:
        st.info("No scorecard data available. Run comparison first.")
