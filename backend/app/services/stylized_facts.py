"""Statistical analysis for market microstructure stylized facts."""

import numpy as np
from scipy import stats


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def compute_return_stats(prices: np.ndarray) -> dict:
    """
    Compute basic return distribution statistics.

    Returns:
        dict with kurtosis, skewness, jarque_bera_stat, jarque_bera_pvalue
    """
    returns = compute_log_returns(prices)
    if len(returns) < 4:
        return {"kurtosis": 0.0, "skewness": 0.0, "jarque_bera_stat": 0.0, "jarque_bera_pvalue": 1.0}

    kurt = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis
    skew = float(stats.skew(returns))
    jb_stat, jb_pvalue = stats.jarque_bera(returns)

    return {
        "kurtosis": kurt,
        "skewness": skew,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
    }


def compute_volatility_clustering(prices: np.ndarray, max_lags: int = 20) -> dict:
    """
    Measure volatility clustering via autocorrelation of absolute returns.

    Returns:
        dict with acf_abs_returns (list), ljung_box_stat, ljung_box_pvalue
    """
    returns = compute_log_returns(prices)
    if len(returns) < max_lags + 2:
        return {"acf_abs_returns": [], "ljung_box_stat": 0.0, "ljung_box_pvalue": 1.0}

    abs_returns = np.abs(returns)

    # Compute ACF of absolute returns
    from statsmodels.tsa.stattools import acf

    acf_values = acf(abs_returns, nlags=max_lags, fft=True)

    # Ljung-Box test on absolute returns
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_result = acorr_ljungbox(abs_returns, lags=[max_lags], return_df=True)
    lb_stat = float(lb_result["lb_stat"].iloc[0])
    lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])

    return {
        "acf_abs_returns": acf_values.tolist(),
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
    }


def compute_autocorrelation(prices: np.ndarray, max_lags: int = 20) -> dict:
    """
    Compute autocorrelation of raw returns and absolute returns.

    Returns:
        dict with acf_returns, acf_abs_returns (lists)
    """
    returns = compute_log_returns(prices)
    if len(returns) < max_lags + 2:
        return {"acf_returns": [], "acf_abs_returns": []}

    from statsmodels.tsa.stattools import acf

    acf_ret = acf(returns, nlags=max_lags, fft=True)
    acf_abs = acf(np.abs(returns), nlags=max_lags, fft=True)

    return {
        "acf_returns": acf_ret.tolist(),
        "acf_abs_returns": acf_abs.tolist(),
    }


def compute_fat_tails(prices: np.ndarray, tail_fraction: float = 0.05) -> dict:
    """
    Analyze fat tails of return distribution.

    Returns:
        dict with hill_estimator, tail_index, qq_theoretical, qq_empirical
    """
    returns = compute_log_returns(prices)
    if len(returns) < 20:
        return {"hill_estimator": 0.0, "tail_index": 0.0, "qq_theoretical": [], "qq_empirical": []}

    abs_returns = np.sort(np.abs(returns))[::-1]  # descending
    n_tail = max(int(len(abs_returns) * tail_fraction), 5)
    tail_values = abs_returns[:n_tail]

    # Hill estimator
    if tail_values[-1] > 0:
        log_ratios = np.log(tail_values[:-1] / tail_values[-1])
        mean_log = float(np.mean(log_ratios))
        hill = float(1.0 / mean_log) if mean_log > 0 else 0.0
    else:
        hill = 0.0

    # QQ plot data (normal vs empirical)
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, min(n, 200)))
    empirical_idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
    empirical = sorted_returns[empirical_idx]

    return {
        "hill_estimator": hill,
        "tail_index": 1.0 / hill if hill > 0 else float("inf"),
        "qq_theoretical": theoretical.tolist(),
        "qq_empirical": empirical.tolist(),
    }


def compute_hurst_exponent(prices: np.ndarray, max_lag: int | None = None) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5: long memory (trending)
    H = 0.5: random walk
    H < 0.5: mean-reverting

    Returns:
        Hurst exponent estimate.
    """
    prices = np.asarray(prices, dtype=float)
    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 20:
        return 0.5

    if max_lag is None:
        max_lag = min(n // 2, 200)

    lags = []
    rs_values = []

    for lag in range(10, max_lag + 1, max(1, max_lag // 20)):
        rs_list = []
        for start in range(0, n - lag, lag):
            segment = returns[start : start + lag]
            mean_r = np.mean(segment)
            deviations = np.cumsum(segment - mean_r)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(segment)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    if len(lags) < 2:
        return 0.5

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)

    return float(np.clip(slope, 0.0, 1.0))


def compare_distributions(
    real_returns: np.ndarray,
    sim_returns: np.ndarray,
) -> dict:
    """
    Compare two return distributions using statistical tests.

    Returns:
        dict with ks_statistic, ks_pvalue, wasserstein_distance
    """
    real_returns = np.asarray(real_returns, dtype=float)
    sim_returns = np.asarray(sim_returns, dtype=float)

    if len(real_returns) < 2 or len(sim_returns) < 2:
        return {"ks_statistic": 1.0, "ks_pvalue": 0.0, "wasserstein_distance": float("inf")}

    # Normalize both to zero mean, unit variance for fair comparison
    r_std = np.std(real_returns)
    s_std = np.std(sim_returns)
    if r_std > 0:
        real_norm = (real_returns - np.mean(real_returns)) / r_std
    else:
        real_norm = real_returns
    if s_std > 0:
        sim_norm = (sim_returns - np.mean(sim_returns)) / s_std
    else:
        sim_norm = sim_returns

    ks_stat, ks_pvalue = stats.ks_2samp(real_norm, sim_norm)
    wass_dist = float(stats.wasserstein_distance(real_norm, sim_norm))

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": wass_dist,
    }


def compute_all_stylized_facts(prices: np.ndarray) -> dict:
    """Compute all stylized facts for a price series."""
    return {
        "return_stats": compute_return_stats(prices),
        "volatility_clustering": compute_volatility_clustering(prices),
        "autocorrelation": compute_autocorrelation(prices),
        "fat_tails": compute_fat_tails(prices),
        "hurst_exponent": compute_hurst_exponent(prices),
    }
