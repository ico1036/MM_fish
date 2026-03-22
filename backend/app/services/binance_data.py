"""Binance aggTrades parquet data loader for market simulation."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_trades(
    symbol: str,
    data_dir: str | Path = "~/intraday_trading/data/futures_ticks",
    max_rows: int | None = None,
) -> dict:
    """
    Load aggTrades from parquet files.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT").
        data_dir: Base directory containing {SYMBOL}/ subdirectories.
        max_rows: Maximum number of rows to load. None = all.

    Returns:
        dict with keys: prices, timestamps, quantities, is_buyer_maker
    """
    data_dir = Path(data_dir).expanduser()
    symbol_dir = data_dir / symbol

    if not symbol_dir.exists():
        raise FileNotFoundError(f"No data directory for {symbol} at {symbol_dir}")

    # Find all parquet files sorted by name (chronological)
    parquet_files = sorted(symbol_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {symbol_dir}")

    columns = ["price", "quantity", "timestamp", "is_buyer_maker"]
    tables = []
    total_rows = 0

    for pf in parquet_files:
        table = pq.read_table(pf, columns=columns)
        if max_rows is not None and total_rows + table.num_rows > max_rows:
            remaining = max_rows - total_rows
            table = table.slice(0, remaining)
        tables.append(table)
        total_rows += table.num_rows
        if max_rows is not None and total_rows >= max_rows:
            break

    import pyarrow as pa

    combined = pa.concat_tables(tables)

    return {
        "prices": combined.column("price").to_numpy(),
        "timestamps": combined.column("timestamp").to_pandas().values,
        "quantities": combined.column("quantity").to_numpy(),
        "is_buyer_maker": combined.column("is_buyer_maker").to_numpy(),
    }


def downsample_trades(
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    every_n: int = 100,
) -> dict:
    """
    Downsample trade data by taking every Nth trade.

    Args:
        prices: Price array.
        quantities: Quantity array.
        is_buyer_maker: Buyer/maker flag array.
        every_n: Take every Nth trade as one tick.

    Returns:
        dict with downsampled: prices, quantities, is_buyer_maker, buy_volume_ratio
    """
    indices = np.arange(0, len(prices), every_n)
    ds_prices = prices[indices]
    ds_quantities = quantities[indices]
    ds_is_buyer_maker = is_buyer_maker[indices]

    # Compute buy volume ratio in each window
    buy_ratios = []
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(indices) else len(prices)
        window_qty = quantities[start:end]
        window_buyer = is_buyer_maker[start:end]
        total = window_qty.sum()
        if total > 0:
            buy_vol = window_qty[~window_buyer].sum()  # is_buyer_maker=False means buyer is taker
            buy_ratios.append(buy_vol / total)
        else:
            buy_ratios.append(0.5)

    return {
        "prices": ds_prices,
        "quantities": ds_quantities,
        "is_buyer_maker": ds_is_buyer_maker,
        "buy_volume_ratio": np.array(buy_ratios),
    }


def get_price_path(
    symbol: str,
    data_dir: str | Path = "~/intraday_trading/data/futures_ticks",
    max_ticks: int = 5000,
    downsample_n: int = 100,
) -> np.ndarray:
    """
    Convenience function: load and downsample to get a price path for simulation.

    Args:
        symbol: Trading pair.
        data_dir: Data directory.
        max_ticks: Maximum number of ticks in output.
        downsample_n: Downsample factor.

    Returns:
        numpy array of prices.
    """
    raw = load_trades(symbol, data_dir, max_rows=max_ticks * downsample_n)
    ds = downsample_trades(raw["prices"], raw["quantities"], raw["is_buyer_maker"], downsample_n)
    return ds["prices"][:max_ticks]
