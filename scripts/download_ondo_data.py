"""Download ONDOUSDT recent trades from Binance API."""

import csv
import json
import urllib.request
from pathlib import Path


def download_trades(symbol: str = "ONDOUSDT", limit: int = 1000) -> Path:
    """Download recent trades and save as CSV."""
    url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "helix-mm/0.1"})

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_trades.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "price", "quantity", "side"])
        writer.writeheader()
        for trade in data:
            writer.writerow({
                "timestamp": trade["time"],
                "price": trade["price"],
                "quantity": trade["qty"],
                "side": "ask" if trade["isBuyerMaker"] else "bid",
            })

    print(f"Downloaded {len(data)} trades to {out_path}")
    return out_path


if __name__ == "__main__":
    download_trades()
