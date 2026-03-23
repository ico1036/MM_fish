"""Scenario engine for external price feeds and stress events."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Scenario:
    name: str
    index_prices: np.ndarray
    events: dict[int, str] = field(default_factory=dict)

    @staticmethod
    def normal_listing(initial_price: float, num_ticks: int, seed: int = 42) -> "Scenario":
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.00002, 0.0005, num_ticks)
        prices = initial_price * np.exp(np.cumsum(returns))
        prices[0] = initial_price
        return Scenario(name="normal_listing", index_prices=prices)

    @staticmethod
    def sp500_crash(initial_price: float, num_ticks: int, crash_tick: int, crash_pct: float = 0.03, seed: int = 42) -> "Scenario":
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.00002, 0.0005, num_ticks)
        crash_per_tick = -crash_pct / 5
        for t in range(crash_tick, min(crash_tick + 5, num_ticks)):
            returns[t] = crash_per_tick
        for t in range(crash_tick + 5, min(crash_tick + 50, num_ticks)):
            returns[t] = rng.normal(0.0001, 0.0015)
        prices = initial_price * np.exp(np.cumsum(returns))
        prices[0] = initial_price
        events = {crash_tick: "sp500_crash"}
        return Scenario(name="sp500_crash", index_prices=prices, events=events)

    @staticmethod
    def funding_spike(initial_price: float, num_ticks: int, spike_tick: int, seed: int = 42) -> "Scenario":
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0001, 0.0005, num_ticks)
        prices = initial_price * np.exp(np.cumsum(returns))
        prices[0] = initial_price
        events = {spike_tick: "funding_spike"}
        return Scenario(name="funding_spike", index_prices=prices, events=events)


class ScenarioEngine:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def get_index_price(self, tick: int) -> float:
        if tick < len(self.scenario.index_prices):
            return float(self.scenario.index_prices[tick])
        return float(self.scenario.index_prices[-1])

    def get_event(self, tick: int) -> str | None:
        return self.scenario.events.get(tick)
