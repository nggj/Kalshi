"""Station-centric baseline predictors (no network)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd


class BaselinePredictor(ABC):
    """Interface for station-level Tmax baseline forecasts."""

    @abstractmethod
    def predict_tmax(self, target_date: date, station: str) -> float:
        """Predict Tmax for a station and date."""


class PersistenceBaseline(BaselinePredictor):
    """Persistence baseline using latest observed Tmax or rolling mean."""

    def __init__(self, history: dict[date, float], window_days: int = 1) -> None:
        self.history = history
        self.window_days = window_days

    def predict_tmax(self, target_date: date, station: str) -> float:
        del station
        prior = sorted([d for d in self.history if d < target_date])
        if not prior:
            raise ValueError("No prior observations for persistence baseline")

        if self.window_days <= 1:
            return float(self.history[prior[-1]])

        selected = prior[-self.window_days :]
        vals = [self.history[d] for d in selected]
        return float(sum(vals) / len(vals))


class ClimatologyBaseline(BaselinePredictor):
    """Day-of-year climatology baseline built from observed history."""

    def __init__(self, history: dict[date, float]) -> None:
        self._doy_map: dict[int, list[float]] = {}
        self._global_mean = float(sum(history.values()) / len(history)) if history else 0.0
        for d, tmax in history.items():
            self._doy_map.setdefault(d.timetuple().tm_yday, []).append(float(tmax))

    def predict_tmax(self, target_date: date, station: str) -> float:
        del station
        doy = target_date.timetuple().tm_yday
        vals = self._doy_map.get(doy)
        if not vals:
            return self._global_mean
        return float(sum(vals) / len(vals))


class PublicModelBaselineCSV(BaselinePredictor):
    """CSV adapter for pre-exported public model station forecasts."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def predict_tmax(self, target_date: date, station: str) -> float:
        date_str = target_date.isoformat()
        mask = (self.df["date"] == date_str) & (self.df["station"] == station)
        rows = self.df.loc[mask]
        if rows.empty:
            raise ValueError("No public model baseline value for station/date")
        return float(rows.iloc[0]["tmax"])
