"""Alpha exploration and visualization on Coinbase BTC candles.

DISCLAIMER:
This module is a research prototype created by James Sawyer. It is provided
“as-is” for educational and exploratory purposes only and should not be relied
upon for investment decisions, trading strategies, or financial advice of any
kind. Users assume all responsibility for any outcomes when they run or adapt
this code.

This module loads `coinbase_candles.csv`, builds density-based features,
clusters price regimes, fits simple forward-return models, and renders a
multi-panel dashboard suitable for exploratory analysis.
"""

__author__ = "James Sawyer"
__version__ = "0.1.0"

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

# ---------------------------------------------------------------------------
# Configuration (no CLI or environment parsing)
# ---------------------------------------------------------------------------

CSV_PATH = Path(__file__).with_name("coinbase_candles.csv")
TIME_COLUMN = "time"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"

MAX_ROWS = 5_000

ZONE_BINS = 300
ZONE_SIGMA = 4.0
REGIME_CLUSTERS = 6
FORWARD_HORIZON = 3

GB_MAX_DEPTH = 4
GB_LEARNING_RATE = 0.05
TP_QUANTILE = 0.7

RNG_SEED = 1234
SKLEARN_SEED = 1234
LOG_LEVEL_NAME = "INFO"


def configure_logging() -> logging.Logger:
    """Configure and return a module-level logger."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL_NAME, logging.INFO),
        format="%(asctime)s %(levelname)-8s coinbase_alpha | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("coinbase_alpha_poc")


LOG = configure_logging()


def configure_determinism(seed: int = RNG_SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible behavior."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class AlphaArtifacts:
    """Container for fitted models and derived objects."""

    direction: str
    expected_return: float
    tp_return: float
    tp_price: float
    gb_model: GradientBoostingRegressor
    tp_model: QuantileRegressor
    kde: gaussian_kde
    validation_metrics: Dict[str, float] | None = None


def load_coinbase_ohlc(max_rows: int | None = MAX_ROWS) -> pd.DataFrame:
    """Load OHLC candles from the Coinbase CSV."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Coinbase candles CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {TIME_COLUMN, OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN)

    if max_rows is not None and max_rows > 0:
        df = df.tail(max_rows)

    out = pd.DataFrame(
        {
            "timestamp": df[TIME_COLUMN].dt.tz_convert(None),
            "open": df[OPEN_COLUMN].astype(float),
            "high": df[HIGH_COLUMN].astype(float),
            "low": df[LOW_COLUMN].astype(float),
            "close": df[CLOSE_COLUMN].astype(float),
        },
    )
    return out.reset_index(drop=True)


def build_zones(df: pd.DataFrame, bins: int, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed buyer/seller density profiles over price."""
    g_low = float(df["low"].min())
    g_high = float(df["high"].max())

    price_bins = np.linspace(g_low, g_high, bins + 1)
    buyer = np.zeros(bins, dtype=float)
    seller = np.zeros(bins, dtype=float)

    for _, row in df.iterrows():
        dr = float(row["high"] - row["low"])
        if dr == 0:
            continue

        loc = float((row["close"] - row["low"]) / dr)
        lower_wick = float(min(row["open"], row["close"]) - row["low"])
        upper_wick = float(row["high"] - max(row["open"], row["close"]))

        lo = max(int(np.searchsorted(price_bins, row["low"]) - 1), 0)
        hi = min(int(np.searchsorted(price_bins, row["high"]) - 1), bins - 1)

        if loc < 0.25:
            buyer[lo : hi + 1] += 1.0 + (lower_wick / dr)
        elif loc > 0.75:
            seller[lo : hi + 1] += 1.0 + (upper_wick / dr)

    if sigma > 0:
        buyer = gaussian_filter1d(buyer, sigma)
        seller = gaussian_filter1d(seller, sigma)

    return price_bins, buyer, seller


def assign_density(df: pd.DataFrame, price_bins: np.ndarray, buyer: np.ndarray, seller: np.ndarray) -> pd.DataFrame:
    """Attach density-derived features to the bar data."""
    mid = (price_bins[:-1] + price_bins[1:]) / 2.0
    idx = np.searchsorted(mid, df["close"].values)
    idx = np.clip(idx, 0, len(buyer) - 1)

    out = df.copy()
    out["buyer_density"] = buyer[idx]
    out["seller_density"] = seller[idx]
    out["zone_signal"] = out["buyer_density"] - out["seller_density"]

    dr = (out["high"] - out["low"]).replace(0, np.nan)
    out["loc"] = (out["close"] - out["low"]) / dr

    lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
    upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
    out["wick_ratio"] = lower_wick - upper_wick

    return out


def add_forward(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Append forward-return columns for the given horizons."""
    out = df.copy()
    for horizon in horizons:
        out[f"fwd_{horizon}"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return out


def cluster_regimes(df: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, KMeans]:
    """Cluster bars into regimes based on density and wick features."""
    features = df[["loc", "zone_signal", "wick_ratio"]].fillna(0.0).values
    km = KMeans(n_clusters=n_clusters, n_init=25, random_state=SKLEARN_SEED)
    df_out = df.copy()
    df_out["regime"] = km.fit_predict(features)
    return df_out, km


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    """Build a basic feature matrix for models."""
    cols = ["loc", "zone_signal", "wick_ratio", "regime"]
    return df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_alpha(df: pd.DataFrame, horizon: int, depth: int, lr: float) -> GradientBoostingRegressor | None:
    """Fit a gradient boosting model for expected forward return."""
    df2 = df.dropna(subset=[f"fwd_{horizon}"])
    if len(df2) < 50:
        return None

    X = build_X(df2)
    y = df2[f"fwd_{horizon}"]

    split = int(len(X) * 0.7)
    if split <= 0 or split >= len(X):
        return None

    Xt, Xv = X.iloc[:split], X.iloc[split:]
    yt, _ = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(
        max_depth=depth,
        learning_rate=lr,
        n_estimators=200,
        random_state=SKLEARN_SEED,
    )
    model.fit(Xt, yt)
    return model


def fit_take_profit(df: pd.DataFrame, horizon: int, quantile: float) -> QuantileRegressor | None:
    """Fit a quantile regression model to estimate a take-profit return."""
    df2 = df.dropna(subset=[f"fwd_{horizon}"])
    if len(df2) < 50:
        return None

    X = build_X(df2)
    y = df2[f"fwd_{horizon}"]

    model = QuantileRegressor(quantile=quantile, alpha=0.0)
    model.fit(X, y)
    return model


def kde_forward_returns(df: pd.DataFrame, horizon: int) -> gaussian_kde:
    """Fit a kernel density estimator over observed forward returns."""
    vals = df[f"fwd_{horizon}"].dropna().values
    if len(vals) == 0:
        raise ValueError("No forward returns available for KDE")
    return gaussian_kde(vals)


def validate_alpha_model(
    df: pd.DataFrame,
    model: GradientBoostingRegressor,
    horizon: int,
) -> Dict[str, float]:
    """Compute simple hold-out validation metrics for the alpha model."""
    df2 = df.dropna(subset=[f"fwd_{horizon}"])
    if df2.empty:
        return {}

    X = build_X(df2)
    y = df2[f"fwd_{horizon}"]

    split = int(len(X) * 0.7)
    if split <= 0 or split >= len(X):
        return {}

    Xt, Xv = X.iloc[:split], X.iloc[split:]
    yt, yv = y.iloc[:split], y.iloc[split:]

    pred = model.predict(Xv)
    errors = yv - pred

    with np.errstate(divide="ignore", invalid="ignore"):
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        if np.std(yv) > 0 and np.std(pred) > 0:
            corr = float(np.corrcoef(yv, pred)[0, 1])
        else:
            corr = float("nan")

    sign_y = np.sign(yv)
    sign_pred = np.sign(pred)
    directional_accuracy = float(np.mean(sign_y == sign_pred))

    metrics: Dict[str, float] = {
        "val_mse": mse,
        "val_mae": mae,
        "val_corr": corr,
        "val_directional_accuracy": directional_accuracy,
    }

    LOG.info(
        "Alpha model validation: MSE=%.6e MAE=%.6e corr=%.4f dir_acc=%.2f%%",
        metrics["val_mse"],
        metrics["val_mae"],
        metrics["val_corr"],
        metrics["val_directional_accuracy"] * 100.0,
    )

    return metrics


def run_full_pipeline(df_bars: pd.DataFrame) -> Tuple[pd.DataFrame, AlphaArtifacts, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run the full alpha pipeline on bar data and return artifacts."""
    price_bins, buyer, seller = build_zones(df_bars, ZONE_BINS, ZONE_SIGMA)
    df = assign_density(df_bars, price_bins, buyer, seller)
    df, _ = cluster_regimes(df, REGIME_CLUSTERS)
    df = add_forward(df, [FORWARD_HORIZON])

    gb_model = fit_alpha(df, FORWARD_HORIZON, GB_MAX_DEPTH, GB_LEARNING_RATE)
    tp_model = fit_take_profit(df, FORWARD_HORIZON, TP_QUANTILE)
    kde = kde_forward_returns(df, FORWARD_HORIZON)
    validation_metrics: Dict[str, float] | None = None

    if gb_model is not None:
        validation_metrics = validate_alpha_model(df, gb_model, FORWARD_HORIZON)

    latest = df.iloc[-1]
    X_latest = build_X(df).iloc[[-1]]

    exp_ret = float(gb_model.predict(X_latest)[0]) if gb_model is not None else 0.0
    tp_ret = float(tp_model.predict(X_latest)[0]) if tp_model is not None else 0.0

    if exp_ret > 0:
        direction = "BUY"
    elif exp_ret < 0:
        direction = "SELL"
    else:
        direction = "HOLD"

    tp_price = float(latest["close"] * (1.0 + tp_ret))

    artifacts = AlphaArtifacts(
        direction=direction,
        expected_return=exp_ret,
        tp_return=tp_ret,
        tp_price=tp_price,
        gb_model=gb_model if gb_model is not None else GradientBoostingRegressor(),
        tp_model=tp_model if tp_model is not None else QuantileRegressor(quantile=TP_QUANTILE, alpha=0.0),
        kde=kde,
        validation_metrics=validation_metrics,
    )

    return df, artifacts, (price_bins, buyer, seller)


def plot_price_and_regimes(df: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot close price over time colored by regime."""
    if "regime" not in df.columns:
        raise ValueError("DataFrame must contain 'regime' column for plotting")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    regimes = sorted(df["regime"].unique())
    palette = sns.color_palette("tab10", n_colors=len(regimes))

    for regime, color in zip(regimes, palette):
        mask = df["regime"] == regime
        ax.plot(
            df.loc[mask, "timestamp"],
            df.loc[mask, "close"],
            linestyle="-",
            marker="",
            color=color,
            label=f"Regime {regime}",
        )

    ax.set_title("BTC-USD Close Price by Regime")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    ax.legend(loc="upper left", ncol=2, fontsize="small")
    return ax


def plot_density_zones(price_bins: np.ndarray, buyer: np.ndarray, seller: np.ndarray, ax: Axes | None = None) -> Axes:
    """Plot buyer and seller density profiles over price."""
    centers = (price_bins[:-1] + price_bins[1:]) / 2.0

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(buyer, centers, label="Buyer density", color="green")
    ax.plot(seller, centers, label="Seller density", color="red")
    ax.set_xlabel("Density")
    ax.set_ylabel("Price")
    ax.set_title("Buyer/Seller Density Zones")
    ax.legend()
    return ax


def plot_forward_return_distribution(df: pd.DataFrame, horizon: int, kde: gaussian_kde, ax: Axes | None = None) -> Axes:
    """Plot histogram and KDE of forward returns."""
    col = f"fwd_{horizon}"
    if col not in df.columns:
        raise ValueError(f"Missing forward return column: {col}")

    vals = df[col].dropna().values
    if len(vals) == 0:
        raise ValueError("No forward returns available for plotting")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(vals, bins=50, kde=False, stat="density", color="steelblue", edgecolor="white", ax=ax)

    grid = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(grid, kde(grid), color="darkorange", label="KDE")

    ax.set_title(f"Forward Return Distribution (h={horizon})")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_regime_stats(df: pd.DataFrame, horizon: int, ax: Axes | None = None) -> Axes:
    """Plot per-regime average forward returns with counts."""
    col = f"fwd_{horizon}"
    if col not in df.columns:
        raise ValueError(f"Missing forward return column: {col}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    grouped = df.dropna(subset=[col]).groupby("regime")
    stats = grouped[col].agg(["mean", "std", "count"]).reset_index()

    ax.bar(stats["regime"], stats["mean"], color="#3a7bd5", edgecolor="black", alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)

    for _, row in stats.iterrows():
        ax.text(
            row["regime"],
            row["mean"],
            f"n={int(row['count'])}",
            ha="center",
            va="bottom" if row["mean"] >= 0 else "top",
            fontsize=8,
        )

    ax.set_title(f"Per-Regime Average Forward Return (h={horizon})")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean forward return")
    return ax


def plot_dashboard(
    df: pd.DataFrame,
    price_bins: np.ndarray,
    buyer: np.ndarray,
    seller: np.ndarray,
    horizon: int,
    kde: gaussian_kde,
) -> None:
    """Render a multi-panel dashboard for the alpha view."""
    sns.set_theme(style="darkgrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=110)

    plot_price_and_regimes(df, ax=axes[0, 0])
    plot_density_zones(price_bins, buyer, seller, ax=axes[0, 1])
    plot_forward_return_distribution(df, horizon, kde, ax=axes[1, 0])
    plot_regime_stats(df, horizon, ax=axes[1, 1])

    fig.suptitle("Coinbase BTC-USD Density/Regime Alpha Playground", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def run_poc() -> None:
    """Run the full PoC on Coinbase BTC candles and show plots."""
    configure_determinism()

    LOG.info("Loading Coinbase OHLC candles from %s", CSV_PATH)
    bars = load_coinbase_ohlc(MAX_ROWS)
    LOG.info("Loaded %d OHLC bars", len(bars))

    df, artifacts, zone_tuple = run_full_pipeline(bars)

    LOG.info(
        "Latest signal: dir=%s exp_ret=%.5f tp_ret=%.5f tp_price=%.2f",
        artifacts.direction,
        artifacts.expected_return,
        artifacts.tp_return,
        artifacts.tp_price,
    )

    if artifacts.validation_metrics:
        LOG.info(
            "Validation metrics: %s",
            ", ".join(f"{k}={v:.6f}" for k, v in artifacts.validation_metrics.items()),
        )

    price_bins, buyer, seller = zone_tuple
    plot_dashboard(df, price_bins, buyer, seller, FORWARD_HORIZON, artifacts.kde)
    plt.show()


if __name__ == "__main__":
    run_poc()
