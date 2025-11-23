"""Feature-rich dashboard + modeling toolkit for EU LNG data.

This script builds an interactive Plotly dashboard that combines:
    • Robust feature engineering (including candlestick-inspired metrics)
    • STL seasonality detection with explicit start/end windows
    • Prophet-based forecasts with confidence bands
    • CatBoost regression for feature importance insight

Running the script will write:
    1. HTML dashboard  -> eu_lng_forecast_dashboard.html
    2. Feature table   -> eu_lng_feature_matrix.csv
    3. Season windows  -> eu_lng_seasonality_windows.csv
    4. CatBoost preds  -> eu_lng_catboost_predictions.csv
    5. Metrics JSON    -> eu_lng_dashboard_metrics.json
"""

from __future__ import annotations

import argparse
import calendar
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.seasonal import STL

DATA_PATH = Path("eu_lng_snapshot.csv")
OUTPUT_HTML = Path("eu_lng_forecast_dashboard.html")
FEATURE_CSV = Path("eu_lng_feature_matrix.csv")
SEASONALITY_CSV = Path("eu_lng_seasonality_windows.csv")
CATBOOST_CSV = Path("eu_lng_catboost_predictions.csv")
METRICS_JSON = Path("eu_lng_dashboard_metrics.json")


@dataclass
class SeasonalityResult:
    seasonal: pd.Series
    trend: pd.Series
    resid: pd.Series
    segments: pd.DataFrame


@dataclass
class ProphetResult:
    forecast: pd.DataFrame
    model: Prophet


@dataclass
class CatBoostResult:
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metrics: dict[str, np.float64]
    model: CatBoostRegressor
    best_params: dict[str, np.float64] | None = None


def load_lng_data(path: Path) -> pd.DataFrame:
    """Load ALSI LNG snapshot data and enforce schema."""

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("lng", "gwh", "sendOut"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["gwh"])
    return df


def derive_ohlc(series: pd.Series, window: int = 5) -> tuple[pd.Series, ...]:
    """Approximate OHLC data from a single time series for candlestick features."""

    close = series
    open_ = series.shift(1)
    rolling = series.rolling(window=window, min_periods=1)
    high = rolling.max()
    low = rolling.min()
    return open_, high, low, close


def add_candlestick_features(
    df: pd.DataFrame, trend_lookback: int = 7, window: int = 5
) -> pd.DataFrame:
    """Add richer candlestick / volatility features derived from the GWh series."""

    open_, high, low, close = derive_ohlc(df["gwh"], window=window)
    open_filled = open_.fillna(close)
    rng = (high - low).replace(0, np.nan)
    eps = 1e-9

    df["cs_open"] = open_filled
    df["cs_high"] = high
    df["cs_low"] = low
    df["cs_close"] = close

    body = close - open_filled
    df["body_size"] = body / (rng + eps)
    df["body_abs"] = body.abs()
    df["close_position"] = (close - low) / (rng + eps)
    df["trend_momentum"] = (close - close.shift(trend_lookback)) / (rng + eps)
    df["breakout_high"] = (high - high.shift(1)) / (rng + eps)
    df["breakout_low_inv"] = -(low - low.shift(1)) / (rng + eps)

    upper_shadow = (high - np.maximum(close, open_filled)).clip(lower=0)
    lower_shadow = (np.minimum(close, open_filled) - low).clip(lower=0)
    df["upper_wick"] = upper_shadow / (rng + eps)
    df["lower_wick"] = lower_shadow / (rng + eps)
    df["wick_polarity"] = np.tanh(df["lower_wick"] - df["upper_wick"])
    df["wick_ratio"] = (df["lower_wick"] - df["upper_wick"]) / (
        df["lower_wick"] + df["upper_wick"] + eps
    )
    df["wick_ratio"] = df["wick_ratio"].clip(-10, 10)
    df["range_percent"] = rng / (open_filled + eps)
    df["body_vs_range"] = df["body_size"] * df["range_percent"]

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    df["true_range"] = true_range
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct_of_close"] = df["atr_14"] / (close.abs() + eps)
    df["range_vs_atr"] = rng / (df["atr_14"] + eps)

    ha_close = (open_filled + high + low + close) / 4
    ha_open = pd.Series(index=df.index, dtype=np.float64)
    if not df.empty:
        ha_open.iloc[0] = (open_filled.iloc[0] + close.iloc[0]) / 2
        for idx in range(1, len(df)):
            ha_open.iloc[idx] = (ha_open.iloc[idx - 1] + ha_close.iloc[idx - 1]) / 2
    df["ha_open"] = ha_open
    df["ha_close"] = ha_close
    df["ha_trend_strength"] = (ha_close - ha_open) / (df["atr_14"] + eps)
    df["ha_body_vs_shadow"] = (ha_close - ha_open) / (upper_shadow + lower_shadow + eps)
    df["candle_direction"] = np.sign(body).fillna(0)
    return df


def _expanding_window_slices(n_samples: int, n_splits: int) -> list[tuple[slice, slice]]:
    """Return expanding-window train/validation slices for time-series CV."""

    if n_samples < 4:
        return []
    n_splits = max(1, n_splits)
    min_train = max(int(n_samples * 0.4), 30)
    min_train = min(min_train, n_samples - 2)
    test_size = max(1, (n_samples - min_train) // n_splits)
    slices: list[tuple[slice, slice]] = []

    train_end = min_train
    while len(slices) < n_splits and train_end + test_size <= n_samples:
        slices.append((slice(0, train_end), slice(train_end, train_end + test_size)))
        train_end += test_size

    if not slices:
        slices.append((slice(0, n_samples - 1), slice(n_samples - 1, n_samples)))
    return slices


def engineer_features(df: pd.DataFrame, *, ohlc_window: int, trend_lookback: int) -> pd.DataFrame:
    """Create modeling features ranging from rolling stats to seasonal encodings."""

    engineered = df.copy()
    engineered["gwh_ma7"] = engineered["gwh"].rolling(7).mean()
    engineered["gwh_ma30"] = engineered["gwh"].rolling(30).mean()
    engineered["gwh_ma90"] = engineered["gwh"].rolling(90).mean()
    engineered["gwh_std14"] = engineered["gwh"].rolling(14).std()
    engineered["gwh_pct_change"] = engineered["gwh"].pct_change()
    engineered["gwh_diff"] = engineered["gwh"].diff()

    engineered["lng_pct_change"] = engineered["lng"].pct_change()
    engineered["sendout_pct_change"] = engineered["sendOut"].pct_change()
    engineered["sendout_ma7"] = engineered["sendOut"].rolling(7).mean()
    engineered["sendout_ma30"] = engineered["sendOut"].rolling(30).mean()
    engineered["lng_ma7"] = engineered["lng"].rolling(7).mean()
    engineered["lng_ma30"] = engineered["lng"].rolling(30).mean()

    for lag in (1, 3, 7, 14, 30):
        engineered[f"gwh_lag_{lag}"] = engineered["gwh"].shift(lag)
    engineered["sendout_lag_7"] = engineered["sendOut"].shift(7)
    engineered["lng_lag_7"] = engineered["lng"].shift(7)

    day_of_year = engineered["date"].dt.dayofyear.astype(np.float64)
    week_of_year = engineered["date"].dt.isocalendar().week.astype(np.float64)
    engineered["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    engineered["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    engineered["wow_sin"] = np.sin(2 * np.pi * week_of_year / 52.0)
    engineered["wow_cos"] = np.cos(2 * np.pi * week_of_year / 52.0)
    engineered["month"] = engineered["date"].dt.month

    engineered = add_candlestick_features(
        engineered, trend_lookback=trend_lookback, window=ohlc_window
    )
    return engineered


def analyze_seasonality(
    df: pd.DataFrame, period: int, threshold_std: np.float64
) -> SeasonalityResult:
    """Use STL to extract seasonal components and derive explicit date windows."""

    series = df["gwh"].interpolate().to_numpy()
    stl = STL(series, period=period, robust=True).fit()
    seasonal = pd.Series(stl.seasonal, index=df["date"], name="seasonal")
    trend = pd.Series(stl.trend, index=df["date"], name="trend")
    resid = pd.Series(stl.resid, index=df["date"], name="residual")

    threshold = seasonal.std() * threshold_std
    signal = np.where(seasonal > threshold, 1, np.where(seasonal < -threshold, -1, 0))
    segments: list[dict[str, object]] = []
    start_idx = 0
    current_state = signal[0]

    for idx in range(1, len(signal)):
        if signal[idx] != current_state:
            segments.append(_build_segment(df, seasonal, start_idx, idx - 1, int(current_state)))
            start_idx = idx
            current_state = signal[idx]
    segments.append(_build_segment(df, seasonal, start_idx, len(signal) - 1, int(current_state)))

    segment_df = pd.DataFrame([seg for seg in segments if seg["state"] != 0])
    return SeasonalityResult(seasonal=seasonal, trend=trend, resid=resid, segments=segment_df)


def _build_segment(
    df: pd.DataFrame, seasonal: pd.Series, start_idx: int, end_idx: int, state: int
) -> dict[str, object]:
    """Helper to translate start/end indices into metadata."""

    start_date = df["date"].iloc[start_idx]
    end_date = df["date"].iloc[end_idx]
    window = seasonal.iloc[start_idx : end_idx + 1]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "state": state,
        "duration_days": int((end_date - start_date).days + 1),
        "max_abs_seasonal": np.float64(window.abs().max()),
    }


def run_prophet_model(df: pd.DataFrame, horizon: int) -> ProphetResult:
    """Train Prophet with LNG send-out as extra regressors and forecast future."""

    prophet_df = df[["date", "gwh", "lng", "sendOut"]].rename(columns={"date": "ds", "gwh": "y"})
    prophet_df = prophet_df.dropna(subset=["y"]).reset_index(drop=True)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for reg in ("lng", "sendOut"):
        model.add_regressor(reg)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    future = future.merge(prophet_df[["ds", "lng", "sendOut"]], on="ds", how="left")
    future["lng"] = future["lng"].ffill().bfill()
    future["sendOut"] = future["sendOut"].ffill().bfill()
    forecast = model.predict(future)
    return ProphetResult(forecast=forecast, model=model)


def optimize_catboost_hyperparams(
    train_df: pd.DataFrame, feature_cols: Sequence[str], n_trials: int, n_splits: int
) -> tuple[dict[str, np.float64], np.float64 | None]:
    """Tune CatBoost hyperparameters with Optuna using expanding-window CV."""

    if n_trials <= 0 or len(train_df) < 60:
        return {}, None

    train_df = train_df.copy()
    features = train_df[list(feature_cols)]
    target = train_df["gwh"]
    slices = _expanding_window_slices(len(train_df), n_splits)
    if not slices:
        return {}, None

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> np.float64:
        params = {
            "iterations": trial.suggest_int("iterations", 400, 1200),
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "loss_function": "RMSE",
            "bootstrap_type": "Bernoulli",
            "random_seed": 42,
            "verbose": False,
        }
        fold_losses: list[np.float64] = []
        for train_slice, valid_slice in slices:
            X_train = features.iloc[train_slice]
            y_train = target.iloc[train_slice]
            X_valid = features.iloc[valid_slice]
            y_valid = target.iloc[valid_slice]
            if X_train.empty or X_valid.empty:
                continue
            model = CatBoostRegressor(**params)
            try:
                model.fit(X_train, y_train)
            except Exception:
                return np.float64("inf")
            preds = model.predict(X_valid)
            rmse = np.float64(np.sqrt(np.mean((y_valid - preds) ** 2)))
            if not np.isfinite(rmse):
                return np.float64("inf")
            fold_losses.append(rmse)
        if not fold_losses:
            return np.float64("inf")
        return np.float64(np.mean(fold_losses))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_trial.params, np.float64(study.best_value)


def train_catboost(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    holdout_days: int,
    *,
    optuna_trials: int = 0,
    optuna_splits: int = 3,
) -> CatBoostResult:
    """Fit CatBoost on engineered features to explain/predict GWh values."""

    df_model = df.dropna(subset=list(feature_cols) + ["gwh"]).copy()
    holdout_days = min(holdout_days, len(df_model) // 3)
    split_point = len(df_model) - holdout_days
    train_df = df_model.iloc[:split_point]
    test_df = df_model.iloc[split_point:]

    tuned_params, cv_rmse = optimize_catboost_hyperparams(
        train_df, feature_cols, optuna_trials, optuna_splits
    )
    base_params: dict[str, object] = {
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
    }
    base_params.update(tuned_params)

    model = CatBoostRegressor(**base_params)
    model.fit(
        train_df[list(feature_cols)],
        train_df["gwh"],
        eval_set=(test_df[list(feature_cols)], test_df["gwh"]),
    )
    test_df = test_df.copy()
    test_df["prediction"] = model.predict(test_df[list(feature_cols)])
    test_df["residual"] = test_df["gwh"] - test_df["prediction"]

    metrics = {
        "mae": np.float64(np.mean(np.abs(test_df["residual"]))),
        "rmse": np.float64(np.sqrt(np.mean(test_df["residual"] ** 2))),
    }
    denom = np.clip(np.abs(test_df["gwh"]), 1e-6, None)
    metrics["mape_pct"] = np.float64(np.mean(np.abs(test_df["residual"] / denom)) * 100)
    metrics["bias"] = np.float64(np.mean(test_df["residual"]))
    if cv_rmse is not None:
        metrics["optuna_cv_rmse"] = cv_rmse

    predictions = (
        test_df[["date", "gwh", "prediction", "residual"]]
        .rename(columns={"gwh": "actual"})
        .reset_index(drop=True)
    )
    predictions["split"] = "holdout"

    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    return CatBoostResult(
        predictions=predictions,
        feature_importance=importances,
        metrics=metrics,
        model=model,
        best_params=tuned_params or None,
    )


def build_dashboard(
    df: pd.DataFrame,
    prophet_res: ProphetResult,
    catboost_res: CatBoostResult,
    seasonality: SeasonalityResult,
) -> go.Figure:
    """Create a 4-row Plotly dashboard mixing forecasts, residuals, and feature insights."""

    prophet_history = get_prophet_history_alignment(df, prophet_res)
    month_profile = seasonality_month_profile(seasonality)
    forecast = prophet_res.forecast
    history_mask = forecast["ds"] <= df["date"].max()
    future_mask = ~history_mask
    forecast_table = build_forecast_table(forecast.loc[future_mask], limit=18)
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        specs=[
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "table"}],
        ],
        subplot_titles=(
            "Actual vs Prophet Forecast vs CatBoost Holdout",
            "STL Seasonality Windows",
            "Residual Stress Test",
            "CatBoost Feature Importance",
            "Seasonality by Month",
            "Forward Forecast Table",
        ),
    )
    actual_trace = go.Scatter(
        x=df["date"],
        y=df["gwh"],
        name="Actual GWh",
        line=dict(color="#00d8ff", width=2),
    )
    fig.add_trace(actual_trace, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=forecast.loc[history_mask, "ds"],
            y=forecast.loc[history_mask, "yhat"],
            name="Prophet Fit",
            line=dict(color="#ff6b9d", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.loc[future_mask, "ds"],
            y=forecast.loc[future_mask, "yhat"],
            name="Prophet Forecast",
            line=dict(color="#ffa600", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.loc[future_mask, "ds"],
            y=forecast.loc[future_mask, "yhat_upper"],
            name="Forecast Upper",
            mode="lines",
            line=dict(color="rgba(255,166,0,0.25)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.loc[future_mask, "ds"],
            y=forecast.loc[future_mask, "yhat_lower"],
            name="Forecast Lower",
            fill="tonexty",
            line=dict(color="rgba(255,166,0,0.25)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=catboost_res.predictions["date"],
            y=catboost_res.predictions["prediction"],
            name="CatBoost Holdout",
            mode="lines+markers",
            line=dict(color="#14ff8c"),
        ),
        row=1,
        col=1,
    )

    # Seasonality subplot
    fig.add_trace(
        go.Scatter(
            x=seasonality.trend.index,
            y=seasonality.trend,
            name="Trend",
            line=dict(color="#00bcd4"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seasonality.seasonal.index,
            y=seasonality.seasonal,
            name="Seasonal",
            line=dict(color="#f45b69"),
        ),
        row=2,
        col=1,
    )

    for _, segment in seasonality.segments.iterrows():
        color = "rgba(0,255,170,0.15)" if segment["state"] == 1 else "rgba(255,99,132,0.15)"
        fig.add_vrect(
            x0=segment["start_date"],
            x1=segment["end_date"],
            fillcolor=color,
            line_width=0,
            layer="below",
            row=2,
            col=1,
        )
        fig.add_vrect(
            x0=segment["start_date"],
            x1=segment["end_date"],
            fillcolor=color,
            line_width=0,
            layer="below",
            row=3,
            col=1,
        )

    # Residual diagnostics subplot
    fig.add_trace(
        go.Scatter(
            x=catboost_res.predictions["date"],
            y=catboost_res.predictions["residual"],
            name="CatBoost Residual",
            mode="lines",
            line=dict(color="#14ff8c"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prophet_history["ds"],
            y=prophet_history["residual"],
            name="Prophet Residual",
            mode="lines",
            line=dict(color="#ffa600", dash="dash"),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="#999999", dash="dot"), row=3, col=1)

    # Feature importance bar chart
    fig.add_trace(
        go.Bar(
            x=catboost_res.feature_importance["importance"],
            y=catboost_res.feature_importance["feature"],
            orientation="h",
            marker=dict(color="#9c6bff"),
            name="Feature Importance",
        ),
        row=4,
        col=1,
    )

    if not month_profile.empty:
        color_scale = month_profile["seasonal_mean"].apply(
            lambda value: "#1de9b6" if value >= 0 else "#ff8a80"
        )
        fig.add_trace(
            go.Bar(
                x=month_profile["month_name"],
                y=month_profile["seasonal_mean"],
                marker=dict(color=color_scale),
                name="Seasonality Mean",
            ),
            row=5,
            col=1,
        )

    if not forecast_table.empty:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Date", "Forecast", "Lower", "Upper"],
                    fill_color="#1f2a44",
                    font=dict(color="white"),
                ),
                cells=dict(
                    values=[
                        [d.strftime("%Y-%m-%d") for d in forecast_table["date"]],
                        [f"{val:,.1f}" for val in forecast_table["yhat"]],
                        [f"{val:,.1f}" for val in forecast_table["yhat_lower"]],
                        [f"{val:,.1f}" for val in forecast_table["yhat_upper"]],
                    ],
                    fill_color="#111727",
                    font=dict(color="#f2f2f2"),
                ),
            ),
            row=6,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=1850,
        title="EU LNG Forecasting & Seasonality Explorer",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#060b17",
        plot_bgcolor="#060b17",
        font=dict(family="IBM Plex Sans, Helvetica", size=12),
    )
    fig.update_yaxes(title_text="GWh", row=1, col=1)
    fig.update_yaxes(title_text="STL Components", row=2, col=1)
    fig.update_yaxes(title_text="Residual (GWh)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Importance", row=4, col=1)
    fig.update_yaxes(title_text="Avg Seasonal Impact", row=5, col=1)
    fig.update_xaxes(title_text="Month", row=5, col=1)
    return fig


def _forecast_summary_lines(df: pd.DataFrame, prophet_res: ProphetResult) -> list[str]:
    """Generate textual insights for the Prophet forecast horizon."""

    last_date = df["date"].max()
    last_actual = df.loc[df["date"] == last_date, "gwh"].iloc[-1]
    forecast = prophet_res.forecast
    future = forecast[forecast["ds"] > last_date]
    if future.empty:
        return ["  No forward-looking Prophet forecast was generated."]

    horizon_days = future["ds"].nunique()
    start_row = future.iloc[0]
    end_row = future.iloc[-1]
    delta = end_row["yhat"] - start_row["yhat"]
    rel_change = delta / (abs(start_row["yhat"]) + 1e-9) * 100
    trend_direction = "rising" if delta >= 0 else "softening"

    next_30 = future.head(min(30, len(future)))
    avg_next_30 = next_30["yhat"].mean()
    best_future = future.loc[future["yhat"].idxmax()]
    worst_future = future.loc[future["yhat"].idxmin()]

    lines = [
        f"  Horizon: {horizon_days} days ahead ({future['ds'].min().date()} -> {future['ds'].max().date()})",
        f"  Trend: {trend_direction} (Δ={delta:,.1f} GWh, {rel_change:+.2f}%)",
        f"  Last actual vs next 30d avg: {last_actual:,.1f} -> {avg_next_30:,.1f} GWh",
        f"  Projected peak: {best_future['ds'].date()} @ {best_future['yhat']:,.1f} GWh",
        f"  Projected trough: {worst_future['ds'].date()} @ {worst_future['yhat']:,.1f} GWh",
    ]
    return lines


def get_prophet_history_alignment(df: pd.DataFrame, prophet_res: ProphetResult) -> pd.DataFrame:
    """Align Prophet in-sample predictions with actuals to compute residuals."""

    history = prophet_res.forecast[prophet_res.forecast["ds"] <= df["date"].max()].copy()
    history = history.merge(df[["date", "gwh"]], left_on="ds", right_on="date", how="left")
    history.rename(columns={"gwh": "actual"}, inplace=True)
    history["residual"] = history["actual"] - history["yhat"]
    return history


def add_prophet_baseline_feature(df: pd.DataFrame, prophet_res: ProphetResult) -> pd.DataFrame:
    """Enrich engineered features with Prophet's in-sample baseline expectation."""

    history = get_prophet_history_alignment(df, prophet_res)
    baseline = (
        history[["ds", "yhat"]]
        .rename(columns={"ds": "date", "yhat": "prophet_baseline"})
        .drop_duplicates(subset="date")
    )
    enriched = df.merge(baseline, on="date", how="left")
    enriched["prophet_baseline"] = enriched["prophet_baseline"].ffill().bfill()
    return enriched


def seasonality_month_profile(seasonality: SeasonalityResult) -> pd.DataFrame:
    """Aggregate STL seasonal component by calendar month for reporting."""

    if seasonality.seasonal.empty:
        return pd.DataFrame(columns=["month", "month_name", "seasonal_mean", "seasonal_std"])
    season_df = pd.DataFrame({"date": seasonality.seasonal.index, "seasonal": seasonality.seasonal})
    season_df["month"] = season_df["date"].dt.month
    summary = (
        season_df.groupby("month")["seasonal"]
        .agg(["mean", "std", "max", "min"])
        .rename(columns={"mean": "seasonal_mean", "std": "seasonal_std"})
        .reset_index()
        .sort_values("month")
    )
    summary["month_name"] = summary["month"].apply(lambda m: calendar.month_abbr[m])
    summary["phase"] = np.where(summary["seasonal_mean"] >= 0, "Positive", "Negative")
    return summary


def build_forecast_table(future: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return a trimmed forecast table for the dashboard footer."""

    if future.empty:
        return pd.DataFrame(columns=["date", "yhat", "yhat_lower", "yhat_upper"])
    table = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(limit).copy()
    table.rename(columns={"ds": "date"}, inplace=True)
    table["date"] = table["date"].dt.date
    return table


def _diagnostics_summary_lines(
    catboost_res: CatBoostResult, prophet_history: pd.DataFrame
) -> list[str]:
    """Summarize residual distributions for both modeling approaches."""

    lines: list[str] = []

    cb = catboost_res.predictions.dropna(subset=["residual"]).copy()
    if not cb.empty:
        cb_abs = cb["residual"].abs()
        worst_idx = cb_abs.idxmax()
        worst_row = cb.loc[worst_idx]
        lines.append(
            f"  CatBoost residual median {cb['residual'].median():+.1f} GWh | 95th abs {np.percentile(cb_abs, 95):.1f}"
        )
        lines.append(
            f"  CatBoost worst miss: {worst_row['date'].date()} ({worst_row['residual']:+.1f} GWh)"
        )

    ph = prophet_history.dropna(subset=["residual"]).copy()
    if not ph.empty:
        ph_abs = ph["residual"].abs()
        worst_idx = ph_abs.idxmax()
        worst_row = ph.loc[worst_idx]
        lines.append(
            f"  Prophet residual median {ph['residual'].median():+.1f} GWh | 95th abs {np.percentile(ph_abs, 95):.1f}"
        )
        lines.append(
            f"  Prophet worst miss: {worst_row['ds'].date()} ({worst_row['residual']:+.1f} GWh)"
        )

    if not lines:
        lines.append("  Residual diagnostics unavailable (insufficient overlap).")
    return lines


def print_summary(
    catboost_res: CatBoostResult,
    seasonality: SeasonalityResult,
    prophet_res: ProphetResult,
    df: pd.DataFrame,
    output_html: Path,
) -> None:
    """Emit console-friendly summary insights."""

    print("\n" + "=" * 80)
    print("EU LNG DASHBOARD SUMMARY")
    print("=" * 80)
    print(f"Dashboard saved to: {output_html.resolve()}")
    print("\nCatBoost Holdout Metrics:")
    for key, value in catboost_res.metrics.items():
        print(f"  {key.upper():<10}: {value:,.4f}")
    if catboost_res.best_params:
        print("\nCatBoost tuned hyperparameters:")
        for param, val in sorted(catboost_res.best_params.items()):
            if isinstance(val, np.float64):
                val_str = f"{val:,.4f}"
            else:
                val_str = str(val)
            print(f"  {param:<20}: {val_str}")
    print("\nDetected Seasonality Windows (state=+1 bullish / -1 bearish):")
    if seasonality.segments.empty:
        print("  No significant seasonal regimes detected with the current threshold.")
    else:
        for _, seg in seasonality.segments.iterrows():
            direction = "Positive" if seg["state"] == 1 else "Negative"
            print(
                f"  {direction:<8} {seg['start_date'].date()} -> {seg['end_date'].date()} "
                f"({int(seg['duration_days'])} days, peak +/-{seg['max_abs_seasonal']:.1f})"
            )
    month_profile = seasonality_month_profile(seasonality)
    print("\nSeasonality Monthly Pulse:")
    if month_profile.empty:
        print("  Not enough data to summarize monthly tendencies.")
    else:
        for _, row in month_profile.iterrows():
            print(
                f"  {row['month_name']:<3} | {row['phase']:<8} mean={row['seasonal_mean']:+.2f} "
                f"(σ={row['seasonal_std']:.2f})"
            )
    print("\nForecast Outlook (Prophet):")
    for line in _forecast_summary_lines(df, prophet_res):
        print(line)
    prophet_history = get_prophet_history_alignment(df, prophet_res)
    print("\nModel Diagnostics:")
    for line in _diagnostics_summary_lines(catboost_res, prophet_history):
        print(line)
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EU LNG forecasting + seasonality dashboard")
    parser.add_argument(
        "--csv-path", type=Path, default=DATA_PATH, help="Input CSV (eu_lng_snapshot.csv)"
    )
    parser.add_argument(
        "--prophet-horizon", type=int, default=120, help="Future days to forecast with Prophet"
    )
    parser.add_argument(
        "--season-period", type=int, default=365, help="STL seasonality period (days)"
    )
    parser.add_argument(
        "--season-threshold",
        type=np.float64,
        default=0.4,
        help="Std-dev multiplier to mark a segment as seasonal",
    )
    parser.add_argument(
        "--ohlc-window", type=int, default=5, help="Rolling window for derived OHLC features"
    )
    parser.add_argument(
        "--trend-lookback", type=int, default=14, help="Lookback for trend_momentum"
    )
    parser.add_argument(
        "--catboost-holdout", type=int, default=120, help="Days reserved for CatBoost validation"
    )
    parser.add_argument(
        "--catboost-optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for CatBoost tuning (0=skip)",
    )
    parser.add_argument(
        "--catboost-optuna-splits",
        type=int,
        default=3,
        help="Expanding-window splits for Optuna CV",
    )
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML, help="Dashboard HTML path")
    parser.add_argument(
        "--feature-csv", type=Path, default=FEATURE_CSV, help="Feature matrix CSV output"
    )
    parser.add_argument(
        "--seasonality-csv", type=Path, default=SEASONALITY_CSV, help="Season windows CSV"
    )
    parser.add_argument(
        "--catboost-csv", type=Path, default=CATBOOST_CSV, help="CatBoost prediction CSV"
    )
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON, help="Metrics JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_lng_data(args.csv_path)
    df_features = engineer_features(
        df, ohlc_window=args.ohlc_window, trend_lookback=args.trend_lookback
    )

    seasonality = analyze_seasonality(df_features, args.season_period, args.season_threshold)
    feature_cols = [
        "gwh_ma7",
        "gwh_ma30",
        "gwh_ma90",
        "gwh_std14",
        "gwh_pct_change",
        "gwh_diff",
        "lng_pct_change",
        "sendout_pct_change",
        "sendout_ma7",
        "sendout_ma30",
        "lng_ma7",
        "lng_ma30",
        "gwh_lag_1",
        "gwh_lag_3",
        "gwh_lag_7",
        "gwh_lag_14",
        "gwh_lag_30",
        "sendout_lag_7",
        "lng_lag_7",
        "doy_sin",
        "doy_cos",
        "wow_sin",
        "wow_cos",
        "body_size",
        "close_position",
        "trend_momentum",
        "breakout_high",
        "breakout_low_inv",
        "wick_polarity",
        "wick_ratio",
        "body_vs_range",
        "body_abs",
        "true_range",
        "atr_14",
        "atr_pct_of_close",
        "range_vs_atr",
        "ha_trend_strength",
        "ha_body_vs_shadow",
        "candle_direction",
        "prophet_baseline",
    ]

    prophet_res = run_prophet_model(df_features, args.prophet_horizon)
    df_features = add_prophet_baseline_feature(df_features, prophet_res)
    catboost_res = train_catboost(
        df_features,
        feature_cols,
        args.catboost_holdout,
        optuna_trials=args.catboost_optuna_trials,
        optuna_splits=args.catboost_optuna_splits,
    )

    fig = build_dashboard(df_features, prophet_res, catboost_res, seasonality)
    fig.write_html(args.output_html)

    df_features.to_csv(args.feature_csv, index=False)
    seasonality.segments.to_csv(args.seasonality_csv, index=False)
    catboost_res.predictions.to_csv(args.catboost_csv, index=False)
    with open(args.metrics_json, "w") as f:
        json.dump(
            {
                "prophet_horizon_days": args.prophet_horizon,
                "catboost_metrics": catboost_res.metrics,
                "seasonality_window_count": len(seasonality.segments),
            },
            f,
            indent=2,
            default=str,
        )
    print_summary(catboost_res, seasonality, prophet_res, df_features, args.output_html)


if __name__ == "__main__":
    main()
