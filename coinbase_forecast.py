"""Forecast price movement on Coinbase candles using a small PyTorch sequence model."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_FILE = Path("coinbase_candles.csv")
TIME_COLUMN = "time"
TARGET_COLUMN = "close"
WINDOW_SIZE = 24
PREDICTION_HORIZON = 48
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEED = 42
LOG_LEVEL = "DEBUG"
PLOT_FILE = Path("coinbase_forecast.png")
FIGURE_DPI = 140
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Append structured key-value context to every log message."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.setdefault("extra", {})
        structure = extra.pop("message_log", "")
        if structure:
            msg = f"{msg} | {structure}"
        return msg, kwargs


def configure_logger() -> logging.Logger:
    """Prepare a colorlog logger that exposes multiple verbosity levels."""

    # Build a colorized handler to ensure structured multi-level output.
    logger = logging.getLogger("coinbase_forecast")
    if logger.handlers:
        return logger
    handler = colorlog.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s%(reset)s %(name)s %(message)s"
    )
    handler.setFormatter(color_formatter)
    logger.addHandler(handler)
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger = StructuredLoggerAdapter(logger, {})
    return logger


def set_global_seed(seed: int) -> None:
    """Enforce deterministic behavior across Python, NumPy, and PyTorch RNGs."""

    # Publish the seed to each random generator so every run mirrors the previous one.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_price_series(path: Path) -> tuple[pd.Series, np.ndarray]:
    """Load Coinbase candle data, ensure monotonic timestamps, and return normalized arrays."""

    # Accept raw CSV input and align samples by increasing timestamps for sequential modeling.
    dataframe = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    dataframe = dataframe.sort_values(by=TIME_COLUMN, inplace=False)
    times = dataframe[TIME_COLUMN].reset_index(drop=True)
    if hasattr(times.dt, "tz") and times.dt.tz is not None:
        times = times.dt.tz_convert(None)
    prices = dataframe[TARGET_COLUMN].astype("float64").to_numpy()
    return times, prices


def normalize_series(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize the price series to zero mean and unit variance for model stability."""

    # Use simple z-score normalization that can be inverted for final reporting.
    mean_value = float(np.mean(values))
    std_value = float(np.std(values)) or 1.0
    normalized_values = (values - mean_value) / std_value
    return normalized_values, mean_value, std_value


class SequenceDataset(Dataset):
    """Slide a fixed length window through normalized prices for supervised learning."""

    def __init__(self, series: np.ndarray, window: int, horizon: int, offset: int = 0) -> None:
        # Validate that the provided series is long enough for the requested configuration.
        if len(series) < window + horizon:
            raise ValueError("Series is too short for the requested window and horizon.")
        self.series = series
        self.window = window
        self.horizon = horizon
        self.offset = offset
        self.length = len(series) - window - horizon + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        start = index
        feature_window = self.series[start : start + self.window]
        target_index = start + self.window + self.horizon - 1
        target_value = self.series[target_index]
        # Convert raw numpy arrays to tensors before returning them to the dataloader.
        feature_tensor = torch.from_numpy(feature_window.astype("float32"))
        target_tensor = torch.tensor(target_value, dtype=torch.float32)
        return feature_tensor, target_tensor, index

    def target_position(self, sample_index: int) -> int:
        """Translate a sample index into the global position of its target value."""

        # Compute the global index that aligns with the sliding window origin.
        return self.offset + sample_index + self.window + self.horizon - 1


class ForecastNet(nn.Module):
    """Simple GRU-based encoder with a small MLP head to regress the next price."""

    def __init__(self, feature_size: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # Process the input through the GRU layers to capture temporal patterns.
        hidden_output, _ = self.gru(sequence)
        # Extract the final hidden state before the regression head.
        last_hidden = hidden_output[:, -1, :]
        prediction = self.head(last_hidden)
        return prediction


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    logger: logging.Logger,
) -> None:
    """Run a single epoch of training while logging the aggregated loss."""

    # Perform gradient updates for every batch while capturing the loss history.
    model.train()
    epoch_losses: list[float] = []
    for features, targets, _ in dataloader:
        optimizer.zero_grad()
        features = features.unsqueeze(-1).to(DEVICE)
        targets = targets.unsqueeze(-1).to(DEVICE)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))
    average_loss = float(np.mean(epoch_losses))
    logger.info("Completed training epoch", extra={"message_log": f"train_loss={average_loss:.4f}"})


def evaluate_forecast(
    model: nn.Module,
    dataset: SequenceDataset,
    dataloader: DataLoader,
    times: pd.Series,
    mean: float,
    std: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute predictions on the validation set and return aligned metadata for plotting."""

    # Collect predictions and true values while threading timestamps for plotting.
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    timestamps: list[pd.Timestamp] = []
    with torch.no_grad():
        for features, batch_targets, batch_indices in dataloader:
            features = features.unsqueeze(-1).to(DEVICE)
            outputs = model(features)
            for value, target, index_tensor in zip(outputs, batch_targets, batch_indices):
                timestamp = times.iat[dataset.target_position(int(index_tensor))]
                timestamps.append(timestamp)
                predictions.append(float(value.item()) * std + mean)
                targets.append(float(target.item()) * std + mean)
    evaluation_frame = pd.DataFrame(
        {"timestamp": timestamps, "actual": targets, "prediction": predictions}
    )
    residuals = evaluation_frame["actual"] - evaluation_frame["prediction"]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    logger.info("Validation RMSE", extra={"message_log": f"{rmse:.2f}"})
    return evaluation_frame


def plot_forecast(frame: pd.DataFrame, logger: logging.Logger) -> None:
    """Render the prediction trace alongside the actual price and residual distribution."""

    # Use seaborn styling to keep the visualization legible and publication ready.
    sns.set(style="darkgrid")
    # Sort timestamps and strip timezone information to keep axes readable.
    frame = frame.sort_values("timestamp", inplace=False).reset_index(drop=True)
    timestamps = frame["timestamp"]
    if getattr(timestamps.dt, "tz", None) is not None:
        frame["timestamp"] = timestamps.dt.tz_convert(None)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=FIGURE_DPI, sharex=True)
    axes[0].plot(
        frame["timestamp"],
        frame["actual"],
        label="Actual",
        color="#3a7bd5",
        linewidth=1.8,
    )
    axes[0].plot(
        frame["timestamp"],
        frame["prediction"],
        label="Prediction",
        color="#f65058",
        linewidth=1.2,
    )
    axes[0].fill_between(
        frame["timestamp"],
        frame["actual"],
        frame["prediction"],
        color="#f65058",
        alpha=0.08,
        label="Error band",
        where=frame["prediction"].notna(),
    )
    forecast_mask = frame["prediction"].notna()
    if forecast_mask.any():
        forecast_start = frame.loc[forecast_mask, "timestamp"].iloc[0]
        forecast_end = frame["timestamp"].iloc[-1]
        # Highlight the region where the model projects future behavior.
        axes[0].axvspan(
            forecast_start,
            forecast_end,
            color="#6c5ce7",
            alpha=0.1,
            label="Forecast window",
        )
        axes[0].axvline(forecast_start, color="#6c5ce7", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title("Coinbase Candle Forecast")
    # Track residual deviations over time instead of a detached histogram.
    residuals = frame["actual"] - frame["prediction"]
    if forecast_mask.any():
        axes[1].plot(
            frame.loc[forecast_mask, "timestamp"],
            residuals[forecast_mask],
            color="#ffa600",
            linewidth=1.0,
        )
        axes[1].axvspan(
            forecast_start,
            forecast_end,
            color="#6c5ce7",
            alpha=0.1,
        )
        axes[1].axvline(forecast_start, color="#6c5ce7", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.0, color="#2f4858", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Prediction Residuals Over Time")
    fig.tight_layout()
    fig.savefig(PLOT_FILE)
    logger.info("Saved forecast figure", extra={"message_log": f"path={PLOT_FILE}"})


def main() -> None:
    """Execute the complete training, evaluation, and visualization pipeline."""

    logger = configure_logger()
    set_global_seed(SEED)
    logger.info("Loading Coinbase data", extra={"message_log": f"path={DATA_FILE}"})
    times, prices = load_price_series(DATA_FILE)
    start_time = times.iat[0] if not times.empty else "n/a"
    end_time = times.iat[len(times) - 1] if not times.empty else "n/a"
    logger.debug(
        "Loaded time series",
        extra={"message_log": (f"samples={len(prices)} start={start_time} end={end_time}")},
    )
    normalized_prices, mean_price, std_price = normalize_series(prices)
    logger.debug(
        "Normalization parameters",
        extra={"message_log": f"mean={mean_price:.2f} std={std_price:.2f}"},
    )
    # Split the normalized series deterministically into training and validation segments.
    train_cut = int(len(normalized_prices) * TRAIN_RATIO)
    training_series = normalized_prices[:train_cut]
    validation_series = normalized_prices[train_cut - WINDOW_SIZE :]
    train_dataset = SequenceDataset(training_series, WINDOW_SIZE, PREDICTION_HORIZON, offset=0)
    validation_offset = train_cut - WINDOW_SIZE
    val_dataset = SequenceDataset(
        validation_series, WINDOW_SIZE, PREDICTION_HORIZON, offset=validation_offset
    )
    logger.debug(
        "Dataset configuration",
        extra={
            "message_log": (
                f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
                f"window={WINDOW_SIZE} horizon={PREDICTION_HORIZON}"
            )
        },
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.debug(
        "DataLoader configuration",
        extra={
            "message_log": (
                f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
                f"batch_size={BATCH_SIZE}"
            )
        },
    )
    logger.info("Building model", extra={"message_log": f"device={DEVICE}"})
    model = ForecastNet(feature_size=1, hidden_size=HIDDEN_SIZE, layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    # Iterate over epochs to gradually minimize the forecasting error.
    for epoch in range(1, EPOCHS + 1):
        logger.info("Epoch start", extra={"message_log": f"{epoch=}"})
        train_model(model, train_loader, optimizer, criterion, logger)
    forecast_frame = evaluate_forecast(
        model, val_dataset, val_loader, times, mean_price, std_price, logger
    )
    plot_frame = pd.DataFrame({"timestamp": times, "actual": prices})
    prediction_frame = forecast_frame[["timestamp", "prediction"]]
    plot_frame = plot_frame.merge(prediction_frame, on="timestamp", how="left")
    plot_forecast(plot_frame, logger)


if __name__ == "__main__":
    main()
