from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GRB006_SESSION = "20240821_121447"
GRB006_TRIAL_TS_PATH = Path(
    "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl"
)
GRB006_SPIKE_TIMES_PATHS = [
    Path(
        "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/"
        "20240821_121447_ks4_spike_times.pkl"
    ),
    Path("/Users/gabriel/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl"),
]


def resolve_grb006_spike_times_path() -> Path:
    for path in GRB006_SPIKE_TIMES_PATHS:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in GRB006_SPIKE_TIMES_PATHS)
    raise FileNotFoundError(
        f"Could not find GRB006 KS4 spike-time export in:\n{searched}"
    )


def load_local_spike_times(
    spike_times_path: Path, sampling_rate: float = 30000.0
) -> tuple[list[int], list[np.ndarray]]:
    spike_df = pd.read_pickle(spike_times_path)
    unit_ids = spike_df["unit_id"].astype(int).tolist()
    spike_times = [
        np.asarray(times, dtype=float) / sampling_rate
        for times in spike_df["spike_times"].tolist()
    ]
    return unit_ids, spike_times


def load_grb006_first_stim(trial_ts_path: Path = GRB006_TRIAL_TS_PATH) -> np.ndarray:
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True)
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    return first_stim[np.isfinite(first_stim)]


def baseline_mean(
    peth_trials: np.ndarray,
    bin_centers: np.ndarray,
    baseline_window: tuple[float, float],
) -> float:
    mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    return float(peth_trials.mean(axis=0)[mask].mean())


def plot_mean_sem_trace(
    ax,
    bin_centers: np.ndarray,
    peth_trials: np.ndarray,
    color: str,
    label: str | None = None,
) -> None:
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def mark_peaks(ax, peak_row, color: str, marker: str = "v", markersize: float = 7):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, marker, color=color, markersize=markersize, zorder=5)
