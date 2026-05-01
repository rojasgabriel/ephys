"""Continuous-spike Rastermap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

DEFAULT_BIN_MS = 100.0
MIN_UNITS_FOR_RASTERMAP = 10
DEFAULT_N_PCS = 200
DEFAULT_LOCALITY = 0.75
DEFAULT_TIME_LAG_WINDOW = 5
DEFAULT_MAX_CLUSTERS = 100


@dataclass(frozen=True)
class ContinuousSpikeMatrix:
    unit_ids: np.ndarray
    bin_edges_s: np.ndarray
    spike_counts: np.ndarray
    trial_idx_by_bin: np.ndarray | None = None
    absolute_bin_start_s: np.ndarray | None = None
    absolute_bin_stop_s: np.ndarray | None = None


@dataclass(frozen=True)
class RastermapResult:
    subject: str
    session: str
    unit_ids: np.ndarray
    depth: np.ndarray
    bin_edges_s: np.ndarray
    spike_counts: np.ndarray
    trial_idx_by_bin: np.ndarray | None
    absolute_bin_start_s: np.ndarray | None
    absolute_bin_stop_s: np.ndarray | None
    isort: np.ndarray
    embedding: np.ndarray
    x_embedding: np.ndarray | None
    n_clusters: int


def build_continuous_spike_count_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    *,
    bin_ms: float = DEFAULT_BIN_MS,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
) -> ContinuousSpikeMatrix:
    """Bin unit spike times into a neurons-by-time count matrix."""
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive.")
    if not spike_times_by_unit:
        raise ValueError("No units were provided.")

    bin_s = bin_ms / 1000.0
    unit_ids = np.asarray(list(spike_times_by_unit.keys()), dtype=int)
    spike_arrays = [
        np.asarray(spike_times_by_unit[unit_id], dtype=float) for unit_id in unit_ids
    ]
    finite_spike_arrays = [
        spikes[np.isfinite(spikes)] for spikes in spike_arrays if spikes.size
    ]

    if t_stop_s is None:
        max_spike_times = [
            float(spikes.max()) for spikes in finite_spike_arrays if spikes.size
        ]
        if not max_spike_times:
            raise ValueError("No finite spike times were provided.")
        t_stop_s = max(max_spike_times)

    if t_stop_s <= t_start_s:
        raise ValueError("t_stop_s must be greater than t_start_s.")

    n_bins = int(np.ceil((t_stop_s - t_start_s) / bin_s))
    bin_edges_s = t_start_s + np.arange(n_bins + 1, dtype=float) * bin_s
    if bin_edges_s[-1] < t_stop_s:
        bin_edges_s = np.append(bin_edges_s, bin_edges_s[-1] + bin_s)

    spike_counts = np.zeros((len(unit_ids), len(bin_edges_s) - 1), dtype=np.float32)
    for unit_index, spikes in enumerate(spike_arrays):
        finite_spikes = spikes[np.isfinite(spikes)]
        spike_counts[unit_index], _ = np.histogram(finite_spikes, bins=bin_edges_s)

    return ContinuousSpikeMatrix(
        unit_ids=unit_ids,
        bin_edges_s=bin_edges_s,
        spike_counts=spike_counts,
    )


def trial_windows_from_metadata(trial_df) -> np.ndarray:
    """Return OBX-clock windows from initiation to response/feedback."""
    required_columns = ["t_sync", "t_initiate", "t_response", "trial_start_ts"]
    missing_columns = [
        column for column in required_columns if column not in trial_df.columns
    ]
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {missing_columns}")

    bpod_sync = trial_df["t_sync"].to_numpy(dtype=float)
    obx_sync = trial_df["trial_start_ts"].to_numpy(dtype=float)
    valid_sync = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    if valid_sync.sum() < 2:
        raise ValueError("At least two valid sync points are required.")

    t_initiate = trial_df["t_initiate"].to_numpy(dtype=float)
    t_response = trial_df["t_response"].to_numpy(dtype=float)
    finite_window = np.isfinite(t_initiate) & np.isfinite(t_response)
    start_s = np.interp(
        t_initiate[finite_window], bpod_sync[valid_sync], obx_sync[valid_sync]
    )
    stop_s = np.interp(
        t_response[finite_window], bpod_sync[valid_sync], obx_sync[valid_sync]
    )
    valid_window = stop_s > start_s
    return np.column_stack([start_s[valid_window], stop_s[valid_window]])


def build_trial_window_spike_count_matrix(
    spike_times_by_unit: Mapping[int, np.ndarray],
    trial_windows_s: np.ndarray,
    *,
    bin_ms: float = DEFAULT_BIN_MS,
) -> ContinuousSpikeMatrix:
    """Bin spikes inside trial windows and concatenate windows in time."""
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive.")
    if not spike_times_by_unit:
        raise ValueError("No units were provided.")
    trial_windows_s = np.asarray(trial_windows_s, dtype=float)
    if trial_windows_s.ndim != 2 or trial_windows_s.shape[1] != 2:
        raise ValueError("trial_windows_s must be an array with shape (n_trials, 2).")
    finite_windows = trial_windows_s[
        np.isfinite(trial_windows_s).all(axis=1)
        & (trial_windows_s[:, 1] > trial_windows_s[:, 0])
    ]
    if finite_windows.size == 0:
        raise ValueError("No valid trial windows were provided.")

    bin_s = bin_ms / 1000.0
    unit_ids = np.asarray(list(spike_times_by_unit.keys()), dtype=int)
    spike_arrays = [
        np.asarray(spike_times_by_unit[unit_id], dtype=float) for unit_id in unit_ids
    ]

    counts_by_trial = []
    trial_idx_by_bin = []
    absolute_bin_start_s = []
    absolute_bin_stop_s = []
    elapsed_edges = [0.0]
    elapsed_s = 0.0
    for trial_index, (start_s, stop_s) in enumerate(finite_windows):
        duration_s = stop_s - start_s
        n_bins = int(np.ceil(duration_s / bin_s))
        relative_edges = np.arange(n_bins + 1, dtype=float) * bin_s
        relative_edges[-1] = duration_s
        absolute_edges = start_s + relative_edges
        trial_counts = np.zeros((len(unit_ids), n_bins), dtype=np.float32)
        for unit_index, spikes in enumerate(spike_arrays):
            finite_spikes = spikes[np.isfinite(spikes)]
            trial_counts[unit_index], _ = np.histogram(
                finite_spikes, bins=absolute_edges
            )
        counts_by_trial.append(trial_counts)
        trial_idx_by_bin.extend([trial_index] * n_bins)
        absolute_bin_start_s.extend(absolute_edges[:-1])
        absolute_bin_stop_s.extend(absolute_edges[1:])
        elapsed_s += duration_s
        elapsed_edges.extend((elapsed_s - duration_s) + relative_edges[1:])

    return ContinuousSpikeMatrix(
        unit_ids=unit_ids,
        bin_edges_s=np.asarray(elapsed_edges, dtype=float),
        spike_counts=np.concatenate(counts_by_trial, axis=1),
        trial_idx_by_bin=np.asarray(trial_idx_by_bin, dtype=int),
        absolute_bin_start_s=np.asarray(absolute_bin_start_s, dtype=float),
        absolute_bin_stop_s=np.asarray(absolute_bin_stop_s, dtype=float),
    )


def choose_rastermap_cluster_count(
    n_units: int,
    *,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> int:
    if n_units < MIN_UNITS_FOR_RASTERMAP:
        raise ValueError(
            f"Rastermap needs at least {MIN_UNITS_FOR_RASTERMAP} units; got {n_units}."
        )
    if n_units < max_clusters:
        return max(2, n_units - 1)
    return max_clusters


def fit_rastermap(
    spike_counts: np.ndarray,
    *,
    rastermap_cls=None,
    n_pcs: int = DEFAULT_N_PCS,
    locality: float = DEFAULT_LOCALITY,
    time_lag_window: int = DEFAULT_TIME_LAG_WINDOW,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
):
    """Fit Rastermap on a neurons-by-time matrix."""
    if spike_counts.ndim != 2:
        raise ValueError("spike_counts must be a 2D neurons-by-time matrix.")
    n_units = spike_counts.shape[0]
    n_clusters = choose_rastermap_cluster_count(n_units, max_clusters=max_clusters)
    if rastermap_cls is None:
        from rastermap import Rastermap

        rastermap_cls = Rastermap
    model = rastermap_cls(
        n_PCs=n_pcs,
        n_clusters=n_clusters,
        locality=locality,
        time_lag_window=time_lag_window,
    ).fit(spike_counts.astype("float32", copy=False))
    return model, n_clusters


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    return np.divide(
        values - means,
        stds,
        out=np.zeros_like(values, dtype=float),
        where=stds > 0,
    )


def fit_session_continuous_rastermap(
    subject: str,
    session: str,
    *,
    bin_ms: float = DEFAULT_BIN_MS,
    trial_window_only: bool = True,
    rastermap_cls=None,
) -> RastermapResult:
    from ephys.src.utils.io_session_units import fetch_good_units_with_depth

    spike_times_by_unit, depth_by_unit = fetch_good_units_with_depth(subject, session)
    if trial_window_only:
        from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata
        from ephys.src.utils.io_digital_events import fetch_session_events

        align_ev = fetch_session_events(subject, session)
        trial_df = fetch_trial_metadata(subject, session, align_ev)
        if trial_df is None:
            raise RuntimeError(
                f"Could not load trial metadata for {subject} {session}."
            )
        trial_windows_s = trial_windows_from_metadata(trial_df)
        matrix = build_trial_window_spike_count_matrix(
            spike_times_by_unit,
            trial_windows_s,
            bin_ms=bin_ms,
        )
    else:
        matrix = build_continuous_spike_count_matrix(spike_times_by_unit, bin_ms=bin_ms)
    model, n_clusters = fit_rastermap(matrix.spike_counts, rastermap_cls=rastermap_cls)
    isort = np.asarray(model.isort, dtype=int)
    embedding = np.asarray(model.embedding)
    x_embedding = getattr(model, "X_embedding", None)
    if x_embedding is not None:
        x_embedding = np.asarray(x_embedding)
    depth = np.asarray(
        [depth_by_unit[int(unit_id)] for unit_id in matrix.unit_ids],
        dtype=float,
    )
    return RastermapResult(
        subject=subject,
        session=session,
        unit_ids=matrix.unit_ids,
        depth=depth,
        bin_edges_s=matrix.bin_edges_s,
        spike_counts=matrix.spike_counts,
        trial_idx_by_bin=matrix.trial_idx_by_bin,
        absolute_bin_start_s=matrix.absolute_bin_start_s,
        absolute_bin_stop_s=matrix.absolute_bin_stop_s,
        isort=isort,
        embedding=embedding,
        x_embedding=x_embedding,
        n_clusters=n_clusters,
    )


def heatmap_for_result(result: RastermapResult) -> np.ndarray:
    if result.x_embedding is not None and result.x_embedding.size:
        return np.asarray(result.x_embedding, dtype=float)
    return zscore_rows(result.spike_counts)[result.isort]
