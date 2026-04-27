from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GRB006_SUBJECT = "GRB006"
GRB006_SESSION = "20240821_121447"
GRB006_TRIAL_TS_PATH = Path(
    "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl"
)
GRB006_TRIAL_TS_FALLBACK_PATH = Path(
    "/Users/gabriel/Downloads/Organized/Code/trial_ts.pkl"
)


def resolve_grb006_trial_ts_path() -> Path:
    for path in (GRB006_TRIAL_TS_PATH, GRB006_TRIAL_TS_FALLBACK_PATH):
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find GRB006 trial_ts.pkl in:\n"
        f"{GRB006_TRIAL_TS_PATH}\n{GRB006_TRIAL_TS_FALLBACK_PATH}"
    )


def load_grb006_first_stim(trial_ts_path: Path | None = None) -> np.ndarray:
    trial_ts_path = trial_ts_path or resolve_grb006_trial_ts_path()
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True)
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    return first_stim[np.isfinite(first_stim)]


def fetch_db_spike_times(
    subject: str, session: str, unit_criteria_id: int = 1
) -> tuple[list[int], list[np.ndarray]]:
    from ephys.src.utils.utils_IO import fetch_good_units

    st_per_unit = fetch_good_units(subject, session, unit_criteria_id)
    return list(st_per_unit.keys()), list(st_per_unit.values())


def fetch_grb006_db_spike_times(
    unit_criteria_id: int = 1,
) -> tuple[list[int], list[np.ndarray]]:
    return fetch_db_spike_times(GRB006_SUBJECT, GRB006_SESSION, unit_criteria_id)


def enrich_trial_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    trial_df = trial_df.reset_index(drop=True).copy()
    trial_df["prev_response"] = trial_df["response"].shift(1)
    trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
    trial_df["prev_stim_rate"] = trial_df["stim_rate_vision"].shift(1)
    return trial_df


def derive_local_trial_signature(local_row: pd.Series) -> tuple[int | None, int, int]:
    rate = local_row.get("trial_rate")
    rate_key = int(rate) if np.isfinite(rate) else None
    outcome = int(local_row.get("trial_outcome"))
    side = local_row.get("response_side")
    if np.isfinite(side):
        choice = 1 if int(side) == 1 else -1
    else:
        choice = 0
    return rate_key, choice, outcome


def derive_full_trial_signature(full_row: pd.Series) -> tuple[int | None, int, int]:
    rate = full_row.get("stim_rate_vision")
    rate_key = int(rate) if np.isfinite(rate) else None
    choice = int(full_row.get("response", 0))
    if full_row.get("rewarded", 0) == 1:
        outcome = 1
    elif full_row.get("with_choice", 0) == 1:
        outcome = 0
    else:
        outcome = 2
    return rate_key, choice, outcome


def align_local_trials_to_full_trial_df(
    local_trial_ts: pd.DataFrame, full_trial_df: pd.DataFrame
) -> np.ndarray:
    matched_idx = []
    start = 0
    full_signatures = [
        derive_full_trial_signature(row) for _, row in full_trial_df.iterrows()
    ]
    for _, local_row in local_trial_ts.iterrows():
        target = derive_local_trial_signature(local_row)
        found = None
        for idx in range(start, len(full_signatures)):
            if full_signatures[idx] == target:
                found = idx
                break
        if found is None:
            relaxed = (target[0], target[2])
            for idx in range(start, len(full_signatures)):
                probe = full_signatures[idx]
                if (probe[0], probe[2]) == relaxed:
                    found = idx
                    break
        if found is None:
            raise RuntimeError(
                "Could not align local paired trial rows to Chipmunk trial rows "
                f"for {target} starting at full trial index {start}."
            )
        matched_idx.append(found)
        start = found + 1
    return np.asarray(matched_idx, dtype=int)


def fetch_chipmunk_session_trials(subject: str, session: str) -> pd.DataFrame:
    from labdata.schema import DecisionTask  # noqa: F401
    from chipmunk import Chipmunk  # type: ignore

    restriction = f"subject_name = '{subject}' AND session_name = '{session}'"
    trial_df = pd.DataFrame(
        (Chipmunk * Chipmunk.Trial * Chipmunk.TrialParameters & restriction).fetch(
            order_by="trial_num"
        )
    )
    if trial_df.empty:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}")
    return enrich_trial_df(trial_df)


def load_grb006_aligned_trial_data(
    trial_ts_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_ts_path = trial_ts_path or resolve_grb006_trial_ts_path()
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True).copy()
    trial_df = fetch_chipmunk_session_trials(GRB006_SUBJECT, GRB006_SESSION)
    trial_ts["trial_idx"] = align_local_trials_to_full_trial_df(trial_ts, trial_df)
    return trial_df, trial_ts


def load_grb006_hybrid_session_inputs(
    unit_criteria_id: int = 1,
    trial_ts_path: Path | None = None,
) -> tuple[list[int], list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    trial_df, trial_ts = load_grb006_aligned_trial_data(trial_ts_path)
    unit_ids, spike_times = fetch_grb006_db_spike_times(unit_criteria_id)
    return unit_ids, spike_times, trial_df, trial_ts


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
