from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ephys.src.utils.trial_alignment import (
    fetch_chipmunk_trial_table,
    map_local_trial_rows_to_chipmunk_trials,
)

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


def load_grb006_aligned_trial_data(
    trial_ts_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_ts_path = trial_ts_path or resolve_grb006_trial_ts_path()
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True).copy()
    trial_df = fetch_chipmunk_trial_table(GRB006_SUBJECT, GRB006_SESSION)
    trial_ts["trial_idx"] = map_local_trial_rows_to_chipmunk_trials(trial_ts, trial_df)
    return trial_df, trial_ts


def load_grb006_hybrid_session_inputs(
    unit_criteria_id: int = 1,
    trial_ts_path: Path | None = None,
) -> tuple[list[int], list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    trial_df, trial_ts = load_grb006_aligned_trial_data(trial_ts_path)
    unit_ids, spike_times = fetch_grb006_db_spike_times(unit_criteria_id)
    return unit_ids, spike_times, trial_df, trial_ts
