from __future__ import annotations

import numpy as np
import pandas as pd

from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table
from ephys.src.utils.utils_analysis import build_trial_stim_classification


def trial_start_from_row(row: pd.Series) -> float:
    if "center_port_entries" in row.index:
        entries = row["center_port_entries"]
        if entries is None or len(entries) == 0:
            return np.nan
        return float(entries[0])
    if "cp_entry" in row.index:
        return float(row["cp_entry"]) if np.isfinite(row["cp_entry"]) else np.nan
    return np.nan


def load_db_behavior(
    subject: str, session: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    from ephys.src.utils.utils_IO import fetch_session_events, fetch_trial_metadata

    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}")

    trial_df = enrich_chipmunk_trial_table(trial_df)
    trial_ts = build_trial_stim_classification(align_ev, trial_df).reset_index(
        drop=True
    )
    first_stim_times = np.asarray(align_ev["first_stim_ev_15ms"], dtype=float)
    return trial_df, trial_ts, first_stim_times
