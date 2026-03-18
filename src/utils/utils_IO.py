from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from labdata.schema import (
    Dataset,
    DatasetEvents,
    SpikeSorting,
    UnitMetrics,
    EphysRecording,
    UnitCount,
)


PortEventDict = Dict[str, Dict[str, np.ndarray]]


def fetch_good_units(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> dict[int, np.ndarray]:
    """Fetch spike times (in seconds) for units passing quality criteria.

    Returns a dict mapping unit_id → spike_times_seconds, sorted by depth.
    """
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    good_unit_ids = (
        sess_query
        * (UnitCount.Unit & f"unit_criteria_id = {unit_criteria_id}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

    good_units = pd.DataFrame(
        ((SpikeSorting.Unit & good_unit_ids) * UnitMetrics).fetch(
            "unit_id", "spike_times", "depth", as_dict=True
        )
    )

    srate = float(
        (EphysRecording.ProbeSetting() & sess_query).fetch("sampling_rate")[0]
    )
    good_units = good_units.sort_values("depth", ascending=True)
    st_per_unit = {
        row["unit_id"]: row["spike_times"] / srate for _, row in good_units.iterrows()
    }
    return st_per_unit


def fetch_session_events(
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    """Fetch digital events for a session and derive stimulus event arrays.

    Returns a dict with keys: stim, trial_start, frames, left_port,
    center_port, right_port, stim_ev, first_stim_ev.
    """
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    dset = (Dataset() & sess_query).proj()
    events = DatasetEvents.Digital() & dset
    events = pd.DataFrame(events.fetch_synced())

    align_ev: dict[str, np.ndarray] = {
        "stim": events.query("event_name == '0'").event_timestamps.values[0],
        "trial_start": events.query("event_name == '2'").event_timestamps.values[0],
        "frames": events.query("event_name == '3'").event_timestamps.values[0],
        "left_port": events.query("event_name == '4'").event_timestamps.values[0],
        "center_port": events.query("event_name == '5'").event_timestamps.values[0],
        "right_port": events.query(
            "event_name == '6' & stream_name == 'obx'"
        ).event_timestamps.values[0],
    }

    stim = np.asarray(align_ev["stim"], dtype=float)

    # 1. Parse noisy toggling edges into continuous pulse bursts
    max_within_pulse_gap_s = 0.020
    if stim.size > 0:
        stim_sorted = np.sort(stim)
        split_idx = np.where(np.diff(stim_sorted) > max_within_pulse_gap_s)[0] + 1
        bursts = np.split(stim_sorted, split_idx)

        onsets = np.array([b[0] for b in bursts])
        durations = np.array([b[-1] - b[0] for b in bursts])
    else:
        onsets = np.array([])
        durations = np.array([])

    # 2. Assign class labels based on pulse duration
    tol_s = 2e-3
    diff_15 = np.abs(durations - 0.015)
    diff_30 = np.abs(durations - 0.030)

    is_15 = diff_15 <= tol_s
    is_30 = diff_30 <= tol_s
    labels = np.where(is_15, "15ms", np.where(is_30, "30ms", "unknown"))

    # 3. Create core event streams
    stim_ev = onsets
    stim_ev_15ms = onsets[labels == "15ms"]
    stim_ev_30ms = onsets[labels == "30ms"]

    def extract_first_events(ev_array, sep_s=1.0):
        if ev_array.size == 0:
            return ev_array
        keep = np.r_[True, np.diff(ev_array) > sep_s]
        return ev_array[keep]

    first_stim_ev = extract_first_events(stim_ev)
    first_stim_ev_15ms = extract_first_events(stim_ev_15ms)
    first_stim_ev_30ms = extract_first_events(stim_ev_30ms)

    align_ev.update(
        {
            "stim_ev": stim_ev,
            "first_stim_ev": first_stim_ev,
            "stim_ev_15ms": stim_ev_15ms,
            "stim_ev_30ms": stim_ev_30ms,
            "first_stim_ev_15ms": first_stim_ev_15ms,
            "first_stim_ev_30ms": first_stim_ev_30ms,
        }
    )

    return align_ev


def fetch_trial_metadata(
    subject: str,
    session: str,
    align_ev: dict[str, np.ndarray],
) -> Optional[pd.DataFrame]:
    """Fetch Chipmunk trial metadata and align with OBX trial_start timestamps.

    Returns a DataFrame with trial-level metadata or None if Chipmunk data
    is unavailable.
    """
    try:
        from chipmunk import Chipmunk  # type: ignore

        sess_dicts = (
            SpikeSorting()
            & f'subject_name = "{subject}"'
            & f'session_name = "{session}"'
        ).fetch("subject_name", "session_name", as_dict=True)

        trial_data = (
            (Chipmunk() & sess_dicts)
            * Chipmunk.Trial().proj("response", "rewarded")
            * Chipmunk.TrialParameters().proj("stim_rate_vision", "category_boundary")
        ).fetch(format="frame")
        tdf: pd.DataFrame = trial_data.reset_index(
            level=["subject_name", "session_name", "dataset_name"], drop=True
        ).sort_index()

        trial_starts = align_ev["trial_start"]
        n = min(len(trial_starts), len(tdf))
        if len(trial_starts) != len(tdf):
            print(
                f"Warning: {len(trial_starts)} OBX trial_start pulses vs "
                f"{len(tdf)} Chipmunk trials — using first {n}"
            )
        trial_df = tdf.iloc[:n].copy()
        trial_df["trial_start_ts"] = trial_starts[:n]
        trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
        trial_df["prev_response"] = trial_df["response"].shift(1)
        trial_df["stim_category"] = pd.cut(
            trial_df["stim_rate_vision"] - trial_df["category_boundary"],
            bins=[-np.inf, -1e-9, 1e-9, np.inf],
            labels=["low_rate", "boundary", "high_rate"],
        )
        return trial_df
    except Exception as e:
        print(f"Could not load Chipmunk trial metadata: {e}")
        return None
