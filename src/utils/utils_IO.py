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
REQUIRED_LOGICAL_EVENTS = (
    "visual_stim",
    "trial_start",
    "frames",
    "left_port",
    "center_port",
    "right_port",
)


def _format_available_event_rows(events: pd.DataFrame) -> list[str]:
    return sorted(
        {
            f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
            for row in events[["dataset_name", "stream_name", "event_name"]].itertuples(
                index=False
            )
        }
    )


def _fetch_session_digital_events(subject: str, session: str) -> pd.DataFrame:
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()
    dset = (Dataset() & sess_query).proj()
    events = pd.DataFrame((DatasetEvents.Digital() & dset).fetch_synced())
    if events.empty:
        raise ValueError(f"No DatasetEvents.Digital rows found for {subject} {session}")
    return events


def _fetch_event_mapping_rows(subject: str, session: str) -> pd.DataFrame:
    from labdata_plugin.analysisschema import EventMapping

    mapping = pd.DataFrame(
        (
            EventMapping()
            & f'subject_name = "{subject}"'
            & f'session_name = "{session}"'
        ).fetch(as_dict=True)
    )
    if mapping.empty:
        raise ValueError(f"No EventMapping rows found for {subject} {session}")
    return mapping


def _resolve_mapped_event_timestamps(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    mapped_event_names = mapping["event_name"].tolist()
    missing = [
        name for name in REQUIRED_LOGICAL_EVENTS if name not in mapped_event_names
    ]
    if missing:
        available_mappings = sorted(set(mapped_event_names))
        raise ValueError(
            f"Missing EventMapping rows for {subject} {session}: {missing}. "
            f"Available mapped names: {available_mappings}"
        )

    duplicates = mapping["event_name"][mapping["event_name"].duplicated()].unique()
    if duplicates.size:
        raise ValueError(
            f"Duplicate EventMapping rows for {subject} {session}: "
            f"{sorted(duplicates.tolist())}"
        )

    resolved: dict[str, np.ndarray] = {}
    available_rows = _format_available_event_rows(events)
    for logical_name in REQUIRED_LOGICAL_EVENTS:
        row = mapping.loc[mapping["event_name"] == logical_name].iloc[0]
        mask = (
            (events["dataset_name"] == row["source_dataset_name"])
            & (events["stream_name"] == row["source_stream_name"])
            & (events["event_name"] == row["source_event_name"])
        )
        if not mask.any():
            raise ValueError(
                f"Mapped source row is missing for {subject} {session} "
                f"{logical_name}: {row['source_dataset_name']}:"
                f"{row['source_stream_name']}:{row['source_event_name']}. "
                f"Available rows: {available_rows}"
            )
        resolved[logical_name] = np.asarray(
            events.loc[mask, "event_timestamps"].iloc[0], dtype=float
        )
    return resolved


def _extract_trial_start_onsets(trial_start_events: np.ndarray) -> np.ndarray:
    trial_start_events = np.asarray(trial_start_events, dtype=float)
    return trial_start_events[::2]


def _extract_first_events(ev_array: np.ndarray, sep_s: float = 1.0) -> np.ndarray:
    if ev_array.size == 0:
        return ev_array
    keep = np.r_[True, np.diff(ev_array) > sep_s]
    return ev_array[keep]


def _build_stim_event_streams(stim: np.ndarray) -> dict[str, np.ndarray]:
    stim = np.asarray(stim, dtype=float)

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

    tol_s = 2e-3
    if durations.size and np.allclose(durations, 0.0):
        # Historical GRB006 repairs insert onset-only visual events instead of
        # raw TTL edges, so treat the mapped row as a 15 ms-only stim stream.
        labels = np.full(durations.shape, "15ms", dtype=object)
    else:
        diff_15 = np.abs(durations - 0.015)
        diff_30 = np.abs(durations - 0.030)
        is_15 = diff_15 <= tol_s
        is_30 = diff_30 <= tol_s
        labels = np.where(is_15, "15ms", np.where(is_30, "30ms", "unknown"))

    stim_ev = onsets
    stim_ev_15ms = onsets[labels == "15ms"]
    stim_ev_30ms = onsets[labels == "30ms"]

    return {
        "stim_ev": stim_ev,
        "first_stim_ev": _extract_first_events(stim_ev),
        "stim_ev_15ms": stim_ev_15ms,
        "stim_ev_30ms": stim_ev_30ms,
        "first_stim_ev_15ms": _extract_first_events(stim_ev_15ms),
        "first_stim_ev_30ms": _extract_first_events(stim_ev_30ms),
    }


def _fetch_good_units_table(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> tuple[pd.DataFrame, float]:
    """Return good-unit rows with spike_times and depth, sorted by depth.

    `unit_criteria_id=1` is the project's standard quality criterion set
    (amplitude / SNR / contamination thresholds defined upstream in labdata).
    Don't change without reason — most downstream analyses assume criterion 1.
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
    return good_units, srate


def fetch_good_units(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> dict[int, np.ndarray]:
    """Fetch spike times (in seconds) for units passing quality criteria.

    Returns a dict mapping unit_id → spike_times_seconds, sorted by depth.
    """
    good_units, srate = _fetch_good_units_table(subject, session, unit_criteria_id)
    st_per_unit = {
        row["unit_id"]: row["spike_times"] / srate for _, row in good_units.iterrows()
    }
    return st_per_unit


def fetch_good_units_with_depth(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Fetch good units and aligned depth metadata sorted by depth."""
    good_units, srate = _fetch_good_units_table(subject, session, unit_criteria_id)
    st_per_unit = {
        row["unit_id"]: row["spike_times"] / srate for _, row in good_units.iterrows()
    }
    depth_per_unit = {
        int(row["unit_id"]): float(row["depth"]) for _, row in good_units.iterrows()
    }
    return st_per_unit, depth_per_unit


def fetch_session_events(
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    """Fetch digital events for a session and derive stimulus event arrays.

    Raw digital edges on the stim channel are noisy: a single logical pulse
    toggles many times. They are merged into discrete bursts by splitting on
    any gap > 20 ms, then each burst's duration is classified against the two
    expected pulse widths (15 ms and 30 ms, ±2 ms tolerance). Bursts that
    match neither are labeled "unknown" and excluded from the width-specific
    streams but still appear in `stim_ev` / `first_stim_ev`.

    `first_*` variants keep only the first onset within each 1 s window, so
    they approximate the first pulse of each stimulus train — the cleaner
    alignment event when comparing single-pulse responses (e.g. for the
    double-peak analysis).

    Returns a dict with the following keys (all np.ndarray of timestamps in
    seconds, possibly empty):
      - `stim`              : raw stim edges (no merging)
      - `trial_start`, `frames`, `left_port`, `center_port`, `right_port`
      - `stim_ev`           : onsets of all merged pulses (any width)
      - `first_stim_ev`     : first-of-train onsets across all widths
      - `stim_ev_15ms`      : onsets of pulses classified as 15 ms
      - `stim_ev_30ms`      : onsets of pulses classified as 30 ms
      - `first_stim_ev_15ms`: first-of-train onsets for 15 ms pulses only
      - `first_stim_ev_30ms`: first-of-train onsets for 30 ms pulses only
    """
    events = _fetch_session_digital_events(subject, session)
    mapping = _fetch_event_mapping_rows(subject, session)
    resolved = _resolve_mapped_event_timestamps(events, mapping, subject, session)

    # Trial-start TTLs still arrive as on/off edge pairs, regardless of the
    # physical source row used for this session.
    trial_start = _extract_trial_start_onsets(resolved["trial_start"])
    align_ev: dict[str, np.ndarray] = {
        "stim": resolved["visual_stim"],
        "trial_start": trial_start,
        "frames": resolved["frames"],
        "left_port": resolved["left_port"],
        "center_port": resolved["center_port"],
        "right_port": resolved["right_port"],
    }
    align_ev.update(_build_stim_event_streams(align_ev["stim"]))
    return align_ev


def fetch_trial_metadata(
    subject: str,
    session: str,
    align_ev: dict[str, np.ndarray],
) -> Optional[pd.DataFrame]:
    """Fetch Chipmunk trial metadata and align with OBX trial_start timestamps.

    Returns a DataFrame with trial-level metadata or None if Chipmunk data
    is unavailable.

    Trial-count mismatches are treated conservatively. A one-trial mismatch is
    tolerated as a likely trailing partial trial and is truncated with a
    warning. Larger mismatches raise instead of silently truncating.
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
            * Chipmunk.Trial().proj(
                "response",
                "with_choice",
                "rewarded",
                "early_withdrawal",
                "t_start",
                "t_sync",
                "t_initiate",
                "t_stim",
                "t_gocue",
                "t_react",
                "t_response",
                "stim_duration",
            )
            * Chipmunk.TrialParameters().proj("stim_rate_vision", "category_boundary")
        ).fetch(format="frame")
        tdf: pd.DataFrame = trial_data.reset_index(
            level=["subject_name", "session_name", "dataset_name"], drop=True
        ).sort_index()

        trial_starts = align_ev["trial_start"]
        n_obx = len(trial_starts)
        n_chipmunk = len(tdf)
        n = min(n_obx, n_chipmunk)
        mismatch = abs(n_obx - n_chipmunk)
        if n == 0:
            raise ValueError(
                f"No aligned trials available for {subject} {session}: "
                f"OBX={n_obx}, Chipmunk={n_chipmunk}"
            )
        if mismatch:
            if mismatch > 1:
                raise ValueError(
                    f"Suspicious trial-count mismatch for {subject} {session}: "
                    f"OBX trial_start pulses={n_obx}, Chipmunk trials={n_chipmunk}. "
                    "Refusing to silently truncate."
                )
            print(
                f"Warning: {subject} {session} has a 1-trial OBX/Chipmunk mismatch "
                f"(OBX={n_obx}, Chipmunk={n_chipmunk}); truncating to {n}."
            )
        trial_df = tdf.iloc[:n].copy()
        trial_df["trial_start_ts"] = trial_starts[:n]
        trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
        trial_df["prev_response"] = trial_df["response"].shift(1)
        trial_df["prev_stim_rate"] = trial_df["stim_rate_vision"].shift(1)
        trial_df["stim_category"] = pd.cut(
            trial_df["stim_rate_vision"] - trial_df["category_boundary"],
            bins=[-np.inf, -1e-9, 1e-9, np.inf],
            labels=["low_rate", "boundary", "high_rate"],
        )
        return trial_df
    except Exception as e:
        print(f"Could not load Chipmunk trial metadata: {e}")
        return None


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
    from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table
    from ephys.src.utils.utils_analysis import build_trial_stim_classification

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
