from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import spks  # type: ignore
import chiCa.chiCa as chiCa  # type: ignore
from os.path import join as pjoin
from glob import glob


PortEventDict = Dict[str, Dict[str, np.ndarray]]


def load_sync_data(
    sessionpath: str, sync_port: int = 0
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    float,
    dict[str, np.ndarray],
]:
    print("Loading nisync data...")
    (nionsets, nioffsets), (nisync, nimeta), (apsyncdata) = spks.sync.load_ni_sync_data(
        sessionpath=sessionpath
    )
    aponsets = apsyncdata[0]["file0_sync_onsets"][6]

    corrected_onsets = {}
    corrected_offsets = {}
    for k in nionsets.keys():
        corrected_onsets[k] = spks.sync.interp1d(
            nionsets[sync_port], aponsets, fill_value="extrapolate"
        )(nionsets[k]).astype("uint64")
        corrected_offsets[k] = spks.sync.interp1d(
            nionsets[sync_port], aponsets, fill_value="extrapolate"
        )(nioffsets[k]).astype("uint64")
    del k

    nitime = spks.sync.interp1d(
        nionsets[sync_port], aponsets, fill_value="extrapolate"
    )(np.arange(len(nisync)))
    srate = apsyncdata[0]["sampling_rate"]
    t = nitime / srate
    visual_analog = nisync[:, 0]
    audio_analog = nisync[:, 1]
    print("Success!\n-----")

    analog_signals: dict[str, np.ndarray] = {
        "visual": visual_analog,
        "audio": audio_analog,
    }

    return corrected_onsets, corrected_offsets, t, srate, analog_signals


def process_port_events(
    corrected_onsets: dict[int, np.ndarray],
    corrected_offsets: dict[int, np.ndarray],
    srate: float,
) -> tuple[np.ndarray, Optional[PortEventDict]]:
    print("Loading nidaq events...")
    trial_starts = corrected_onsets[2] / srate

    if len(corrected_onsets.keys()) <= 3:
        print("No port events registered in this session. Proceeding without them...")
        return trial_starts, None

    print("Port events found. Proceeding with extracting them...")
    port_events: PortEventDict = {
        "center_port": {
            "entries": corrected_onsets[4] / srate,
            "exits": corrected_offsets[4] / srate,
        },
        "left_port": {
            "entries": corrected_onsets[3] / srate,
            "exits": corrected_offsets[3] / srate,
        },
        "right_port": {
            "entries": corrected_onsets[5] / srate,
            "exits": corrected_offsets[5] / srate,
        },
    }

    return trial_starts, port_events


def process_trial_data(
    sessionpath: str,
    trial_starts: np.ndarray,
    t: np.ndarray,
    srate: float,
    analog_signals: dict[str, np.ndarray],
    port_events: Optional[PortEventDict],
    animal: str,
    session: str,
) -> tuple[Any, pd.DataFrame, dict[str, np.ndarray]]:
    behavior_data = chiCa.load_trialdata(
        glob(pjoin(sessionpath, f"chipmunk/{animal}_{session}_chipmunk_*.mat"))[0]
    )

    stim_ts_per_channel: dict[str, np.ndarray] = {}
    for channel_name, signal in analog_signals.items():
        if not isinstance(signal, np.ndarray):
            continue
        if signal.ndim != 1:
            continue
        if channel_name == "visual":
            thresh = 5000
        elif channel_name == "audio":
            thresh = 2500
        stim_ts_per_channel[channel_name] = detect_stim_events(
            t, srate, signal, amp_threshold=thresh, time_threshold=0.04
        )

    if not stim_ts_per_channel:
        raise ValueError("No 1-D analog signals provided for stimulus detection.")

    if port_events is None:
        raise Warning("No port events found. Proceeding without them...")
        trial_ts = get_trial_ts(
            trial_starts,
            stim_ts_per_channel,
            behavior_data,
        )
    else:
        trial_ts = get_trial_ts(
            trial_starts,
            stim_ts_per_channel,
            behavior_data,
            port_events,
        )
        trial_ts = trial_ts[trial_ts.trial_outcome != -1].copy()

        for itrial, data in trial_ts.iterrows():
            if len(data.center_port_entries) == 0:
                continue
            mask = (
                data.center_port_entries
                < data.first_stim_ts  # this breaks because the audio stream detection is not good right now so it misses those events
            )  # true if entry is before first stim onset. sometimes mice will briefly poke again in the center before reporting their choice
            true_entry = data.center_port_entries[
                mask
            ][
                -1
            ]  # get the last entry before stim onset since sometimes the first poke will not be long enough to be considered a valid entry
            trial_ts.loc[itrial, "center_port_entries"] = [  # type: ignore[index]
                true_entry
            ]  # replace the entries with the last entry before stim onset

            mask = (
                data.center_port_exits > true_entry
            )  # true if exit is after the last entry before stim onset
            true_exit = data.center_port_exits[mask][
                0
            ]  # get the first exit after stim onset
            trial_ts.loc[itrial, "center_port_exits"] = [  # type: ignore[index]
                true_exit
            ]  # replace the exits with the first exit after stim onset
        trial_ts.insert(
            trial_ts.shape[1], "response", trial_ts.apply(get_response_ts, axis=1)
        )

    trial_ts.insert(
        trial_ts.shape[1],
        "stationary_stims",
        trial_ts.apply(get_stationary_stims, axis=1),
    )
    trial_ts.insert(
        trial_ts.shape[1], "movement_stims", trial_ts.apply(get_movement_stims, axis=1)
    )

    trial_ts.insert(
        0,
        "category",
        trial_ts.apply(
            lambda x: (
                "left"
                if x.trial_rate < 12
                else ("right" if x.trial_rate > 12 else "boundary")
            ),
            axis=1,
        ),
    )

    print("Success!")
    return behavior_data, trial_ts, stim_ts_per_channel


def get_trial_ts(
    trial_starts: np.ndarray,
    stim_ts_per_channel: dict[str, np.ndarray],
    behavior_data: Any,
    port_events: Optional[PortEventDict] = None,
) -> pd.DataFrame:
    trial_starts = np.asarray(trial_starts)
    trial_data: list[dict[str, Any]] = []

    channel_names = list(stim_ts_per_channel.keys())
    channel_events = {
        name: np.searchsorted(stim_ts_per_channel[name], trial_starts)
        for name in channel_names
    }

    # TODO: see if this is working properly
    if channel_names:
        combined_stim_ts = np.sort(
            np.unique(
                np.concatenate(
                    [
                        stim_ts_per_channel[name]
                        for name in channel_names
                        if stim_ts_per_channel[name].size > 0
                    ]
                )
            )
        )
    else:
        combined_stim_ts = np.array([])

    combined_events = (
        np.searchsorted(combined_stim_ts, trial_starts)
        if combined_stim_ts.size > 0
        else np.zeros_like(trial_starts, dtype=int)
    )

    for ti in range(len(trial_starts) - 1):
        start_time, end_time = trial_starts[ti], trial_starts[ti + 1]
        stim_dict = {}

        for name in channel_names:
            channel_start = channel_events[name][ti]
            channel_end = channel_events[name][ti + 1]
            channel_ts = stim_ts_per_channel[name][channel_start:channel_end]
            stim_dict[f"stim_ts_{name}"] = channel_ts

        combined_start = combined_events[ti]
        combined_end = combined_events[ti + 1]
        stim_events_in_interval = combined_stim_ts[combined_start:combined_end]

        detected_events = combined_end - combined_start
        first_stim_ts = (
            stim_events_in_interval[0] if len(stim_events_in_interval) > 0 else np.nan
        )

        trial_dict = {
            "trial_rate": len(behavior_data.stimulus_event_timestamps[ti]),
            "detected_events": detected_events,
            "trial_start": start_time,
            "stim_ts": stim_events_in_interval,
            "first_stim_ts": first_stim_ts,
            "stimulus_modality": behavior_data.stimulus_modality[ti],
            "response_side": behavior_data.response_side[ti],
            "correct_side": behavior_data.correct_side[ti],
            "trial_outcome": behavior_data.outcome_record[ti],
        }

        if port_events is not None:
            # Vectorized operations for port events
            for port_name, events in port_events.items():
                for event_type in ["entries", "exits"]:
                    event_times = events[event_type]
                    mask = (event_times > start_time) & (event_times < end_time)
                    trial_dict[f"{port_name}_{event_type}"] = event_times[mask]

        trial_dict.update(stim_dict)

        trial_data.append(trial_dict)

    return pd.DataFrame(trial_data)


def detect_stim_events(
    time_vector: np.ndarray,
    srate: float,
    signal: np.ndarray,
    amp_threshold: float = 5000,
    time_threshold: float = 0.04,
) -> np.ndarray:
    ii = np.where(np.diff(signal > amp_threshold) == 1)[0]
    return time_vector[ii[np.diff(np.hstack([0, ii])) > time_threshold * srate]]


def get_response_ts(row: pd.Series) -> Optional[float]:
    try:
        w: Optional[float] = float(row["center_port_exits"][-1])
    except Exception:
        w = None
        print(
            f"Couldn't find a withdrawal time for trial{row}... Make sure this is correct."
        )

    if w is None:
        return None

    left_candidates: Sequence[float] = [
        float(entry) for entry in row["left_port_entries"] if entry > w
    ]
    right_candidates: Sequence[float] = [
        float(entry) for entry in row["right_port_entries"] if entry > w
    ]

    left = left_candidates[0] if left_candidates else None
    right = right_candidates[0] if right_candidates else None

    if left is not None and pd.notna(left):
        return left
    if right is not None and pd.notna(right):
        return right
    return None


def get_stationary_stims(row: pd.Series) -> np.ndarray:
    return row.stim_ts[row.stim_ts < row.center_port_exits[0]]


def get_movement_stims(row: pd.Series, max_tseconds: float = 0.4) -> np.ndarray:
    if "center_port_exits" in row.index:  # this check might be redundant 4/16/25
        if len(row.center_port_exits) != 0:
            return row.stim_ts[row.stim_ts > row.center_port_exits[0]]
    else:
        return row.stim_ts[
            np.logical_and(
                row.first_stim_ts + 0.5 < row.stim_ts,
                row.stim_ts < row.first_stim_ts + 0.5 + max_tseconds,
            )
        ]
    return np.array([])
