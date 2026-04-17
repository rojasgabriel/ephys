"""Behavior + stat-vs-move summary for GRB006 and GRB058.

Builds a 6x3 figure:
  Row 0: GRB006 behavior summary
  Row 1: GRB006 example units
  Row 2: GRB006 all/low/high stat-vs-move scatters
  Row 3: GRB058 behavior summary
  Row 4: GRB058 example units
  Row 5: GRB058 all/low/high stat-vs-move scatters
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import sem, wilcoxon
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
    compute_unit_selectivity,
    find_unique_cross_trial_offset_pairs,
)

LOCAL_TRIAL_TS = Path("/Users/gabriel/Downloads/Organized/Code/trial_ts.pkl")
LOCAL_SPIKE_TIMES = Path(
    "/Users/gabriel/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl"
)
OUTPUT_STEM_MAIN = "loco_scatter_grb006-3rdStat_grb058-lastStat"
OUTPUT_STEM_OVERLAY = "loco_overlay_grb006-3rdStat_grb058-lastStat"
OUTPUT_STEM_OVERLAY_SUMMARY = "loco_timingCtrl_grb006-3rdStat_grb058-lastStat"
OUTPUT_STEM_DEPTH = "loco_depth_grb006-3rdStat_grb058-lastStat"
OUT_PATH = Path(f"/Users/gabriel/lib/ephys/figures/{OUTPUT_STEM_MAIN}.pdf")
OUT_PATH_OVERLAY = Path(f"/Users/gabriel/lib/ephys/figures/{OUTPUT_STEM_OVERLAY}.pdf")
OUT_PATH_OVERLAY_SUMMARY = Path(
    f"/Users/gabriel/lib/ephys/figures/{OUTPUT_STEM_OVERLAY_SUMMARY}.pdf"
)
OUT_PATH_DEPTH = Path(f"/Users/gabriel/lib/ephys/figures/{OUTPUT_STEM_DEPTH}.pdf")

PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
)
PETH_SCALE_BACK = PETH_KWARGS["binwidth_ms"] / 1000.0
RESP_WINDOW = (0.04, 0.10)
EFFECT_WINDOW = (0.0, 0.12)
PEAK_HALF_WINDOW_S = 0.015
CANONICAL_LATENCY_WINDOW = (0.015, 0.070)
TITLE_KW = dict(fontsize=9, pad=3)
RATE_SPLIT_HZ = 12.0
DEPTH_BIN_WIDTH_UM = 100.0
PNG_DPI = 300
MAIN_ANCHOR_CONFIG = {
    "GRB006": {"stationary_index": 2},
    "GRB058": {"stationary_index": None},
}
SUMMARY_OFFSET_WINDOW_S = (0.5, 0.7)
SUMMARY_OFFSET_WIGGLE_S = 0.1


@dataclass
class BehaviorSummary:
    n_trials: int
    paired_trial_count: int
    outcome_labels: list[str]
    outcome_counts: np.ndarray
    paired_rate_values: np.ndarray
    paired_rate_counts: np.ndarray
    fixation_durations: np.ndarray
    response_durations: np.ndarray


@dataclass
class SessionInputs:
    subject: str
    session: str
    unit_ids: list[int]
    spike_times: list[np.ndarray]
    unit_depth_um: np.ndarray
    first_stim_times: np.ndarray
    trial_df: pd.DataFrame
    trial_ts: pd.DataFrame
    classified_trial_rates: pd.Series


@dataclass
class SessionAnalysis:
    subject: str
    session: str
    unit_ids: list[int]
    spike_times: list[np.ndarray]
    unit_depth_um: np.ndarray
    behavior: BehaviorSummary
    paired_trial_df: pd.DataFrame
    bin_centers: np.ndarray
    paired_last_stat: np.ndarray
    paired_first_move: np.ndarray
    stationary_label: str
    paired_trial_rates: np.ndarray
    peth_stat_all: np.ndarray
    peth_move_all: np.ndarray
    stat_trial_matrix: np.ndarray
    move_trial_matrix: np.ndarray
    example_indices: list[int]
    peak_latencies: np.ndarray
    delta_move: np.ndarray
    qvals_move: np.ndarray
    pk_stat_all: np.ndarray
    pk_move_all: np.ndarray
    pk_stat_low: np.ndarray
    pk_move_low: np.ndarray
    pk_stat_high: np.ndarray
    pk_move_high: np.ndarray
    sem_stat_all: np.ndarray
    sem_move_all: np.ndarray
    sem_stat_low: np.ndarray
    sem_move_low: np.ndarray
    sem_stat_high: np.ndarray
    sem_move_high: np.ndarray
    rate_threshold_hz: float
    low_count: int
    high_count: int
    paired_trial_count: int
    n_units: int


@dataclass(frozen=True)
class ComparisonDef:
    key: str
    row_label: str
    description: str
    mode: str
    stationary_index_by_subject: dict[str, int | None] | None = None
    offset_window_s: tuple[float, float] | None = None
    wiggle_room_s: float | None = None


def ordinal_label(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def extract_session_conditioned_anchors(
    trial_ts: pd.DataFrame, stationary_index: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    has_move = trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
    if stationary_index is None:
        has_stat = trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
        paired_trials = trial_ts[has_stat & has_move]
        paired_stat = np.array(
            [stims[-1] for stims in paired_trials["stationary_stims"]],
            dtype=float,
        )
        stationary_label = "last stationary"
    else:
        has_stat = trial_ts["stationary_stims"].apply(
            lambda x: len(x) > stationary_index
        )
        paired_trials = trial_ts[has_stat & has_move]
        paired_stat = np.array(
            [stims[stationary_index] for stims in paired_trials["stationary_stims"]],
            dtype=float,
        )
        stationary_label = f"{ordinal_label(stationary_index + 1)} stationary"
    paired_first_move = np.array(
        [stims[0] for stims in paired_trials["movement_stims"]],
        dtype=float,
    )
    paired_trial_idx = paired_trials["trial_idx"].to_numpy(dtype=int)
    return paired_stat, paired_first_move, paired_trial_idx, stationary_label


def trial_start_from_row(row: pd.Series) -> float:
    if "center_port_entries" in row.index:
        entries = row["center_port_entries"]
        if entries is None or len(entries) == 0:
            return np.nan
        return float(entries[0])
    if "cp_entry" in row.index:
        return float(row["cp_entry"]) if np.isfinite(row["cp_entry"]) else np.nan
    return np.nan


def extract_offset_matched_anchors(
    trial_ts: pd.DataFrame,
    offset_window_s: tuple[float, float],
    wiggle_room_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    lo, hi = offset_window_s
    records = []
    for _, row in trial_ts.iterrows():
        trial_start = trial_start_from_row(row)
        if not np.isfinite(trial_start):
            continue
        stationary_stims = (
            row["stationary_stims"] if row["stationary_stims"] is not None else []
        )
        movement_stims = (
            row["movement_stims"] if row["movement_stims"] is not None else []
        )

        for i, st in enumerate(stationary_stims):
            if i == 0:
                continue
            offset = float(st - trial_start)
            if lo <= offset <= hi:
                records.append(
                    {
                        "trial_idx": int(row["trial_idx"]),
                        "stim_time": float(st),
                        "offset": offset,
                        "movement_status": 0,
                    }
                )
        for mt in movement_stims:
            offset = float(mt - trial_start)
            if lo <= offset <= hi:
                records.append(
                    {
                        "trial_idx": int(row["trial_idx"]),
                        "stim_time": float(mt),
                        "offset": offset,
                        "movement_status": 1,
                    }
                )

    stims_offset_df = pd.DataFrame(records)
    if stims_offset_df.empty:
        return (
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            "offset-matched 0.5-0.7 s",
        )

    matched_pairs_df = find_unique_cross_trial_offset_pairs(
        stims_offset_df,
        wiggle_room=wiggle_room_s,
    )
    if matched_pairs_df.empty:
        return (
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            "offset-matched 0.5-0.7 s",
        )

    paired_stat = matched_pairs_df["stat_stim_time"].to_numpy(dtype=float)
    paired_move = matched_pairs_df["move_stim_time"].to_numpy(dtype=float)
    paired_idx = matched_pairs_df["move_trial_idx"].to_numpy(dtype=int)
    label = f"offset-matched {lo:.1f}-{hi:.1f} s"
    return paired_stat, paired_move, paired_idx, label


def extract_comparison_anchors(
    inputs: SessionInputs,
    comparison: ComparisonDef,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if comparison.mode == "indexed_stationary":
        stationary_index = None
        if comparison.stationary_index_by_subject is not None:
            stationary_index = comparison.stationary_index_by_subject.get(
                inputs.subject
            )
        return extract_session_conditioned_anchors(inputs.trial_ts, stationary_index)
    if comparison.mode == "offset_matched":
        if comparison.offset_window_s is None or comparison.wiggle_room_s is None:
            raise ValueError(f"Missing offset matching parameters for {comparison.key}")
        return extract_offset_matched_anchors(
            inputs.trial_ts,
            offset_window_s=comparison.offset_window_s,
            wiggle_room_s=comparison.wiggle_room_s,
        )
    raise ValueError(f"Unknown comparison mode: {comparison.mode}")


def load_local_spike_times(
    spike_times_path: Path, sampling_rate: float = 30000.0
) -> tuple[list[int], list[np.ndarray]]:
    with spike_times_path.open("rb") as f:
        spike_df = pickle.load(f)

    unit_ids = spike_df["unit_id"].astype(int).tolist()
    spike_times = [
        np.asarray(times, dtype=float) / sampling_rate
        for times in spike_df["spike_times"].tolist()
    ]
    return unit_ids, spike_times


def load_local_trial_ts(
    trial_ts_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True).copy()
    trial_ts["trial_idx"] = np.arange(len(trial_ts))
    first_stim_times = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim_times = first_stim_times[np.isfinite(first_stim_times)]
    classified_trial_rates = pd.Series(
        trial_ts["trial_rate"].to_numpy(dtype=float),
        index=trial_ts["trial_idx"].to_numpy(),
    )
    return trial_ts, first_stim_times, classified_trial_rates


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


def load_db_behavior(
    subject: str, session: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
    from ephys.src.utils.utils_IO import fetch_session_events, fetch_trial_metadata

    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}")

    trial_df = enrich_trial_df(trial_df)
    trial_ts = build_trial_stim_classification(align_ev, trial_df).reset_index(
        drop=True
    )
    first_stim_times = np.asarray(align_ev["first_stim_ev_15ms"], dtype=float)
    classified_trial_rates = (
        trial_df["stim_rate_vision"]
        .iloc[trial_ts["trial_idx"].to_numpy()]
        .set_axis(trial_ts["trial_idx"].to_numpy())
    )
    return trial_df, trial_ts, first_stim_times, classified_trial_rates


def load_local_spikes_db_behavior(
    subject: str,
    session: str,
    trial_ts_path: Path,
    spike_times_path: Path,
    sampling_rate: float = 30000.0,
):
    from ephys.src.utils.utils_IO import fetch_good_units_with_depth

    print(f"\nLoading hybrid session: {subject} {session}")
    trial_df = fetch_chipmunk_session_trials(subject, session)
    trial_ts, first_stim_times, classified_trial_rates = load_local_trial_ts(
        trial_ts_path
    )
    matched_full_idx = align_local_trials_to_full_trial_df(trial_ts, trial_df)
    trial_ts["trial_idx"] = matched_full_idx
    classified_trial_rates = pd.Series(
        trial_df["stim_rate_vision"].iloc[matched_full_idx].to_numpy(dtype=float),
        index=matched_full_idx,
    )
    unit_ids, spike_times = load_local_spike_times(spike_times_path, sampling_rate)
    _, depth_per_unit = fetch_good_units_with_depth(subject, session)
    unit_depth_um = np.array(
        [depth_per_unit.get(uid, np.nan) for uid in unit_ids], dtype=float
    )
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired trials: {len(trial_ts)}  First stims: {len(first_stim_times)}"
    )
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        unit_depth_um=unit_depth_um,
        first_stim_times=first_stim_times,
        trial_df=trial_df,
        trial_ts=trial_ts,
        classified_trial_rates=classified_trial_rates,
    )


def load_db_session(subject: str, session: str):
    from ephys.src.utils.utils_IO import fetch_good_units_with_depth

    print(f"\nLoading DB session: {subject} {session}")
    trial_df, trial_ts, first_stim_times, classified_trial_rates = load_db_behavior(
        subject, session
    )
    st_per_unit, depth_per_unit = fetch_good_units_with_depth(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    unit_depth_um = np.array([depth_per_unit[uid] for uid in unit_ids], dtype=float)
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired trials: {len(trial_ts)}  First stims: {len(first_stim_times)}"
    )
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        unit_depth_um=unit_depth_um,
        first_stim_times=first_stim_times,
        trial_df=trial_df,
        trial_ts=trial_ts,
        classified_trial_rates=classified_trial_rates,
    )


def resp_per_unit(peth, bc, window=RESP_WINDOW):
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def resp_sem_log_per_unit(peth, bc, window=RESP_WINDOW):
    mask = (bc >= window[0]) & (bc < window[1])
    per_trial = peth[:, :, mask].mean(axis=2)  # (n_units, n_trials)
    mean_rate = per_trial.mean(axis=1)
    log_rates = np.log(
        np.clip(per_trial, 1e-3, None)
    )  # clip avoids log(0) on zero-spike trials
    sem_log = log_rates.std(axis=1) / np.sqrt(per_trial.shape[1])
    lower = mean_rate * (1.0 - np.exp(-sem_log))  # bar below dot, data units
    upper = mean_rate * (np.exp(sem_log) - 1.0)  # bar above dot, data units
    return lower, upper


def wilcox_str(a, b):
    if len(a) == 0 or len(b) == 0:
        return "no trials"
    _, p = wilcoxon(a, b)
    pct = 100 * (b > a).mean()
    return f"{pct:.0f}% above diag  p={p:.3f}"


def per_unit_move_vs_stat_stats(
    peth_stat,
    peth_move,
    bc,
    window=EFFECT_WINDOW,
    half_window_s=PEAK_HALF_WINDOW_S,
):
    effect_mask = (bc >= window[0]) & (bc < window[1])
    if not effect_mask.any():
        raise ValueError("EFFECT_WINDOW does not overlap available bins.")

    n_units, n_trials, _ = peth_stat.shape
    stat_trials = np.zeros((n_units, n_trials), dtype=float)
    move_trials = np.zeros((n_units, n_trials), dtype=float)
    peak_latencies = np.full(n_units, np.nan, dtype=float)

    mean_stat = peth_stat.mean(axis=1)
    mean_move = peth_move.mean(axis=1)
    mean_combined = 0.5 * (mean_stat + mean_move)
    bc_effect = bc[effect_mask]

    for ui in range(n_units):
        peak_idx = np.argmax(mean_combined[ui, effect_mask])
        peak_t = bc_effect[peak_idx]
        peak_latencies[ui] = peak_t
        local_mask = (
            (bc >= peak_t - half_window_s)
            & (bc <= peak_t + half_window_s)
            & effect_mask
        )
        if not local_mask.any():
            local_mask = np.zeros_like(bc, dtype=bool)
            local_mask[np.argmin(np.abs(bc - peak_t))] = True
        stat_trials[ui] = peth_stat[ui][:, local_mask].mean(axis=1)
        move_trials[ui] = peth_move[ui][:, local_mask].mean(axis=1)

    delta = move_trials.mean(axis=1) - stat_trials.mean(axis=1)
    pvals = np.ones(len(delta), dtype=float)
    for ui in range(len(delta)):
        diff = move_trials[ui] - stat_trials[ui]
        if np.allclose(diff, 0.0):
            pvals[ui] = 1.0
            continue
        try:
            _, p = wilcoxon(
                move_trials[ui],
                stat_trials[ui],
                alternative="two-sided",
                zero_method="wilcox",
            )
        except ValueError:
            p = 1.0
        pvals[ui] = p if np.isfinite(p) else 1.0
    qvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]
    return delta, pvals, qvals, peak_latencies, stat_trials, move_trials


def snr_per_unit(peth, bc, window=RESP_WINDOW):
    mask = (bc >= window[0]) & (bc < window[1])
    resp = peth[:, :, mask].mean(axis=2)
    unit_mean = resp.mean(axis=1)
    unit_sem = resp.std(axis=1) / np.sqrt(resp.shape[1])
    return unit_mean / (unit_sem + 1e-3)


def split_resp_per_unit(spike_times, stat_times, move_times, bc):
    if len(stat_times) == 0 or len(move_times) == 0:
        empty = np.array([])
        empty2 = np.zeros((2, 0))
        return empty, empty, empty2, empty2
    peth_stat, _, _ = compute_population_peth(
        spike_times,
        stat_times,
        **PETH_KWARGS,
    )
    peth_move, _, _ = compute_population_peth(
        spike_times,
        move_times,
        **PETH_KWARGS,
    )
    peth_stat *= PETH_SCALE_BACK
    peth_move *= PETH_SCALE_BACK
    return (
        resp_per_unit(peth_stat, bc),
        resp_per_unit(peth_move, bc),
        np.vstack(resp_sem_log_per_unit(peth_stat, bc)),
        np.vstack(resp_sem_log_per_unit(peth_move, bc)),
    )


def behavior_summary_from_inputs(
    trial_df: pd.DataFrame,
    trial_ts: pd.DataFrame,
    paired_trial_idx: np.ndarray,
) -> BehaviorSummary:
    rewarded = trial_df["rewarded"].to_numpy() == 1
    if "with_choice" in trial_df:
        with_choice = trial_df["with_choice"].to_numpy() == 1
    else:
        with_choice = np.isin(trial_df["response"].to_numpy(), (-1, 1))
    incorrect = with_choice & ~rewarded
    no_choice = ~(rewarded | incorrect)
    outcome_counts = np.array(
        [rewarded.sum(), incorrect.sum(), no_choice.sum()],
        dtype=int,
    )

    paired_trial_df = trial_df.iloc[paired_trial_idx].copy()
    paired_rates = paired_trial_df["stim_rate_vision"].to_numpy(dtype=float)
    valid_rate_mask = np.isfinite(paired_rates)
    paired_rate_values, paired_rate_counts = np.unique(
        paired_rates[valid_rate_mask],
        return_counts=True,
    )

    fixation_durations = paired_trial_df["t_react"].to_numpy(
        dtype=float
    ) - paired_trial_df["t_initiate"].to_numpy(dtype=float)
    fixation_mask = (
        np.isfinite(fixation_durations)
        & (fixation_durations > 0)
        & (fixation_durations < 30)
    )
    fixation_durations = fixation_durations[fixation_mask]

    response_durations = paired_trial_df["t_response"].to_numpy(
        dtype=float
    ) - paired_trial_df["t_react"].to_numpy(dtype=float)
    response_mask = (
        np.isfinite(response_durations)
        & (response_durations > 0)
        & (response_durations < 5)
    )
    response_durations = response_durations[response_mask]

    return BehaviorSummary(
        n_trials=len(trial_df),
        paired_trial_count=len(paired_trial_idx),
        outcome_labels=["rewarded", "incorrect", "no choice"],
        outcome_counts=outcome_counts,
        paired_rate_values=paired_rate_values,
        paired_rate_counts=paired_rate_counts,
        fixation_durations=fixation_durations,
        response_durations=response_durations,
    )


def primary_peak_signature(mean_psth, bc, base_mask, effect_mask):
    resp = mean_psth[effect_mask] - mean_psth[base_mask].mean()
    t_resp = bc[effect_mask]
    if resp.size == 0 or not np.any(np.isfinite(resp)):
        return {
            "n_peaks": 0,
            "lat": np.nan,
            "amp": 0.0,
            "width_ms": np.nan,
            "tail_ratio": np.nan,
            "smooth_tv": np.nan,
            "smooth_curv": np.nan,
        }
    peak_amp = float(np.nanmax(resp))
    if peak_amp <= 0:
        return {
            "n_peaks": 0,
            "lat": np.nan,
            "amp": 0.0,
            "width_ms": np.nan,
            "tail_ratio": np.nan,
            "smooth_tv": float(np.abs(np.diff(resp)).mean())
            if resp.size > 1
            else np.nan,
            "smooth_curv": (
                float(np.abs(np.diff(resp, n=2)).mean()) if resp.size >= 3 else np.nan
            ),
        }

    prom = max(1.5, 0.25 * peak_amp)
    peaks, _ = find_peaks(resp, prominence=prom, distance=2)
    if not len(peaks):
        peaks = np.array([int(np.argmax(resp))])
    p_idx = peaks[np.argmax(resp[peaks])]
    peak_t = float(t_resp[p_idx])
    half_h = 0.5 * float(resp[p_idx])
    above = np.where(resp >= half_h)[0]
    width_ms = (
        float((t_resp[above[-1]] - t_resp[above[0]]) * 1000) if above.size else np.nan
    )
    tail_mask = t_resp >= (peak_t + 0.03)
    tail_mean = float(resp[tail_mask].mean()) if tail_mask.any() else np.nan
    tail_ratio = (
        tail_mean / float(resp[p_idx])
        if np.isfinite(tail_mean) and resp[p_idx] > 0
        else np.nan
    )
    smooth_tv = float(np.abs(np.diff(resp)).mean()) if resp.size > 1 else np.nan
    smooth_curv = float(np.abs(np.diff(resp, n=2)).mean()) if resp.size >= 3 else np.nan
    return {
        "n_peaks": int(len(peaks)),
        "lat": peak_t,
        "amp": float(resp[p_idx]),
        "width_ms": width_ms,
        "tail_ratio": tail_ratio,
        "smooth_tv": smooth_tv,
        "smooth_curv": smooth_curv,
    }


def pick_best_example(
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    used: set[int],
) -> int | None:
    if not len(candidate_idx):
        return None
    order = np.argsort(scores)[::-1]
    for idx in candidate_idx[order]:
        idx = int(idx)
        if idx not in used:
            return idx
    return int(candidate_idx[order[0]])


def shortlist_string(candidate_idx, scores, unit_ids, top_n=5):
    if len(candidate_idx) == 0:
        return "[]"
    order = np.argsort(scores)[::-1][:top_n]
    return "[" + ", ".join(str(unit_ids[int(candidate_idx[i])]) for i in order) + "]"


def finite_or(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else default


def analyze_session(inputs: SessionInputs, stationary_index: int | None = None):
    print(f"Analyzing {inputs.subject} {inputs.session}")
    paired_last_stat, paired_first_move, paired_trial_idx, stationary_label = (
        extract_session_conditioned_anchors(
            inputs.trial_ts,
            stationary_index=stationary_index,
        )
    )
    print(f"  Trials with {stationary_label}+move stims: {len(paired_trial_idx)}")

    behavior = behavior_summary_from_inputs(
        inputs.trial_df, inputs.trial_ts, paired_trial_idx
    )
    paired_trial_df = inputs.trial_df.iloc[paired_trial_idx].copy()

    peth_all, bin_edges, _ = compute_population_peth(
        spike_times_per_unit=inputs.spike_times,
        alignment_times=inputs.first_stim_times,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=10,
    )
    peth_all *= PETH_SCALE_BACK
    _, masks = compute_unit_selectivity(
        peth_all,
        bin_edges,
        unit_ids=inputs.unit_ids,
        base_window=(-0.04, 0.0),
        resp_window=RESP_WINDOW,
        test="wilcoxon",
        correction="bonferroni",
        alpha=0.05,
    )
    exc_idx = np.where(masks["excited"])[0]
    print(f"  Excited units: {len(exc_idx)} / {len(inputs.unit_ids)}")

    peth_stat_all, _, bc = compute_population_peth(
        inputs.spike_times,
        paired_last_stat,
        **PETH_KWARGS,
    )
    peth_move_all, _, _ = compute_population_peth(
        inputs.spike_times,
        paired_first_move,
        **PETH_KWARGS,
    )
    peth_stat_all *= PETH_SCALE_BACK
    peth_move_all *= PETH_SCALE_BACK

    pk_stat_all = resp_per_unit(peth_stat_all, bc)
    pk_move_all = resp_per_unit(peth_move_all, bc)
    sem_stat_all = np.vstack(resp_sem_log_per_unit(peth_stat_all, bc))
    sem_move_all = np.vstack(resp_sem_log_per_unit(peth_move_all, bc))
    print(
        f"  All units ({len(inputs.unit_ids)}): {wilcox_str(pk_stat_all, pk_move_all)}"
    )

    snr_s = snr_per_unit(peth_stat_all, bc)
    snr_m = snr_per_unit(peth_move_all, bc)
    delta_move, _, qvals_move, peak_latencies, stat_trial_matrix, move_trial_matrix = (
        per_unit_move_vs_stat_stats(peth_stat_all, peth_move_all, bc)
    )
    good_snr_both = (snr_s >= 3.0) & (snr_m >= 3.0)
    good_idx = np.where(good_snr_both)[0]
    sig_exc = (qvals_move < 0.05) & (delta_move > 0) & good_snr_both
    sig_supp = (qvals_move < 0.05) & (delta_move < 0) & good_snr_both
    nonsig = (~sig_exc) & (~sig_supp) & good_snr_both
    print(
        "  Move-vs-stat peak-centered classes "
        f"(FDR<0.05, SNR-both): exc={sig_exc.sum()}  "
        f"supp={sig_supp.sum()}  no-effect={nonsig.sum()}"
    )

    base_mask = (bc >= -0.04) & (bc < 0.0)
    effect_mask = (bc >= EFFECT_WINDOW[0]) & (bc < EFFECT_WINDOW[1])

    mean_stat_all = peth_stat_all.mean(axis=1)
    mean_move_all = peth_move_all.mean(axis=1)
    stat_sig = []
    move_sig = []
    for ui in range(len(inputs.unit_ids)):
        stat_sig.append(
            primary_peak_signature(mean_stat_all[ui], bc, base_mask, effect_mask)
        )
        move_sig.append(
            primary_peak_signature(mean_move_all[ui], bc, base_mask, effect_mask)
        )

    canonical_stat = np.array(
        [
            CANONICAL_LATENCY_WINDOW[0] <= sig["lat"] <= CANONICAL_LATENCY_WINDOW[1]
            if np.isfinite(sig["lat"])
            else False
            for sig in stat_sig
        ]
    )
    canonical_move = np.array(
        [
            CANONICAL_LATENCY_WINDOW[0] <= sig["lat"] <= CANONICAL_LATENCY_WINDOW[1]
            if np.isfinite(sig["lat"])
            else False
            for sig in move_sig
        ]
    )
    latency_gap = np.array(
        [
            abs(move_sig[ui]["lat"] - stat_sig[ui]["lat"])
            if np.isfinite(move_sig[ui]["lat"]) and np.isfinite(stat_sig[ui]["lat"])
            else 0.12
            for ui in range(len(inputs.unit_ids))
        ]
    )
    complexity_penalty = np.array(
        [
            6.0 * max(stat_sig[ui]["n_peaks"] - 1, 0)
            + 6.0 * max(move_sig[ui]["n_peaks"] - 1, 0)
            + 0.14 * max(finite_or(move_sig[ui]["width_ms"], 60.0) - 35.0, 0.0)
            + 8.0 * max(finite_or(move_sig[ui]["tail_ratio"], 0.6) - 0.20, 0.0)
            + 1.2 * max(finite_or(move_sig[ui]["smooth_tv"], 3.0), 0.0)
            + 0.25 * max(finite_or(move_sig[ui]["smooth_curv"], 3.0), 0.0)
            + 40.0 * latency_gap[ui]
            for ui in range(len(inputs.unit_ids))
        ]
    )
    canonical_bonus = np.where(canonical_stat & canonical_move, 5.0, 0.0)
    stat_amp = np.array([sig["amp"] for sig in stat_sig], dtype=float)
    move_amp = np.array([sig["amp"] for sig in move_sig], dtype=float)

    exc_scores = (
        2.2 * delta_move
        + 0.7 * move_amp
        + 0.25 * np.minimum(stat_amp, move_amp)
        + canonical_bonus
        - complexity_penalty
        - 0.35 * np.maximum(stat_amp - move_amp, 0.0)
    )
    supp_scores = (
        2.2 * (-delta_move)
        + 0.7 * stat_amp
        + 0.25 * np.minimum(stat_amp, move_amp)
        + canonical_bonus
        - complexity_penalty
        - 0.45 * np.maximum(move_amp, 0.0)
    )
    noeff_scores = (
        0.7 * (stat_amp + move_amp)
        + canonical_bonus
        - complexity_penalty
        - 8.0 * np.abs(delta_move)
    )

    exc_cands = np.where(sig_exc)[0]
    supp_cands = np.where(sig_supp)[0]
    noeff_cands = np.where(nonsig)[0]

    used: set[int] = set()
    ex_excited_idx = pick_best_example(exc_cands, exc_scores[exc_cands], used)
    if ex_excited_idx is None:
        ex_excited_idx = pick_best_example(good_idx, exc_scores[good_idx], used)
    used.add(ex_excited_idx)

    ex_suppressed_idx = pick_best_example(supp_cands, supp_scores[supp_cands], used)
    if ex_suppressed_idx is None:
        ex_suppressed_idx = pick_best_example(good_idx, supp_scores[good_idx], used)
    used.add(ex_suppressed_idx)

    ex_noeffect_idx = pick_best_example(noeff_cands, noeff_scores[noeff_cands], used)
    if ex_noeffect_idx is None:
        ex_noeffect_idx = pick_best_example(good_idx, noeff_scores[good_idx], used)

    trial_rates = paired_trial_df["stim_rate_vision"].to_numpy(dtype=float)
    valid_rate_mask = np.isfinite(trial_rates)
    rate_threshold_hz = RATE_SPLIT_HZ
    low_mask = valid_rate_mask & (trial_rates < rate_threshold_hz)
    high_mask = valid_rate_mask & (trial_rates > rate_threshold_hz)
    print(
        f"  Rate split: threshold={rate_threshold_hz:.1f} Hz  "
        f"low={low_mask.sum()} trials  high={high_mask.sum()} trials"
    )

    pk_stat_low, pk_move_low, sem_stat_low, sem_move_low = split_resp_per_unit(
        inputs.spike_times,
        paired_last_stat[low_mask],
        paired_first_move[low_mask],
        bc,
    )
    pk_stat_high, pk_move_high, sem_stat_high, sem_move_high = split_resp_per_unit(
        inputs.spike_times,
        paired_last_stat[high_mask],
        paired_first_move[high_mask],
        bc,
    )
    print(f"  Low rate:  {wilcox_str(pk_stat_low, pk_move_low)}")
    print(f"  High rate: {wilcox_str(pk_stat_high, pk_move_high)}")

    print(
        "  Example units: "
        f"exc={inputs.unit_ids[ex_excited_idx]}  "
        f"supp={inputs.unit_ids[ex_suppressed_idx]}  "
        f"no-effect={inputs.unit_ids[ex_noeffect_idx]}"
    )
    if inputs.subject == "GRB006":
        print(
            "  GRB006 shortlist: "
            f"exc={shortlist_string(exc_cands, exc_scores[exc_cands], inputs.unit_ids)}  "
            f"supp={shortlist_string(supp_cands, supp_scores[supp_cands], inputs.unit_ids)}  "
            f"no-effect={shortlist_string(noeff_cands, noeff_scores[noeff_cands], inputs.unit_ids)}"
        )

    return SessionAnalysis(
        subject=inputs.subject,
        session=inputs.session,
        unit_ids=inputs.unit_ids,
        spike_times=inputs.spike_times,
        unit_depth_um=inputs.unit_depth_um,
        behavior=behavior,
        paired_trial_df=paired_trial_df,
        bin_centers=bc,
        paired_last_stat=paired_last_stat,
        paired_first_move=paired_first_move,
        stationary_label=stationary_label,
        paired_trial_rates=trial_rates,
        peth_stat_all=peth_stat_all,
        peth_move_all=peth_move_all,
        stat_trial_matrix=stat_trial_matrix,
        move_trial_matrix=move_trial_matrix,
        example_indices=[ex_excited_idx, ex_suppressed_idx, ex_noeffect_idx],
        peak_latencies=peak_latencies,
        delta_move=delta_move,
        qvals_move=qvals_move,
        pk_stat_all=pk_stat_all,
        pk_move_all=pk_move_all,
        pk_stat_low=pk_stat_low,
        pk_move_low=pk_move_low,
        pk_stat_high=pk_stat_high,
        pk_move_high=pk_move_high,
        sem_stat_all=sem_stat_all,
        sem_move_all=sem_move_all,
        sem_stat_low=sem_stat_low,
        sem_move_low=sem_move_low,
        sem_stat_high=sem_stat_high,
        sem_move_high=sem_move_high,
        rate_threshold_hz=rate_threshold_hz,
        low_count=int(low_mask.sum()),
        high_count=int(high_mask.sum()),
        paired_trial_count=len(paired_last_stat),
        n_units=len(inputs.unit_ids),
    )


def analyze_comparison(inputs: SessionInputs, comparison: ComparisonDef):
    paired_stat, paired_move, paired_trial_idx, stationary_label = (
        extract_comparison_anchors(
            inputs,
            comparison,
        )
    )
    paired_trial_df = inputs.trial_df.iloc[paired_trial_idx].copy()
    _empty2 = np.zeros((2, 0))
    if len(paired_stat) == 0 or len(paired_move) == 0:
        return {
            "subject": inputs.subject,
            "comparison": comparison,
            "stationary_label": stationary_label,
            "pk_stat_all": np.array([]),
            "pk_move_all": np.array([]),
            "pk_stat_low": np.array([]),
            "pk_move_low": np.array([]),
            "pk_stat_high": np.array([]),
            "pk_move_high": np.array([]),
            "sem_stat_all": _empty2,
            "sem_move_all": _empty2,
            "sem_stat_low": _empty2,
            "sem_move_low": _empty2,
            "sem_stat_high": _empty2,
            "sem_move_high": _empty2,
            "low_count": 0,
            "high_count": 0,
            "paired_trial_count": 0,
            "n_units": len(inputs.unit_ids),
        }

    peth_stat_all, _, bc = compute_population_peth(
        inputs.spike_times,
        paired_stat,
        **PETH_KWARGS,
    )
    peth_move_all, _, _ = compute_population_peth(
        inputs.spike_times,
        paired_move,
        **PETH_KWARGS,
    )
    peth_stat_all *= PETH_SCALE_BACK
    peth_move_all *= PETH_SCALE_BACK

    pk_stat_all = resp_per_unit(peth_stat_all, bc)
    pk_move_all = resp_per_unit(peth_move_all, bc)
    sem_stat_all = np.vstack(resp_sem_log_per_unit(peth_stat_all, bc))
    sem_move_all = np.vstack(resp_sem_log_per_unit(peth_move_all, bc))

    trial_rates = paired_trial_df["stim_rate_vision"].to_numpy(dtype=float)
    valid_rate_mask = np.isfinite(trial_rates)
    low_mask = valid_rate_mask & (trial_rates < RATE_SPLIT_HZ)
    high_mask = valid_rate_mask & (trial_rates > RATE_SPLIT_HZ)
    pk_stat_low, pk_move_low, sem_stat_low, sem_move_low = split_resp_per_unit(
        inputs.spike_times,
        paired_stat[low_mask],
        paired_move[low_mask],
        bc,
    )
    pk_stat_high, pk_move_high, sem_stat_high, sem_move_high = split_resp_per_unit(
        inputs.spike_times,
        paired_stat[high_mask],
        paired_move[high_mask],
        bc,
    )
    return {
        "subject": inputs.subject,
        "comparison": comparison,
        "stationary_label": stationary_label,
        "pk_stat_all": pk_stat_all,
        "pk_move_all": pk_move_all,
        "pk_stat_low": pk_stat_low,
        "pk_move_low": pk_move_low,
        "pk_stat_high": pk_stat_high,
        "pk_move_high": pk_move_high,
        "sem_stat_all": sem_stat_all,
        "sem_move_all": sem_move_all,
        "sem_stat_low": sem_stat_low,
        "sem_move_low": sem_move_low,
        "sem_stat_high": sem_stat_high,
        "sem_move_high": sem_move_high,
        "low_count": int(low_mask.sum()),
        "high_count": int(high_mask.sum()),
        "paired_trial_count": len(paired_stat),
        "n_units": len(inputs.unit_ids),
    }


def plot_behavior_row(fig, gs, row_start, result: SessionAnalysis):
    axes = [
        fig.add_subplot(gs[row_start, 0]),
        fig.add_subplot(gs[row_start, 1]),
        fig.add_subplot(gs[row_start, 2]),
    ]
    behavior = result.behavior

    outcome_colors = ["seagreen", "tomato", "#6b7280"]
    outcome_frac = behavior.outcome_counts / max(behavior.n_trials, 1)
    bars = axes[0].bar(
        behavior.outcome_labels,
        outcome_frac,
        color=outcome_colors,
        width=0.7,
    )
    for bar, count, frac in zip(bars, behavior.outcome_counts, outcome_frac):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{count}\n{frac:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylim(0, max(1.0, outcome_frac.max() + 0.18))
    axes[0].set_ylabel("fraction of trials")
    axes[0].set_title(f"Outcomes (n={behavior.n_trials})", **TITLE_KW)

    if len(behavior.paired_rate_values):
        rate_colors = []
        for rate in behavior.paired_rate_values:
            if rate < RATE_SPLIT_HZ:
                rate_colors.append("steelblue")
            elif rate > RATE_SPLIT_HZ:
                rate_colors.append("darkorange")
            else:
                rate_colors.append("0.55")
        axes[1].bar(
            behavior.paired_rate_values,
            behavior.paired_rate_counts,
            color=rate_colors,
            width=1.2,
            edgecolor="black",
            linewidth=0.4,
        )
        axes[1].axvline(RATE_SPLIT_HZ, color="0.25", linestyle="--", lw=0.8)
        axes[1].set_xticks(behavior.paired_rate_values)
        axes[1].set_xlabel("stim rate (Hz)")
        axes[1].set_ylabel("paired trials")
        axes[1].set_title(
            f"Paired trial rates (n={behavior.paired_trial_count})",
            **TITLE_KW,
        )
    else:
        axes[1].text(0.5, 0.5, "No paired rates", ha="center", va="center")
        axes[1].set_title("Paired trial rates", **TITLE_KW)

    timing_data = []
    timing_labels = []
    timing_colors = []
    if len(behavior.fixation_durations):
        timing_data.append(behavior.fixation_durations)
        timing_labels.append("fixation")
        timing_colors.append("steelblue")
    if len(behavior.response_durations):
        timing_data.append(behavior.response_durations)
        timing_labels.append("response")
        timing_colors.append("darkorange")
    if timing_data:
        box = axes[2].boxplot(
            timing_data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], timing_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for med in box["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)
        axes[2].set_xticklabels(timing_labels)
        axes[2].set_ylabel("seconds")
        axes[2].set_title("Timing (fixation / response)", **TITLE_KW)
        medians = [np.median(vals) for vals in timing_data]
        ymax = max(float(np.max(vals)) for vals in timing_data)
        for i, med in enumerate(medians, start=1):
            axes[2].text(i, ymax * 1.03, f"med={med:.2f}s", ha="center", fontsize=8)
        axes[2].set_ylim(0, ymax * 1.18)
    else:
        axes[2].text(0.5, 0.5, "No timing data", ha="center", va="center")
        axes[2].set_title("Timing", **TITLE_KW)

    axes[0].text(
        -0.38,
        1.28,
        f"{result.subject}  {result.session}\nBehavior",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def plot_example_row(fig, gs, row_start, result: SessionAnalysis):
    labels = ["Move-excited", "Move-suppressed", "No effect"]
    row_axes = []
    stat_all = result.peth_stat_all
    move_all = result.peth_move_all
    for col, (ui, label) in enumerate(zip(result.example_indices, labels)):
        ax = fig.add_subplot(gs[row_start, col])
        row_axes.append(ax)
        ps = stat_all[ui]
        pm = move_all[ui]
        ms = ps.mean(axis=0)
        ss = sem(ps, axis=0)
        mm = pm.mean(axis=0)
        sm = sem(pm, axis=0)
        ax.plot(
            result.bin_centers,
            ms,
            color="steelblue",
            lw=1.6,
            label=f"{result.stationary_label} (n={len(result.paired_last_stat)})",
        )
        ax.fill_between(
            result.bin_centers, ms - ss, ms + ss, alpha=0.25, color="steelblue"
        )
        ax.plot(
            result.bin_centers,
            mm,
            color="darkorange",
            lw=1.6,
            label=f"first movement (n={len(result.paired_first_move)})",
        )
        ax.fill_between(
            result.bin_centers, mm - sm, mm + sm, alpha=0.25, color="darkorange"
        )
        ax.axvline(0, color="gray", linestyle="--", lw=0.8)
        ax.set_xlabel("Time from stim onset (s)")
        ax.set_ylabel("sp/s")
        uid = result.unit_ids[ui]
        dm = result.delta_move[ui]
        pt = result.peak_latencies[ui]
        ax.set_title(
            f"{label}  unit {uid}\nΔsp/s={dm:+.1f}, peak={pt * 1e3:.0f} ms",
            **TITLE_KW,
        )
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    row_axes[0].text(
        -0.38,
        1.28,
        f"{result.subject}  {result.session}\nExample units",
        transform=row_axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def scatter_log(ax, pk_s, pk_m, title, sem_s=None, sem_m=None):
    if len(pk_s) == 0 or len(pk_m) == 0:
        ax.text(0.5, 0.55, "No trials", ha="center", va="center", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Stationary (sp/s)")
        ax.set_ylabel("Movement (sp/s)")
        ax.set_title(title, **TITLE_KW)
        return
    all_pos = np.concatenate([pk_s, pk_m])
    all_pos = all_pos[all_pos > 0]
    lim_lo = max(1.0, np.percentile(all_pos, 2)) if all_pos.size else 1.0
    lim_hi = np.percentile(all_pos, 99) * 1.5 if all_pos.size else 100.0
    ax.errorbar(
        pk_s,
        pk_m,
        xerr=sem_s,
        yerr=sem_m,
        fmt="o",
        color="k",
        ms=3,
        alpha=0.45,
        elinewidth=0.6,
        ecolor=(0, 0, 0, 0.06),
        zorder=3,
    )
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary (sp/s)")
    ax.set_ylabel("Movement (sp/s)")
    ax.set_title(title, **TITLE_KW)


def add_mean_marker(ax, pk_s, pk_m, color, label, sem_s=None, sem_m=None):
    if len(pk_s) == 0 or len(pk_m) == 0:
        return
    import matplotlib.colors as mcolors

    mean_x = float(np.mean(pk_s))
    mean_y = float(np.mean(pk_m))
    # Population-level SEM across units in log space for the mean marker error bars
    xerr = ym_err = None
    if sem_s is not None:
        log_s = np.log(np.clip(pk_s, 1e-3, None))
        pop_sem_s = log_s.std() / np.sqrt(len(pk_s))
        xerr = np.array(
            [[mean_x * (1 - np.exp(-pop_sem_s))], [mean_x * (np.exp(pop_sem_s) - 1)]]
        )
    if sem_m is not None:
        log_m = np.log(np.clip(pk_m, 1e-3, None))
        pop_sem_m = log_m.std() / np.sqrt(len(pk_m))
        ym_err = np.array(
            [[mean_y * (1 - np.exp(-pop_sem_m))], [mean_y * (np.exp(pop_sem_m) - 1)]]
        )
    r, g, b = mcolors.to_rgb(color)
    ax.errorbar(
        mean_x,
        mean_y,
        xerr=xerr,
        yerr=ym_err,
        fmt="o",
        ms=8,
        mfc=color,
        mec="white",
        mew=0.8,
        alpha=0.85,
        elinewidth=1.2,
        ecolor=(r, g, b, 0.20),
        capsize=2.5,
        zorder=5,
        label=label,
    )


def overlay_scatter_log(ax, panel_specs, title):
    # panel_specs: list of (pk_s, pk_m, color, label[, sem_s, sem_m])
    nonempty = [
        np.concatenate([pk_s, pk_m])
        for pk_s, pk_m, *_ in panel_specs
        if len(pk_s) and len(pk_m)
    ]
    if not nonempty:
        ax.text(0.5, 0.55, "No trials", ha="center", va="center", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Stationary (sp/s)")
        ax.set_ylabel("Movement (sp/s)")
        ax.set_title(title, **TITLE_KW)
        return

    import matplotlib.colors as mcolors

    all_pos = np.concatenate(nonempty)
    all_pos = all_pos[all_pos > 0]
    lim_lo = max(1.0, np.percentile(all_pos, 2)) if all_pos.size else 1.0
    lim_hi = np.percentile(all_pos, 99) * 1.5 if all_pos.size else 100.0
    for spec in panel_specs:
        pk_s, pk_m, color, label = spec[:4]
        sem_s = spec[4] if len(spec) > 4 else None
        sem_m = spec[5] if len(spec) > 5 else None
        if len(pk_s) == 0 or len(pk_m) == 0:
            continue
        r, g, b = mcolors.to_rgb(color)
        ax.errorbar(
            pk_s,
            pk_m,
            xerr=sem_s,
            yerr=sem_m,
            fmt="o",
            color=color,
            ms=3,
            alpha=0.25,
            elinewidth=0.5,
            ecolor=(r, g, b, 0.04),
            zorder=2,
        )
        add_mean_marker(
            ax, pk_s, pk_m, color=color, label=label, sem_s=pk_s, sem_m=pk_m
        )

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary (sp/s)")
    ax.set_ylabel("Movement (sp/s)")
    ax.set_title(title, **TITLE_KW)


def plot_scatter_row(fig, gs, row_start, result: SessionAnalysis):
    axes = [
        fig.add_subplot(gs[row_start, 0]),
        fig.add_subplot(gs[row_start, 1]),
        fig.add_subplot(gs[row_start, 2]),
    ]
    scatter_log(
        axes[0],
        result.pk_stat_all,
        result.pk_move_all,
        f"All rates (units={result.n_units}, trials={result.paired_trial_count})",
        sem_s=result.sem_stat_all,
        sem_m=result.sem_move_all,
    )
    scatter_log(
        axes[1],
        result.pk_stat_low,
        result.pk_move_low,
        f"Low rate (<{result.rate_threshold_hz:.1f} Hz, trials={result.low_count})",
        sem_s=result.sem_stat_low,
        sem_m=result.sem_move_low,
    )
    scatter_log(
        axes[2],
        result.pk_stat_high,
        result.pk_move_high,
        f"High rate (>{result.rate_threshold_hz:.1f} Hz, trials={result.high_count})",
        sem_s=result.sem_stat_high,
        sem_m=result.sem_move_high,
    )
    axes[0].text(
        -0.38,
        1.22,
        f"{result.subject} scatters",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def build_overlay_scatter_figure(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    subject_colors = ["#1f77b4", "#d62728"]
    labels = [
        (
            "All rates",
            lambda r: (
                r.pk_stat_all,
                r.pk_move_all,
                r.paired_trial_count,
                r.sem_stat_all,
                r.sem_move_all,
            ),
        ),
        (
            "Low rate",
            lambda r: (
                r.pk_stat_low,
                r.pk_move_low,
                r.low_count,
                r.sem_stat_low,
                r.sem_move_low,
            ),
        ),
        (
            "High rate",
            lambda r: (
                r.pk_stat_high,
                r.pk_move_high,
                r.high_count,
                r.sem_stat_high,
                r.sem_move_high,
            ),
        ),
    ]

    for col_idx, (ax, (panel_name, getter)) in enumerate(zip(axes, labels)):
        panel_specs = []
        count_lines = []
        for result, color in zip(results, subject_colors):
            pk_s, pk_m, n_trials, sem_s, sem_m = getter(result)
            panel_specs.append((pk_s, pk_m, color, result.subject, sem_s, sem_m))
            count_lines.append(
                f"{result.subject}: units={len(pk_s)}, trials={n_trials}"
            )
        overlay_scatter_log(ax, panel_specs, panel_name)
        if col_idx != 0:
            ax.set_ylabel("")
        ax.text(
            0.97,
            0.03,
            "\n".join(count_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, lw=0.0),
        )

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Cross-subject overlay: session-specific stationary anchor vs first movement",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.88, wspace=0.28)
    return fig


def build_overlay_summary_figure(summary_rows):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    subject_colors = ["#1f77b4", "#d62728"]
    panel_defs = [
        (
            "All rates",
            lambda r: (
                r["pk_stat_all"],
                r["pk_move_all"],
                r["paired_trial_count"],
                r["sem_stat_all"],
                r["sem_move_all"],
            ),
        ),
        (
            "Low rate",
            lambda r: (
                r["pk_stat_low"],
                r["pk_move_low"],
                r["low_count"],
                r["sem_stat_low"],
                r["sem_move_low"],
            ),
        ),
        (
            "High rate",
            lambda r: (
                r["pk_stat_high"],
                r["pk_move_high"],
                r["high_count"],
                r["sem_stat_high"],
                r["sem_move_high"],
            ),
        ),
    ]

    for row_idx, (comparison, row_results) in enumerate(summary_rows):
        for col_idx, (ax, (panel_name, getter)) in enumerate(
            zip(axes[row_idx], panel_defs)
        ):
            panel_specs = []
            count_lines = []
            for result, color in zip(row_results, subject_colors):
                pk_s, pk_m, n_trials, sem_s, sem_m = getter(result)
                panel_specs.append((pk_s, pk_m, color, result["subject"], sem_s, sem_m))
                count_lines.append(
                    f"{result['subject']}: units={len(pk_s)}, trials={n_trials}"
                )
            overlay_scatter_log(ax, panel_specs, panel_name)
            if row_idx != 2:
                ax.set_xlabel("")
            if col_idx != 0:
                ax.set_ylabel("")
            ax.text(
                0.04,
                0.96,
                "\n".join(count_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                bbox=dict(
                    boxstyle="round,pad=0.22", facecolor="white", alpha=0.9, lw=0.0
                ),
            )

        axes[row_idx, 0].text(
            -0.34,
            1.22,
            comparison.row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
        axes[row_idx, 0].text(
            -0.34,
            1.10,
            comparison.description,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=7,
            clip_on=False,
        )

    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Cross-subject locomotion summary with training-regime timing controls\n"
        "GRB006 fixed 0.5 s wait; GRB058 short-wait training session",
        fontsize=12,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.14, right=0.98, bottom=0.06, top=0.88, wspace=0.30, hspace=0.30
    )
    return fig


def build_depth_bins(depth_um, width_um=DEPTH_BIN_WIDTH_UM):
    lo = np.floor(np.nanmin(depth_um) / width_um) * width_um
    hi = np.ceil(np.nanmax(depth_um) / width_um) * width_um
    edges = np.arange(lo, hi + width_um, width_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def depth_bin_summary(depth_um, values, width_um=DEPTH_BIN_WIDTH_UM):
    valid = np.isfinite(depth_um) & np.isfinite(values)
    if not valid.any():
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges, centers = build_depth_bins(depth_um[valid], width_um=width_um)
    bin_idx = np.digitize(depth_um[valid], edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(centers) - 1)
    med = np.full(len(centers), np.nan, dtype=float)
    q25 = np.full(len(centers), np.nan, dtype=float)
    q75 = np.full(len(centers), np.nan, dtype=float)
    counts = np.zeros(len(centers), dtype=int)
    vals = values[valid]
    for bi in range(len(centers)):
        in_bin = bin_idx == bi
        counts[bi] = int(in_bin.sum())
        if not counts[bi]:
            continue
        med[bi] = float(np.median(vals[in_bin]))
        q25[bi] = float(np.percentile(vals[in_bin], 25))
        q75[bi] = float(np.percentile(vals[in_bin], 75))
    keep = counts > 0
    return centers[keep], med[keep], q25[keep], q75[keep]


def build_depth_figure(results):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = ["#1f77b4", "#d62728"]

    all_depths = [
        result.unit_depth_um[np.isfinite(result.unit_depth_um)] for result in results
    ]
    all_depths = [arr for arr in all_depths if len(arr)]
    global_edges, global_centers = build_depth_bins(np.concatenate(all_depths))
    bar_w = DEPTH_BIN_WIDTH_UM * 0.38

    ax = axs[0, 0]
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    for result, color in zip(results, colors):
        valid = np.isfinite(result.unit_depth_um)
        ax.scatter(
            result.unit_depth_um[valid],
            result.delta_move[valid],
            s=20,
            alpha=0.18,
            color=color,
            label=result.subject,
        )
    ax.set_xlabel("Depth (um)")
    ax.set_ylabel("Delta move - stat (sp/s)")
    ax.set_title("Per-unit locomotion effect vs depth", **TITLE_KW)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axs[0, 1]
    for result, color in zip(results, colors):
        centers, med, q25, q75 = depth_bin_summary(
            result.unit_depth_um, result.delta_move
        )
        if len(centers) == 0:
            continue
        ax.errorbar(
            centers,
            med,
            yerr=np.vstack([med - q25, q75 - med]),
            fmt="o-",
            color=color,
            lw=1.4,
            ms=4,
            label=result.subject,
        )
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_xlabel("Depth bin center (um)")
    ax.set_ylabel("Delta move - stat (sp/s)")
    ax.set_title(
        f"Median locomotion effect by depth ({int(DEPTH_BIN_WIDTH_UM)} um bins)",
        **TITLE_KW,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axs[1, 0]
    for i, (result, color) in enumerate(zip(results, colors)):
        valid_depth = result.unit_depth_um[np.isfinite(result.unit_depth_um)]
        counts, _ = np.histogram(valid_depth, bins=global_edges)
        offset = (-0.5 + i) * bar_w
        ax.bar(
            global_centers + offset,
            counts,
            width=bar_w,
            color=color,
            alpha=0.7,
            label=result.subject,
        )
    ax.set_xlabel("Depth bin center (um)")
    ax.set_ylabel("Unit count")
    ax.set_title("Depth distribution of analyzable units", **TITLE_KW)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axs[1, 1]
    summary_rows = []
    labels = []
    for result in results:
        valid = np.isfinite(result.unit_depth_um)
        top = valid & (
            result.unit_depth_um <= np.nanmedian(result.unit_depth_um[valid])
        )
        bottom = valid & (
            result.unit_depth_um > np.nanmedian(result.unit_depth_um[valid])
        )
        summary_rows.append(
            [
                np.nanmean(result.delta_move[top]) if top.any() else np.nan,
                np.nanmean(result.delta_move[bottom]) if bottom.any() else np.nan,
            ]
        )
        labels.append(result.subject)
    summary = np.asarray(summary_rows, dtype=float)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, summary[:, 0], width=0.36, color="0.55", label="shallower half")
    ax.bar(x + 0.18, summary[:, 1], width=0.36, color="0.2", label="deeper half")
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean delta move - stat (sp/s)")
    ax.set_title("Shallower vs deeper half within subject", **TITLE_KW)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Depth analysis of locomotion enhancement (GRB006 vs GRB058)",
        fontsize=12,
    )
    return fig


def build_figure(results):
    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(6, 3, hspace=0.95, wspace=0.35)
    plot_behavior_row(fig, gs, 0, results[0])
    plot_example_row(fig, gs, 1, results[0])
    plot_scatter_row(fig, gs, 2, results[0])
    plot_behavior_row(fig, gs, 3, results[1])
    plot_example_row(fig, gs, 4, results[1])
    plot_scatter_row(fig, gs, 5, results[1])
    fig.suptitle(
        "Behavior + stationary-anchor vs first movement\n"
        "GRB006 uses 3rd stationary; GRB058 uses last stationary",
        fontsize=12,
        y=0.995,
    )
    return fig


def save_pdf_and_png(fig, pdf_path):
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=PNG_DPI)
    print(f"\nFigure saved: {pdf_path}")
    print(f"Figure saved: {png_path}")


def main():
    grb006 = load_local_spikes_db_behavior(
        subject="GRB006",
        session="20240821_121447",
        trial_ts_path=LOCAL_TRIAL_TS,
        spike_times_path=LOCAL_SPIKE_TIMES,
    )
    grb058 = load_db_session(
        subject="GRB058",
        session="20260312_134952",
    )

    results = [
        analyze_session(
            grb006,
            stationary_index=MAIN_ANCHOR_CONFIG["GRB006"]["stationary_index"],
        ),
        analyze_session(
            grb058,
            stationary_index=MAIN_ANCHOR_CONFIG["GRB058"]["stationary_index"],
        ),
    ]
    fig = build_figure(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pdf_and_png(fig, OUT_PATH)
    plt.close(fig)

    fig_overlay = build_overlay_scatter_figure(results)
    save_pdf_and_png(fig_overlay, OUT_PATH_OVERLAY)
    plt.close(fig_overlay)

    summary_comparisons = [
        ComparisonDef(
            key="last_vs_first",
            row_label="Row 1: baseline comparison",
            description="Both subjects use last stationary vs first movement.",
            mode="indexed_stationary",
            stationary_index_by_subject={"GRB006": None, "GRB058": None},
        ),
        ComparisonDef(
            key="timing_matched",
            row_label="Row 2: training-timing correction",
            description="GRB006 uses 2nd stationary; GRB058 uses last stationary.",
            mode="indexed_stationary",
            stationary_index_by_subject={"GRB006": 1, "GRB058": None},
        ),
        ComparisonDef(
            key="offset_500_700ms",
            row_label="Row 3: strict 0.5-0.7 s control",
            description="Closest offset-matched stationary and movement pulses in 0.5-0.7 s.",
            mode="offset_matched",
            offset_window_s=SUMMARY_OFFSET_WINDOW_S,
            wiggle_room_s=SUMMARY_OFFSET_WIGGLE_S,
        ),
    ]
    summary_rows = [
        (
            comparison,
            [
                analyze_comparison(grb006, comparison),
                analyze_comparison(grb058, comparison),
            ],
        )
        for comparison in summary_comparisons
    ]
    fig_overlay_summary = build_overlay_summary_figure(summary_rows)
    save_pdf_and_png(fig_overlay_summary, OUT_PATH_OVERLAY_SUMMARY)
    plt.close(fig_overlay_summary)

    fig_depth = build_depth_figure(results)
    save_pdf_and_png(fig_depth, OUT_PATH_DEPTH)
    plt.close(fig_depth)


if __name__ == "__main__":
    main()
