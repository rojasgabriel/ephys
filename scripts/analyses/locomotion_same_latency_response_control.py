"""Legacy shared-peak locomotion control for GRB006 and GRB058.

This is the stricter locomotion control/comparison surface. It writes three PDF outputs:
  - figures/locomotion/scatter.pdf
  - figures/locomotion/overlay.pdf
  - figures/locomotion/timing_ctrl.pdf

Main comparison:
  - GRB006: 3rd stationary vs first movement
  - GRB058: last stationary vs first movement

Quality gate:
  - baseline-only soft gate; units must rise above baseline in at least one condition
"""

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

from ephys.src.config.locomotion import (
    BASELINE_WINDOW,
    PEAK_HALF_WINDOW_S,
    PETH_KWARGS,
    QVAL_ALPHA,
    RATE_SPLIT_HZ,
    RESP_WINDOW,
)
from ephys.src.utils.grb006_data import load_grb006_hybrid_session_inputs
from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table
from ephys.src.utils.unit_metrics import fetch_spike_duration_ms
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
    compute_unit_selectivity,
    find_unique_cross_trial_offset_pairs,
)

# RESP_WINDOW from config is the single window used for SNR + effect tests +
# peak-centered measurements. The previous RESP_WINDOW=(0.0, 0.12) is now
# unified with RESP_WINDOW.

CANONICAL_LATENCY_WINDOW = (0.015, 0.070)
TITLE_KW = dict(fontsize=11, pad=3)
MAIN_ANCHOR_CONFIG = {
    "GRB006": {"stationary_index": 2},
    "GRB058": {"stationary_index": None},
}
SUMMARY_OFFSET_WINDOW_S = (0.5, 0.7)
SUMMARY_OFFSET_WIGGLE_S = 0.1
FIGURE_DIR = Path("/Users/gabriel/lib/ephys/figures/locomotion")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = FIGURE_DIR / "scatter.pdf"
OUT_PATH_OVERLAY = FIGURE_DIR / "overlay.pdf"
OUT_PATH_OVERLAY_WAVEFORM = FIGURE_DIR / "overlay_waveform_split.pdf"
OUT_PATH_OVERLAY_SUMMARY = FIGURE_DIR / "timing_ctrl.pdf"
NARROW_BROAD_MS = 0.4


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
    spike_duration_ms: np.ndarray
    first_stim_times: np.ndarray
    trial_df: pd.DataFrame
    trial_ts: pd.DataFrame


@dataclass
class SessionAnalysis:
    subject: str
    session: str
    unit_ids: list[int]
    spike_times: list[np.ndarray]
    spike_duration_ms: np.ndarray
    behavior: BehaviorSummary
    bin_centers: np.ndarray
    paired_last_stat: np.ndarray
    paired_first_move: np.ndarray
    stationary_label: str
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


@dataclass(frozen=True)
class ComparisonPairs:
    stat_times: np.ndarray
    move_times: np.ndarray
    stat_trial_idx: np.ndarray
    move_trial_idx: np.ndarray
    stationary_label: str


def ordinal_label(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def extract_session_conditioned_anchors(
    trial_ts: pd.DataFrame, stationary_index: int | None
) -> ComparisonPairs:
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
    return ComparisonPairs(
        stat_times=paired_stat,
        move_times=paired_first_move,
        stat_trial_idx=paired_trial_idx,
        move_trial_idx=paired_trial_idx,
        stationary_label=stationary_label,
    )


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
) -> ComparisonPairs:
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
        return ComparisonPairs(
            stat_times=np.array([]),
            move_times=np.array([]),
            stat_trial_idx=np.array([], dtype=int),
            move_trial_idx=np.array([], dtype=int),
            stationary_label="offset-matched 0.5-0.7 s",
        )

    matched_pairs_df = find_unique_cross_trial_offset_pairs(
        stims_offset_df,
        wiggle_room=wiggle_room_s,
    )
    if matched_pairs_df.empty:
        return ComparisonPairs(
            stat_times=np.array([]),
            move_times=np.array([]),
            stat_trial_idx=np.array([], dtype=int),
            move_trial_idx=np.array([], dtype=int),
            stationary_label="offset-matched 0.5-0.7 s",
        )

    label = f"offset-matched {lo:.1f}-{hi:.1f} s"
    return ComparisonPairs(
        stat_times=matched_pairs_df["stat_stim_time"].to_numpy(dtype=float),
        move_times=matched_pairs_df["move_stim_time"].to_numpy(dtype=float),
        stat_trial_idx=matched_pairs_df["stat_trial_idx"].to_numpy(dtype=int),
        move_trial_idx=matched_pairs_df["move_trial_idx"].to_numpy(dtype=int),
        stationary_label=label,
    )


def extract_comparison_anchors(
    inputs: SessionInputs,
    comparison: ComparisonDef,
) -> ComparisonPairs:
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


def load_local_spikes_db_behavior(
    subject: str,
    session: str,
):
    print(f"\nLoading hybrid session: {subject} {session}")
    unit_ids, spike_times, trial_df, trial_ts = load_grb006_hybrid_session_inputs()
    first_stim_times = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim_times = first_stim_times[np.isfinite(first_stim_times)]
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired trials: {len(trial_ts)}  First stims: {len(first_stim_times)}"
    )
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        spike_duration_ms=spike_duration_ms,
        first_stim_times=first_stim_times,
        trial_df=trial_df,
        trial_ts=trial_ts,
    )


def load_db_session(subject: str, session: str):
    from ephys.src.utils.utils_IO import fetch_good_units

    print(f"\nLoading DB session: {subject} {session}")
    trial_df, trial_ts, first_stim_times = load_db_behavior(subject, session)
    st_per_unit = fetch_good_units(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired trials: {len(trial_ts)}  First stims: {len(first_stim_times)}"
    )
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        spike_duration_ms=spike_duration_ms,
        first_stim_times=first_stim_times,
        trial_df=trial_df,
        trial_ts=trial_ts,
    )


def resp_per_unit(peth, bc, window=RESP_WINDOW):
    """Mean firing rate (sp/s) in the response window, averaged across trials."""
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def baseline_per_unit(peth, bc, window=BASELINE_WINDOW):
    """Mean firing rate (sp/s) in the baseline window, averaged across trials."""
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def above_baseline_mask(pk_stat, pk_move, bl_stat, bl_move):
    """True for units with response > baseline in at least one condition.

    Soft gate against the "suppressed in both, less suppressed in move →
    falsely called enhanced" failure mode.
    """
    if len(pk_stat) == 0:
        return np.array([], dtype=bool)
    return (pk_stat - bl_stat > 0) | (pk_move - bl_move > 0)


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


def wilcox_str(a, b, gate_mask=None):
    """Wilcoxon p + % above diagonal, optionally restricted by `gate_mask`.

    `gate_mask` should be a boolean array of length len(a)/len(b). If provided,
    only units where gate_mask=True are included (e.g. units above baseline).
    """
    if len(a) == 0 or len(b) == 0:
        return "no trials"
    if gate_mask is not None:
        a = a[gate_mask]
        b = b[gate_mask]
        if len(a) == 0:
            return "no units pass baseline gate"
    _, p = wilcoxon(a, b)
    pct = 100 * (b > a).mean()
    median_delta = float(np.median(b - a))
    return f"{pct:.0f}% above diag  Δmedian={median_delta:+.2f} sp/s  p={p:.3f}  n={len(a)}"


def per_unit_move_vs_stat_stats(
    peth_stat,
    peth_move,
    bc,
    window=RESP_WINDOW,
    half_window_s=PEAK_HALF_WINDOW_S,
):
    effect_mask = (bc >= window[0]) & (bc < window[1])
    if not effect_mask.any():
        raise ValueError("RESP_WINDOW does not overlap available bins.")

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
    qvals = multipletests(pvals, alpha=QVAL_ALPHA, method="fdr_bh")[1]
    return delta, pvals, qvals, peak_latencies, stat_trials, move_trials


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
    return (
        resp_per_unit(peth_stat, bc),
        resp_per_unit(peth_move, bc),
        np.vstack(resp_sem_log_per_unit(peth_stat, bc)),
        np.vstack(resp_sem_log_per_unit(peth_move, bc)),
    )


def pair_rate_masks(
    stat_rates: np.ndarray,
    move_rates: np.ndarray,
    rate_threshold_hz: float = RATE_SPLIT_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(stat_rates) & np.isfinite(move_rates)
    low = valid & (stat_rates < rate_threshold_hz) & (move_rates < rate_threshold_hz)
    high = valid & (stat_rates > rate_threshold_hz) & (move_rates > rate_threshold_hz)
    mixed = valid & ~(low | high)
    return low, high, mixed


def behavior_summary_from_inputs(
    trial_df: pd.DataFrame,
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


def analyze_session(
    inputs: SessionInputs,
    stationary_index: int | None = None,
):
    print(f"Analyzing {inputs.subject} {inputs.session}")
    pairs = extract_session_conditioned_anchors(
        inputs.trial_ts,
        stationary_index=stationary_index,
    )
    print(
        f"  Trials with {pairs.stationary_label}+move stims: "
        f"{len(pairs.move_trial_idx)}"
    )

    behavior = behavior_summary_from_inputs(inputs.trial_df, pairs.move_trial_idx)
    paired_trial_df = inputs.trial_df.iloc[pairs.move_trial_idx].copy()

    peth_all, bin_edges, _ = compute_population_peth(
        spike_times_per_unit=inputs.spike_times,
        alignment_times=inputs.first_stim_times,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=10,
    )
    _, masks = compute_unit_selectivity(
        peth_all,
        bin_edges,
        unit_ids=inputs.unit_ids,
        base_window=BASELINE_WINDOW,
        resp_window=RESP_WINDOW,
        test="wilcoxon",
        correction="bonferroni",
        alpha=QVAL_ALPHA,
    )
    exc_idx = np.where(masks["excited"])[0]
    print(f"  Excited units: {len(exc_idx)} / {len(inputs.unit_ids)}")

    peth_stat_all, _, bc = compute_population_peth(
        inputs.spike_times,
        pairs.stat_times,
        **PETH_KWARGS,
    )
    peth_move_all, _, _ = compute_population_peth(
        inputs.spike_times,
        pairs.move_times,
        **PETH_KWARGS,
    )

    pk_stat_all = resp_per_unit(peth_stat_all, bc)
    pk_move_all = resp_per_unit(peth_move_all, bc)
    bl_stat_all = baseline_per_unit(peth_stat_all, bc)
    bl_move_all = baseline_per_unit(peth_move_all, bc)
    above_bl = above_baseline_mask(pk_stat_all, pk_move_all, bl_stat_all, bl_move_all)
    sem_stat_all = np.vstack(resp_sem_log_per_unit(peth_stat_all, bc))
    sem_move_all = np.vstack(resp_sem_log_per_unit(peth_move_all, bc))
    print(
        f"  All units ({len(inputs.unit_ids)}): "
        f"{wilcox_str(pk_stat_all, pk_move_all, gate_mask=above_bl)}"
    )

    delta_move, _, qvals_move, peak_latencies, stat_trial_matrix, move_trial_matrix = (
        per_unit_move_vs_stat_stats(peth_stat_all, peth_move_all, bc)
    )
    good_idx = np.arange(len(inputs.unit_ids))
    sig_exc = (qvals_move < QVAL_ALPHA) & (delta_move > 0)
    sig_supp = (qvals_move < QVAL_ALPHA) & (delta_move < 0)
    nonsig = (~sig_exc) & (~sig_supp)
    print(
        "  Move-vs-stat peak-centered classes "
        f"(FDR<0.05): exc={sig_exc.sum()}  "
        f"supp={sig_supp.sum()}  no-effect={nonsig.sum()}"
    )

    base_mask = (bc >= BASELINE_WINDOW[0]) & (bc < BASELINE_WINDOW[1])
    effect_mask = (bc >= RESP_WINDOW[0]) & (bc < RESP_WINDOW[1])

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
    rate_threshold_hz = RATE_SPLIT_HZ
    low_mask, high_mask, mixed_mask = pair_rate_masks(
        trial_rates,
        trial_rates,
        rate_threshold_hz=rate_threshold_hz,
    )
    print(
        f"  Rate split: threshold={rate_threshold_hz:.1f} Hz  "
        f"low={low_mask.sum()} trials  high={high_mask.sum()} trials"
    )
    if mixed_mask.any():
        print(f"  Excluded {mixed_mask.sum()} threshold-boundary trials from low/high")

    pk_stat_low, pk_move_low, sem_stat_low, sem_move_low = split_resp_per_unit(
        inputs.spike_times,
        pairs.stat_times[low_mask],
        pairs.move_times[low_mask],
        bc,
    )
    pk_stat_high, pk_move_high, sem_stat_high, sem_move_high = split_resp_per_unit(
        inputs.spike_times,
        pairs.stat_times[high_mask],
        pairs.move_times[high_mask],
        bc,
    )
    # Soft baseline gate uses the same per-unit baseline rates from the
    # all-rates PETH (per-unit baseline shouldn't depend on rate subset).
    print(f"  Low rate:  {wilcox_str(pk_stat_low, pk_move_low, gate_mask=above_bl)}")
    print(f"  High rate: {wilcox_str(pk_stat_high, pk_move_high, gate_mask=above_bl)}")

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
        spike_duration_ms=inputs.spike_duration_ms,
        behavior=behavior,
        bin_centers=bc,
        paired_last_stat=pairs.stat_times,
        paired_first_move=pairs.move_times,
        stationary_label=pairs.stationary_label,
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
        paired_trial_count=len(pairs.stat_times),
        n_units=len(inputs.unit_ids),
    )


def analyze_comparison(
    inputs: SessionInputs,
    comparison: ComparisonDef,
):
    pairs = extract_comparison_anchors(inputs, comparison)
    _empty2 = np.zeros((2, 0))
    if len(pairs.stat_times) == 0 or len(pairs.move_times) == 0:
        return {
            "subject": inputs.subject,
            "comparison": comparison,
            "stationary_label": pairs.stationary_label,
            "pk_stat_all": np.array([]),
            "pk_move_all": np.array([]),
            "bl_stat_all": np.array([]),
            "bl_move_all": np.array([]),
            "above_bl_all": np.array([], dtype=bool),
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
        pairs.stat_times,
        **PETH_KWARGS,
    )
    peth_move_all, _, _ = compute_population_peth(
        inputs.spike_times,
        pairs.move_times,
        **PETH_KWARGS,
    )

    pk_stat_all = resp_per_unit(peth_stat_all, bc)
    pk_move_all = resp_per_unit(peth_move_all, bc)
    bl_stat_all = baseline_per_unit(peth_stat_all, bc)
    bl_move_all = baseline_per_unit(peth_move_all, bc)
    above_bl_all = above_baseline_mask(
        pk_stat_all, pk_move_all, bl_stat_all, bl_move_all
    )
    sem_stat_all = np.vstack(resp_sem_log_per_unit(peth_stat_all, bc))
    sem_move_all = np.vstack(resp_sem_log_per_unit(peth_move_all, bc))

    stat_rates = inputs.trial_df.iloc[pairs.stat_trial_idx][
        "stim_rate_vision"
    ].to_numpy(dtype=float)
    move_rates = inputs.trial_df.iloc[pairs.move_trial_idx][
        "stim_rate_vision"
    ].to_numpy(dtype=float)
    low_mask, high_mask, mixed_mask = pair_rate_masks(stat_rates, move_rates)
    if mixed_mask.any():
        print(
            f"  {inputs.subject} {comparison.key}: excluded {mixed_mask.sum()} "
            "cross-stratum or threshold-boundary offset-matched pairs from low/high panels"
        )
    pk_stat_low, pk_move_low, sem_stat_low, sem_move_low = split_resp_per_unit(
        inputs.spike_times,
        pairs.stat_times[low_mask],
        pairs.move_times[low_mask],
        bc,
    )
    pk_stat_high, pk_move_high, sem_stat_high, sem_move_high = split_resp_per_unit(
        inputs.spike_times,
        pairs.stat_times[high_mask],
        pairs.move_times[high_mask],
        bc,
    )
    return {
        "subject": inputs.subject,
        "comparison": comparison,
        "stationary_label": pairs.stationary_label,
        "pk_stat_all": pk_stat_all,
        "pk_move_all": pk_move_all,
        "bl_stat_all": bl_stat_all,
        "bl_move_all": bl_move_all,
        "above_bl_all": above_bl_all,
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
        "paired_trial_count": len(pairs.stat_times),
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
        ax.set_xlabel("Stationary (sp/s)", fontsize=10)
        ax.set_ylabel("Movement (sp/s)", fontsize=10)
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
    ax.set_xlabel("Stationary (sp/s)", fontsize=10)
    ax.set_ylabel("Movement (sp/s)", fontsize=10)
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
        ax.set_xlabel("Stationary (sp/s)", fontsize=10)
        ax.set_ylabel("Movement (sp/s)", fontsize=10)
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
            alpha=0.1,
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
    ax.set_xlabel("Stationary (sp/s)", fontsize=10)
    ax.set_ylabel("Movement (sp/s)", fontsize=10)
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
    for ax, letter in zip(axes, ["A", "B", "C"]):
        ax.text(
            -0.12,
            1.08,
            letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
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


def subset_sem(sem_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if sem_arr.size == 0:
        return sem_arr
    return sem_arr[:, mask]


def waveform_class_mask(result: SessionAnalysis, cell_class: str) -> np.ndarray:
    finite = np.isfinite(result.spike_duration_ms)
    if cell_class == "FS":
        return finite & (result.spike_duration_ms <= NARROW_BROAD_MS)
    if cell_class == "RS":
        return finite & (result.spike_duration_ms > NARROW_BROAD_MS)
    raise ValueError(f"Unknown cell class: {cell_class}")


def build_overlay_waveform_split_figure(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    subject_colors = ["#1f77b4", "#d62728"]
    panel_defs = [
        (
            "All rates",
            lambda r, mask: (
                r.pk_stat_all[mask],
                r.pk_move_all[mask],
                r.paired_trial_count,
                subset_sem(r.sem_stat_all, mask),
                subset_sem(r.sem_move_all, mask),
            ),
        ),
        (
            "Low rate",
            lambda r, mask: (
                r.pk_stat_low[mask],
                r.pk_move_low[mask],
                r.low_count,
                subset_sem(r.sem_stat_low, mask),
                subset_sem(r.sem_move_low, mask),
            ),
        ),
        (
            "High rate",
            lambda r, mask: (
                r.pk_stat_high[mask],
                r.pk_move_high[mask],
                r.high_count,
                subset_sem(r.sem_stat_high, mask),
                subset_sem(r.sem_move_high, mask),
            ),
        ),
    ]

    for row_idx, cell_class in enumerate(["FS", "RS"]):
        for col_idx, (panel_name, getter) in enumerate(panel_defs):
            ax = axes[row_idx, col_idx]
            panel_specs = []
            count_lines = []
            for result, color in zip(results, subject_colors):
                mask = waveform_class_mask(result, cell_class)
                pk_s, pk_m, n_trials, sem_s, sem_m = getter(result, mask)
                label = f"{result.subject} {cell_class}"
                panel_specs.append((pk_s, pk_m, color, label, sem_s, sem_m))
                count_lines.append(
                    f"{result.subject}: units={int(mask.sum())}, trials={n_trials}"
                )
            overlay_scatter_log(ax, panel_specs, panel_name)
            if row_idx != 1:
                ax.set_xlabel("")
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
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, lw=0.0
                ),
            )
        axes[row_idx, 0].text(
            -0.32,
            1.18,
            f"{cell_class} units",
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        axes[row_idx, 0].text(
            -0.32,
            1.07,
            f"putative split by spike duration ({'≤' if cell_class == 'FS' else '>'} {NARROW_BROAD_MS:.1f} ms)",
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=7,
            clip_on=False,
        )

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Cross-subject overlay: locomotion response split by putative FS/RS waveform class",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.11, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.30
    )
    return fig


def build_overlay_summary_figure(summary_rows):
    n_rows = len(summary_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
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
            if row_idx != n_rows - 1:
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
        "Locomotion enhances VISp responses\nGRB006 (expert) · GRB058 (in training)",
        fontsize=12,
        y=0.995,
    )
    return fig


def save_pdf(fig, pdf_path):
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"\nFigure saved: {pdf_path}")


def main():
    grb006 = load_local_spikes_db_behavior(
        subject="GRB006",
        session="20240821_121447",
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
    save_pdf(fig, OUT_PATH)
    plt.close(fig)

    fig_overlay = build_overlay_scatter_figure(results)
    save_pdf(fig_overlay, OUT_PATH_OVERLAY)
    plt.close(fig_overlay)

    fig_overlay_waveform = build_overlay_waveform_split_figure(results)
    save_pdf(fig_overlay_waveform, OUT_PATH_OVERLAY_WAVEFORM)
    plt.close(fig_overlay_waveform)

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
            row_label="Row 2: training-timing correction (2nd stat)",
            description="GRB006 uses 2nd stationary; GRB058 uses last stationary.",
            mode="indexed_stationary",
            stationary_index_by_subject={"GRB006": 1, "GRB058": None},
        ),
        ComparisonDef(
            key="timing_matched_3rd",
            row_label="Row 3: training-timing correction (3rd stat)",
            description="GRB006 uses 3rd stationary; GRB058 uses last stationary.",
            mode="indexed_stationary",
            stationary_index_by_subject={"GRB006": 2, "GRB058": None},
        ),
        ComparisonDef(
            key="offset_500_700ms",
            row_label="Row 4: strict 0.5-0.7 s control",
            description=(
                "Closest offset-matched stationary and movement pulses in 0.5-0.7 s; "
                "low/high panels keep only pairs where both source trials fall in the same rate stratum."
            ),
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
    save_pdf(fig_overlay_summary, OUT_PATH_OVERLAY_SUMMARY)
    plt.close(fig_overlay_summary)


if __name__ == "__main__":
    main()
