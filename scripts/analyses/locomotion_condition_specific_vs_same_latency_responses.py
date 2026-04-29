"""Maintained locomotion figure comparing condition-peak and shared-peak-control readouts.

Outputs:
  - figures/locomotion/readout_comparison_latency_jitter.pdf
  - figures/locomotion/readout_comparison_behavior_matching.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import t

from ephys.src.config.locomotion import (
    BASELINE_WINDOW,
    PETH_KWARGS,
    RESP_WINDOW,
)
from ephys.src.utils.grb006_data import load_grb006_hybrid_session_inputs
from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table
from ephys.src.utils.unit_metrics import fetch_spike_duration_ms
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
)


FIGURE_DIR = Path("/Users/gabriel/lib/ephys/figures/locomotion")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LATENCY_JITTER_FIG_PATH = FIGURE_DIR / "readout_comparison_latency_jitter.pdf"
BEHAVIOR_MATCHING_FIG_PATH = FIGURE_DIR / "readout_comparison_behavior_matching.pdf"

PRIMARY_GRB006 = ("GRB006", "20240821_121447")
PRIMARY_GRB058 = ("GRB058", "20260312_134952")

BOOTSTRAP_ITERS = 2000
PERMUTATION_ITERS = 2000
RNG_SEED = 7
MIN_UNITS_FOR_INFERENCE = 10
MIN_MATCHED_TRIALS = 20
SET1_COLORS = list(colormaps["Set1"].colors)
BACKGROUND_DOT_ALPHA = 0.2
MEAN_CI_LEVEL = 0.95
LOG_PLOT_FLOOR = 0.1


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
) -> SessionInputs:
    print(f"\nLoading hybrid session: {subject} {session}")
    unit_ids, spike_times, trial_df, trial_ts = load_grb006_hybrid_session_inputs()
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    first_stim_times = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim_times = first_stim_times[np.isfinite(first_stim_times)]
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired rows: {len(trial_ts)}  First stims: {len(first_stim_times)}"
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


def load_db_session(subject: str, session: str) -> SessionInputs:
    from ephys.src.utils.utils_IO import fetch_good_units

    print(f"\nLoading DB session: {subject} {session}")
    trial_df, trial_ts, first_stim_times = load_db_behavior(subject, session)
    st_per_unit = fetch_good_units(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  "
        f"Paired rows: {len(trial_ts)}  First stims: {len(first_stim_times)}"
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


def trial_start_from_row(row: pd.Series) -> float:
    if "center_port_entries" in row.index:
        entries = row["center_port_entries"]
        if entries is None or len(entries) == 0:
            return np.nan
        return float(entries[0])
    if "cp_entry" in row.index:
        return float(row["cp_entry"]) if np.isfinite(row["cp_entry"]) else np.nan
    return np.nan


def outcome_label(row: pd.Series) -> str:
    if row.get("rewarded", 0) == 1:
        return "rewarded"
    with_choice = row.get("with_choice", 0)
    response = row.get("response", 0)
    if with_choice == 1 or response in (-1, 1):
        return "incorrect"
    return "no_choice"


def paired_trial_table(inputs: SessionInputs) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, trial_row in inputs.trial_ts.iterrows():
        stationary_stims = trial_row["stationary_stims"]
        movement_stims = trial_row["movement_stims"]
        if len(stationary_stims) == 0 or len(movement_stims) == 0:
            continue
        trial_idx = int(trial_row["trial_idx"])
        full_row = inputs.trial_df.iloc[trial_idx]
        trial_start = trial_start_from_row(trial_row)
        stat_time = float(stationary_stims[-1])
        move_time = float(movement_stims[0])
        fixation_duration = (
            float(full_row["t_react"]) - float(full_row["t_initiate"])
            if np.isfinite(full_row.get("t_react"))
            and np.isfinite(full_row.get("t_initiate"))
            else np.nan
        )
        reaction_time = (
            float(full_row["t_response"]) - float(full_row["t_react"])
            if np.isfinite(full_row.get("t_response"))
            and np.isfinite(full_row.get("t_react"))
            else np.nan
        )
        rows.append(
            {
                "trial_idx": trial_idx,
                "stat_time": stat_time,
                "move_time": move_time,
                "trial_start_time": trial_start,
                "stat_position_in_trial": int(len(stationary_stims)),
                "elapsed_to_stat_s": (
                    stat_time - trial_start if np.isfinite(trial_start) else np.nan
                ),
                "stim_rate_vision": full_row.get("stim_rate_vision", np.nan),
                "prev_outcome": outcome_label(inputs.trial_df.iloc[trial_idx - 1])
                if trial_idx > 0
                else "none",
                "fixation_duration_s": fixation_duration,
                "reaction_time_s": reaction_time,
            }
        )
    paired_df = pd.DataFrame(rows)
    if paired_df.empty:
        raise RuntimeError(
            f"No paired stat/move trials for {inputs.subject} {inputs.session}."
        )
    return paired_df


def response_mean_per_unit(
    peth: np.ndarray, bc: np.ndarray, window: tuple[float, float] = RESP_WINDOW
) -> np.ndarray:
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def baseline_mean_per_unit(
    peth: np.ndarray, bc: np.ndarray, window: tuple[float, float] = BASELINE_WINDOW
) -> np.ndarray:
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def above_baseline_mask(
    pk_stat: np.ndarray, pk_move: np.ndarray, bl_stat: np.ndarray, bl_move: np.ndarray
) -> np.ndarray:
    return (pk_stat - bl_stat > 0) | (pk_move - bl_move > 0)


def per_trial_peak_latencies(peth: np.ndarray, bc: np.ndarray) -> np.ndarray:
    effect_mask = (bc >= RESP_WINDOW[0]) & (bc < RESP_WINDOW[1])
    bc_effect = bc[effect_mask]
    if bc_effect.size == 0:
        raise ValueError("RESP_WINDOW does not overlap available bins.")
    peak_idx = np.argmax(peth[:, :, effect_mask], axis=2)
    return bc_effect[peak_idx]


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(values.std(ddof=1))


def bootstrap_ci(
    values: np.ndarray, *, rng: np.random.Generator, stat=np.nanmedian
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    boots = np.empty(BOOTSTRAP_ITERS, dtype=float)
    for idx in range(BOOTSTRAP_ITERS):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[idx] = stat(sample)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
    stat=np.nanmedian,
) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    boots = np.empty(BOOTSTRAP_ITERS, dtype=float)
    for idx in range(BOOTSTRAP_ITERS):
        sample_a = rng.choice(a, size=a.size, replace=True)
        sample_b = rng.choice(b, size=b.size, replace=True)
        boots[idx] = stat(sample_a) - stat(sample_b)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
    stat=np.nanmedian,
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    observed = abs(stat(a) - stat(b))
    joined = np.concatenate([a, b])
    count = 0
    for _ in range(PERMUTATION_ITERS):
        rng.shuffle(joined)
        perm_a = joined[: a.size]
        perm_b = joined[a.size :]
        if abs(stat(perm_a) - stat(perm_b)) >= observed:
            count += 1
    return float((count + 1) / (PERMUTATION_ITERS + 1))


def ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    return (
        np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
    )


def summarize_null_result(
    values: np.ndarray, *, label: str, rng: np.random.Generator
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(np.nanmedian(values)) if values.size else np.nan
    ci_low, ci_high = bootstrap_ci(values, rng=rng)
    if values.size < MIN_UNITS_FOR_INFERENCE:
        result = "inconclusive"
    elif ci_excludes_zero(ci_low, ci_high):
        result = "null rejected"
    else:
        result = "null not rejected"
    return {
        "metric": label,
        "n": int(values.size),
        "observed_median": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "null_result": result,
    }


def compare_groups_null(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label: str,
    group_a: str,
    group_b: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    diff = float(np.nanmedian(a) - np.nanmedian(b)) if a.size and b.size else np.nan
    ci_low, ci_high = bootstrap_diff_ci(a, b, rng=rng)
    p_value = permutation_pvalue(a, b, rng=rng)
    if min(a.size, b.size) < MIN_UNITS_FOR_INFERENCE:
        result = "inconclusive"
    elif ci_excludes_zero(ci_low, ci_high):
        result = "null rejected"
    else:
        result = "null not rejected"
    return {
        "metric": label,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "observed_median_diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "permutation_p": p_value,
        "null_result": result,
    }


def fixed_bin(value: float, edges: list[float]) -> str:
    if not np.isfinite(value):
        return "nan"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f"{edges[-2]:.2f}+"


def behavior_match_components(paired_df: pd.DataFrame) -> dict[str, pd.Series]:
    elapsed_labels = paired_df["elapsed_to_stat_s"].apply(
        lambda x: fixed_bin(x, [0.0, 0.35, 0.55, 0.75, 1.0, 1.5, 5.0])
    )
    fixation_labels = paired_df["fixation_duration_s"].apply(
        lambda x: fixed_bin(x, [0.0, 0.25, 0.4, 0.6, 0.8, 1.2, 5.0])
    )
    reaction_labels = paired_df["reaction_time_s"].apply(
        lambda x: fixed_bin(x, [0.0, 0.15, 0.3, 0.5, 0.8, 1.5, 5.0])
    )
    stim_rate = paired_df["stim_rate_vision"].fillna(-1).astype(int).astype(str)
    stat_position = paired_df["stat_position_in_trial"].apply(
        lambda x: "3+" if int(x) >= 3 else str(int(x))
    )
    prev_outcome = paired_df["prev_outcome"].astype(str)
    return {
        "stat_position": stat_position,
        "elapsed": elapsed_labels,
        "stim_rate": stim_rate,
        "prev_outcome": prev_outcome,
        "fixation": fixation_labels,
        "reaction": reaction_labels,
    }


def median_delta_summary(
    stat_values: np.ndarray, move_values: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    stat_values = np.asarray(stat_values, dtype=float)
    move_values = np.asarray(move_values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    stat_sel = stat_values[mask]
    move_sel = move_values[mask]
    if stat_sel.size == 0:
        return np.nan, np.nan
    return float(np.median(move_sel - stat_sel)), float(
        100 * np.mean(move_sel > stat_sel)
    )


def mean_and_t_ci(
    values: np.ndarray, *, log_scale: bool = False
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("mean_and_t_ci requires at least one value.")
    if values.size == 1:
        value = float(values[0])
        return value, value, value

    if log_scale:
        log_values = np.log(values)
        mean_log = float(np.mean(log_values))
        dof = values.size - 1
        t_crit = float(t.ppf((1.0 + MEAN_CI_LEVEL) / 2.0, dof))
        sem_log = float(np.std(log_values, ddof=1)) / np.sqrt(values.size)
        lower = float(np.exp(mean_log - t_crit * sem_log))
        upper = float(np.exp(mean_log + t_crit * sem_log))
        mean_value = float(np.exp(mean_log))
        return mean_value, lower, upper

    mean_value = float(np.mean(values))
    dof = values.size - 1
    t_crit = float(t.ppf((1.0 + MEAN_CI_LEVEL) / 2.0, dof))
    sem_value = float(np.std(values, ddof=1)) / np.sqrt(values.size)
    lower = mean_value - t_crit * sem_value
    upper = mean_value + t_crit * sem_value
    return mean_value, lower, upper


def draw_violin(
    ax: plt.Axes,
    groups: list[np.ndarray],
    positions: np.ndarray,
    colors: list[str],
    *,
    rng: np.random.Generator,
) -> None:
    for values, xpos, color in zip(groups, positions, colors):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(
            np.full(values.size, xpos) + jitter,
            values,
            s=8,
            alpha=0.15,
            color=color,
            edgecolors="none",
            zorder=1,
        )
    parts = ax.violinplot(
        groups,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.0)


def compute_session_metrics(
    inputs: SessionInputs,
    paired_df: pd.DataFrame,
    *,
    paired_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if paired_mask is None:
        paired_mask = np.ones(len(paired_df), dtype=bool)
    paired_df = paired_df.loc[np.asarray(paired_mask, dtype=bool)].reset_index(
        drop=True
    )
    if paired_df.empty:
        raise RuntimeError(
            f"No paired trials retained for {inputs.subject} {inputs.session}."
        )

    stat_times = paired_df["stat_time"].to_numpy(dtype=float)
    move_times = paired_df["move_time"].to_numpy(dtype=float)

    peth_stat, _, bc = compute_population_peth(
        inputs.spike_times, stat_times, **PETH_KWARGS
    )
    peth_move, _, _ = compute_population_peth(
        inputs.spike_times, move_times, **PETH_KWARGS
    )

    stat_resp_mean = response_mean_per_unit(peth_stat, bc)
    move_resp_mean = response_mean_per_unit(peth_move, bc)
    stat_baseline_mean = baseline_mean_per_unit(peth_stat, bc)
    move_baseline_mean = baseline_mean_per_unit(peth_move, bc)
    gate_mask = above_baseline_mask(
        stat_resp_mean, move_resp_mean, stat_baseline_mean, move_baseline_mean
    )

    effect_mask = (bc >= RESP_WINDOW[0]) & (bc < RESP_WINDOW[1])
    baseline_mask = (bc >= BASELINE_WINDOW[0]) & (bc < BASELINE_WINDOW[1])
    bc_effect = bc[effect_mask]

    mean_stat = peth_stat.mean(axis=1)
    mean_move = peth_move.mean(axis=1)
    stat_peak_idx = np.argmax(mean_stat[:, effect_mask], axis=1)
    move_peak_idx = np.argmax(mean_move[:, effect_mask], axis=1)
    stat_peak_latency = bc_effect[stat_peak_idx]
    move_peak_latency = bc_effect[move_peak_idx]

    stat_baseline_rate = mean_stat[:, baseline_mask].mean(axis=1)
    move_baseline_rate = mean_move[:, baseline_mask].mean(axis=1)
    stat_peak_raw = mean_stat[:, effect_mask][
        np.arange(mean_stat.shape[0]), stat_peak_idx
    ]
    move_peak_raw = mean_move[:, effect_mask][
        np.arange(mean_move.shape[0]), move_peak_idx
    ]
    stat_peak_shared_baseline = stat_peak_raw - stat_baseline_rate
    move_peak_shared_baseline = move_peak_raw - stat_baseline_rate
    stat_peak_condition_baseline = stat_peak_raw - stat_baseline_rate
    move_peak_condition_baseline = move_peak_raw - move_baseline_rate

    mean_combined = 0.5 * (mean_stat + mean_move)
    shared_peak_idx = np.argmax(mean_combined[:, effect_mask], axis=1)
    shared_peak_latency = bc_effect[shared_peak_idx]

    shared_stat_cond = np.zeros(len(inputs.unit_ids), dtype=float)
    shared_move_cond = np.zeros(len(inputs.unit_ids), dtype=float)
    shared_stat_shared = np.zeros(len(inputs.unit_ids), dtype=float)
    shared_move_shared = np.zeros(len(inputs.unit_ids), dtype=float)
    shared_stat_raw = np.zeros(len(inputs.unit_ids), dtype=float)
    shared_move_raw = np.zeros(len(inputs.unit_ids), dtype=float)
    for unit_idx, shared_idx in enumerate(shared_peak_idx):
        stat_shared_peak = float(mean_stat[unit_idx, effect_mask][shared_idx])
        move_shared_peak = float(mean_move[unit_idx, effect_mask][shared_idx])
        shared_stat_raw[unit_idx] = stat_shared_peak
        shared_move_raw[unit_idx] = move_shared_peak
        shared_stat_cond[unit_idx] = stat_shared_peak - stat_baseline_rate[unit_idx]
        shared_move_cond[unit_idx] = move_shared_peak - move_baseline_rate[unit_idx]
        shared_stat_shared[unit_idx] = stat_shared_peak - stat_baseline_rate[unit_idx]
        shared_move_shared[unit_idx] = move_shared_peak - stat_baseline_rate[unit_idx]

    stat_trial_peak_lat = per_trial_peak_latencies(peth_stat, bc)
    move_trial_peak_lat = per_trial_peak_latencies(peth_move, bc)
    stat_jitter = np.array([safe_std(row) for row in stat_trial_peak_lat], dtype=float)
    move_jitter = np.array([safe_std(row) for row in move_trial_peak_lat], dtype=float)

    condition_peak_shared_delta = move_peak_shared_baseline - stat_peak_shared_baseline
    condition_peak_cond_delta = (
        move_peak_condition_baseline - stat_peak_condition_baseline
    )
    shared_peak_control_shared_delta = shared_move_shared - shared_stat_shared
    shared_peak_control_cond_delta = shared_move_cond - shared_stat_cond
    latency_shift = move_peak_latency - stat_peak_latency
    jitter_shift = move_jitter - stat_jitter
    shared_vs_condition_mismatch = 0.5 * (
        np.abs(shared_peak_latency - stat_peak_latency)
        + np.abs(shared_peak_latency - move_peak_latency)
    )
    latency_contribution = (
        condition_peak_shared_delta - shared_peak_control_shared_delta
    )
    baseline_contribution = condition_peak_cond_delta - condition_peak_shared_delta

    unit_summary = pd.DataFrame(
        {
            "subject": inputs.subject,
            "session": inputs.session,
            "unit_id": inputs.unit_ids,
            "spike_duration_ms": inputs.spike_duration_ms,
            "baseline_gate_pass": gate_mask,
            "stat_response_mean_sp_s": stat_resp_mean,
            "move_response_mean_sp_s": move_resp_mean,
            "stat_baseline_mean_sp_s": stat_baseline_mean,
            "move_baseline_mean_sp_s": move_baseline_mean,
            "stat_peak_latency_s": stat_peak_latency,
            "move_peak_latency_s": move_peak_latency,
            "shared_peak_latency_s": shared_peak_latency,
            "latency_shift_s": latency_shift,
            "stat_jitter_s": stat_jitter,
            "move_jitter_s": move_jitter,
            "jitter_shift_s": jitter_shift,
            "shared_vs_condition_mismatch_s": shared_vs_condition_mismatch,
            "condition_peak_raw_stationary_sp_s": stat_peak_raw,
            "condition_peak_raw_movement_sp_s": move_peak_raw,
            "condition_peak_shared_stat_baseline_stationary_sp_s": stat_peak_shared_baseline,
            "condition_peak_shared_stat_baseline_movement_sp_s": move_peak_shared_baseline,
            "shared_peak_control_raw_stationary_sp_s": shared_stat_raw,
            "shared_peak_control_raw_movement_sp_s": shared_move_raw,
            "shared_peak_control_shared_stat_baseline_stationary_sp_s": shared_stat_shared,
            "shared_peak_control_shared_stat_baseline_movement_sp_s": shared_move_shared,
            "condition_peak_shared_stat_baseline_effect_sp_s": condition_peak_shared_delta,
            "condition_peak_condition_baseline_effect_sp_s": condition_peak_cond_delta,
            "shared_peak_control_shared_stat_baseline_effect_sp_s": shared_peak_control_shared_delta,
            "shared_peak_control_condition_baseline_effect_sp_s": shared_peak_control_cond_delta,
            "latency_contribution_sp_s": latency_contribution,
            "baseline_contribution_sp_s": baseline_contribution,
            "paired_trial_count": len(paired_df),
        }
    )

    session_summary = {
        "subject": inputs.subject,
        "session": inputs.session,
        "paired_trial_count": len(paired_df),
        "unit_count": len(inputs.unit_ids),
        "baseline_gate_unit_count": int(gate_mask.sum()),
        "paired_trial_indices": paired_df["trial_idx"].to_numpy(dtype=int),
    }
    return unit_summary, session_summary


def add_session_analysis_rows(
    rows: list[dict[str, object]],
    unit_df: pd.DataFrame,
    session_info: dict[str, object],
    *,
    rng: np.random.Generator,
) -> None:
    gated_df = unit_df[unit_df["baseline_gate_pass"]].copy()
    session_label = f"{session_info['subject']} {session_info['session']}"

    latency_summary = summarize_null_result(
        gated_df["latency_shift_s"].to_numpy(dtype=float),
        label="Within-animal latency shift",
        rng=rng,
    )
    rows.append(
        {
            "analysis_scope": "within_session",
            "subject": session_info["subject"],
            "session": session_info["session"],
            "comparison": session_label,
            "metric_key": "latency_shift",
            "null_hypothesis": "Locomotion does not systematically shift response latency within this session.",
            "evidence_against_null": "Latency-shift median CI excludes 0.",
            **latency_summary,
        }
    )

    jitter_summary = summarize_null_result(
        gated_df["jitter_shift_s"].to_numpy(dtype=float),
        label="Within-animal jitter shift",
        rng=rng,
    )
    rows.append(
        {
            "analysis_scope": "within_session",
            "subject": session_info["subject"],
            "session": session_info["session"],
            "comparison": session_label,
            "metric_key": "jitter_shift",
            "null_hypothesis": "Locomotion does not systematically change trial-to-trial peak-latency jitter within this session.",
            "evidence_against_null": "Jitter-shift median CI excludes 0.",
            **jitter_summary,
        }
    )

    surface_gap_summary = summarize_null_result(
        gated_df["latency_contribution_sp_s"].to_numpy(dtype=float),
        label="Within-session convention gap",
        rng=rng,
    )
    rows.append(
        {
            "analysis_scope": "within_session",
            "subject": session_info["subject"],
            "session": session_info["session"],
            "comparison": session_label,
            "metric_key": "surface_gap",
            "null_hypothesis": "Condition-peak and shared-peak-control readouts do not differ systematically within this session.",
            "evidence_against_null": "Convention-gap median CI excludes 0.",
            **surface_gap_summary,
        }
    )

    condition_peak_delta_median, condition_peak_pct = median_delta_summary(
        np.zeros(len(gated_df)),
        gated_df["condition_peak_shared_stat_baseline_effect_sp_s"].to_numpy(
            dtype=float
        ),
        np.ones(len(gated_df), dtype=bool),
    )
    shared_peak_control_delta_median, shared_peak_control_pct = median_delta_summary(
        np.zeros(len(gated_df)),
        gated_df["shared_peak_control_shared_stat_baseline_effect_sp_s"].to_numpy(
            dtype=float
        ),
        np.ones(len(gated_df), dtype=bool),
    )
    rows.extend(
        [
            {
                "analysis_scope": "session_effect",
                "subject": session_info["subject"],
                "session": session_info["session"],
                "comparison": session_label,
                "metric_key": "condition_peak_shared_stat_baseline_effect",
                "metric": "Condition-peak shared-stationary-baseline effect",
                "n": int(len(gated_df)),
                "observed_median": condition_peak_delta_median,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "null_result": "",
                "null_hypothesis": "",
                "evidence_against_null": "",
                "percent_above_zero": condition_peak_pct,
                "paired_trial_count": int(session_info["paired_trial_count"]),
                "baseline_gate_unit_count": int(
                    session_info["baseline_gate_unit_count"]
                ),
            },
            {
                "analysis_scope": "session_effect",
                "subject": session_info["subject"],
                "session": session_info["session"],
                "comparison": session_label,
                "metric_key": "shared_peak_control_shared_stat_baseline_effect",
                "metric": "Shared-peak-control shared-stationary-baseline effect",
                "n": int(len(gated_df)),
                "observed_median": shared_peak_control_delta_median,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "null_result": "",
                "null_hypothesis": "",
                "evidence_against_null": "",
                "percent_above_zero": shared_peak_control_pct,
                "paired_trial_count": int(session_info["paired_trial_count"]),
                "baseline_gate_unit_count": int(
                    session_info["baseline_gate_unit_count"]
                ),
            },
        ]
    )


def plot_latency_jitter_figure(
    unit_summary: pd.DataFrame, analysis_rows: pd.DataFrame
) -> None:
    gated = unit_summary[unit_summary["baseline_gate_pass"]].copy()
    session_order = [
        ("GRB006", "20240821_121447"),
        ("GRB058", "20260312_134952"),
    ]
    labels = [subject for subject, _ in session_order]
    latency_groups = [
        gated.loc[
            (gated["subject"] == subject) & (gated["session"] == session),
            "latency_shift_s",
        ].to_numpy(dtype=float)
        for subject, session in session_order
    ]
    jitter_groups = [
        gated.loc[
            (gated["subject"] == subject) & (gated["session"] == session),
            "jitter_shift_s",
        ].to_numpy(dtype=float)
        for subject, session in session_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))
    axes = axes.ravel()
    positions = np.arange(1, len(session_order) + 1)
    violin_colors = [SET1_COLORS[1], SET1_COLORS[0]]
    plot_rng = np.random.default_rng(RNG_SEED)

    draw_violin(axes[0], latency_groups, positions, violin_colors, rng=plot_rng)
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("move - stat peak latency (s)")

    draw_violin(axes[1], jitter_groups, positions, violin_colors, rng=plot_rng)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("move - stat jitter (s)")

    fig.tight_layout()
    fig.savefig(LATENCY_JITTER_FIG_PATH)
    plt.close(fig)


def plot_behavior_matching_panel(
    ax: plt.Axes,
    unit_dfs: dict[str, pd.DataFrame],
    *,
    x_col: str,
    y_col: str,
    column_label: str | None,
    row_label: str | None,
    show_legend: bool,
    axis_limits: tuple[float, float],
) -> None:
    subject_order = ["GRB006", "GRB058"]
    subject_colors = {"GRB006": SET1_COLORS[1], "GRB058": SET1_COLORS[0]}
    lower_limit, upper_limit = axis_limits

    for subject in subject_order:
        df = unit_dfs[subject]
        x = np.maximum(df[x_col].to_numpy(dtype=float), 0.0) + LOG_PLOT_FLOOR
        y = np.maximum(df[y_col].to_numpy(dtype=float), 0.0) + LOG_PLOT_FLOOR
        color = subject_colors[subject]
        ax.scatter(
            x,
            y,
            s=18,
            alpha=BACKGROUND_DOT_ALPHA,
            color=color,
            linewidths=0,
            label="_nolegend_",
            zorder=2,
        )
        mean_x, lower_x, upper_x = mean_and_t_ci(x, log_scale=True)
        mean_y, lower_y, upper_y = mean_and_t_ci(y, log_scale=True)
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=np.array([[mean_x - lower_x], [upper_x - mean_x]]),
            yerr=np.array([[mean_y - lower_y], [upper_y - mean_y]]),
            fmt="o",
            ms=9,
            color=color,
            mfc=color,
            mec="white",
            mew=0.8,
            elinewidth=1.2,
            ecolor=color,
            capsize=2.5,
            alpha=0.95,
            zorder=5,
            label=subject if show_legend else "_nolegend_",
        )

    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        "k--",
        alpha=0.4,
        lw=0.8,
    )
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel("stationary (sp/s)")
    ax.set_ylabel("movement (sp/s)")
    if column_label is not None:
        ax.text(
            0.03,
            0.97,
            column_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, lw=0.0),
        )
    if row_label is not None:
        ax.text(
            -0.24,
            1.05,
            row_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_behavior_matching_figure(
    gated_unit_dfs: dict[str, pd.DataFrame],
    all_unit_dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.0))
    axis_values = []
    for unit_dfs, x_col, y_col in [
        (
            gated_unit_dfs,
            "condition_peak_shared_stat_baseline_stationary_sp_s",
            "condition_peak_shared_stat_baseline_movement_sp_s",
        ),
        (
            gated_unit_dfs,
            "shared_peak_control_shared_stat_baseline_stationary_sp_s",
            "shared_peak_control_shared_stat_baseline_movement_sp_s",
        ),
        (
            all_unit_dfs,
            "condition_peak_raw_stationary_sp_s",
            "condition_peak_raw_movement_sp_s",
        ),
        (
            all_unit_dfs,
            "shared_peak_control_raw_stationary_sp_s",
            "shared_peak_control_raw_movement_sp_s",
        ),
    ]:
        for subject_df in unit_dfs.values():
            axis_values.append(
                np.maximum(subject_df[x_col].to_numpy(dtype=float), 0.0)
                + LOG_PLOT_FLOOR
            )
            axis_values.append(
                np.maximum(subject_df[y_col].to_numpy(dtype=float), 0.0)
                + LOG_PLOT_FLOOR
            )
    all_values = np.concatenate([vals for vals in axis_values if vals.size])
    axis_limits = (
        LOG_PLOT_FLOOR,
        max(1.0, float(np.percentile(all_values, 99) * 1.05)),
    )
    panel_specs = [
        (
            axes[0, 0],
            gated_unit_dfs,
            "condition_peak_shared_stat_baseline_stationary_sp_s",
            "condition_peak_shared_stat_baseline_movement_sp_s",
            "Condition peak",
            "Baseline-gated units",
            True,
            axis_limits,
        ),
        (
            axes[0, 1],
            gated_unit_dfs,
            "shared_peak_control_shared_stat_baseline_stationary_sp_s",
            "shared_peak_control_shared_stat_baseline_movement_sp_s",
            "Shared peak control",
            None,
            False,
            axis_limits,
        ),
        (
            axes[1, 0],
            all_unit_dfs,
            "condition_peak_raw_stationary_sp_s",
            "condition_peak_raw_movement_sp_s",
            None,
            "All units",
            False,
            axis_limits,
        ),
        (
            axes[1, 1],
            all_unit_dfs,
            "shared_peak_control_raw_stationary_sp_s",
            "shared_peak_control_raw_movement_sp_s",
            None,
            None,
            False,
            axis_limits,
        ),
    ]
    for (
        ax,
        unit_dfs,
        x_col,
        y_col,
        column_label,
        row_label,
        show_legend,
        axis_limits,
    ) in panel_specs:
        plot_behavior_matching_panel(
            ax,
            unit_dfs,
            x_col=x_col,
            y_col=y_col,
            column_label=column_label,
            row_label=row_label,
            show_legend=show_legend,
            axis_limits=axis_limits,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    grb006_inputs = load_local_spikes_db_behavior(
        PRIMARY_GRB006[0],
        PRIMARY_GRB006[1],
    )
    grb058_primary_inputs = load_db_session(*PRIMARY_GRB058)

    inputs_by_key = {
        PRIMARY_GRB006: grb006_inputs,
        PRIMARY_GRB058: grb058_primary_inputs,
    }

    paired_tables = {
        key: paired_trial_table(inputs) for key, inputs in inputs_by_key.items()
    }

    all_unit_rows: list[pd.DataFrame] = []
    analysis_rows: list[dict[str, object]] = []
    session_metrics_by_key: dict[
        tuple[str, str], tuple[pd.DataFrame, dict[str, object]]
    ] = {}

    for key, inputs in inputs_by_key.items():
        unit_df, session_info = compute_session_metrics(inputs, paired_tables[key])
        session_metrics_by_key[key] = (unit_df, session_info)
        all_unit_rows.append(unit_df)
        add_session_analysis_rows(analysis_rows, unit_df, session_info, rng=rng)
        print(
            f"{inputs.subject} {inputs.session}: paired={session_info['paired_trial_count']} "
            f"gate={session_info['baseline_gate_unit_count']}/{session_info['unit_count']}"
        )

    unit_summary = pd.concat(all_unit_rows, ignore_index=True)

    grb006_unit_df, _ = session_metrics_by_key[PRIMARY_GRB006]
    grb058_unit_df, _ = session_metrics_by_key[PRIMARY_GRB058]

    grb006_gated = grb006_unit_df[grb006_unit_df["baseline_gate_pass"]]
    grb058_gated = grb058_unit_df[grb058_unit_df["baseline_gate_pass"]]

    analysis_rows.append(
        {
            "analysis_scope": "cross_animal",
            "subject": "GRB006",
            "session": PRIMARY_GRB006[1],
            "comparison": f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]} vs {PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
            "metric_key": "latency_shift",
            "null_hypothesis": "GRB006 and GRB058 have the same locomotion-related latency-shift distribution.",
            "evidence_against_null": "Bootstrap CI for the between-session median difference excludes 0.",
            **compare_groups_null(
                grb058_gated["latency_shift_s"].to_numpy(dtype=float),
                grb006_gated["latency_shift_s"].to_numpy(dtype=float),
                label="Cross-animal latency shift",
                group_a=f"{PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
                group_b=f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]}",
                rng=rng,
            ),
        }
    )
    analysis_rows.append(
        {
            "analysis_scope": "cross_animal",
            "subject": "GRB006",
            "session": PRIMARY_GRB006[1],
            "comparison": f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]} vs {PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
            "metric_key": "jitter_shift",
            "null_hypothesis": "GRB006 and GRB058 have the same locomotion-related jitter-shift distribution.",
            "evidence_against_null": "Bootstrap CI for the between-session median difference excludes 0.",
            **compare_groups_null(
                grb058_gated["jitter_shift_s"].to_numpy(dtype=float),
                grb006_gated["jitter_shift_s"].to_numpy(dtype=float),
                label="Cross-animal jitter shift",
                group_a=f"{PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
                group_b=f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]}",
                rng=rng,
            ),
        }
    )
    analysis_rows.append(
        {
            "analysis_scope": "cross_animal",
            "subject": "GRB006",
            "session": PRIMARY_GRB006[1],
            "comparison": f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]} vs {PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
            "metric_key": "surface_gap",
            "null_hypothesis": "The convention-induced effect drop is the same across animals.",
            "evidence_against_null": "Bootstrap CI for the between-session median difference excludes 0.",
            **compare_groups_null(
                grb058_gated["latency_contribution_sp_s"].to_numpy(dtype=float),
                grb006_gated["latency_contribution_sp_s"].to_numpy(dtype=float),
                label="Cross-animal convention gap",
                group_a=f"{PRIMARY_GRB058[0]} {PRIMARY_GRB058[1]}",
                group_b=f"{PRIMARY_GRB006[0]} {PRIMARY_GRB006[1]}",
                rng=rng,
            ),
        }
    )

    analysis_summary = pd.DataFrame(analysis_rows)

    plot_latency_jitter_figure(unit_summary, analysis_summary)
    plot_behavior_matching_figure(
        gated_unit_dfs={"GRB006": grb006_gated, "GRB058": grb058_gated},
        all_unit_dfs={"GRB006": grb006_unit_df, "GRB058": grb058_unit_df},
        output_path=BEHAVIOR_MATCHING_FIG_PATH,
    )
    print("\nWrote:")
    print(f"  {LATENCY_JITTER_FIG_PATH}")
    print(f"  {BEHAVIOR_MATCHING_FIG_PATH}")


if __name__ == "__main__":
    main()
