"""Niell-style locomotion approximation for flash responses, split by FS/RS.

This is a supporting comparison script, not part of the maintained locomotion
surface. It approximates the Niell/Stryker stationary-vs-moving comparison
using the movement-state labels available in this dataset.

Two modes are shown for each session:
  1. Paired-anchor: first stationary vs first movement within qualifying trials
  2. Paired-anchor: last stationary vs first movement within qualifying trials

For each unit and condition:
  - build the mean PSTH
  - subtract the condition-specific pre-stim baseline
  - take the peak value in the response window

The resulting scatter is colored by putative waveform class:
  - FS: spike duration <= 0.4 ms (blue)
  - RS: spike duration > 0.4 ms (green)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from ephys.src.config.locomotion import BASELINE_WINDOW, PETH_KWARGS, RESP_WINDOW
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
)

LOCAL_TRIAL_TS = Path("/Users/gabriel/Downloads/Organized/Code/trial_ts.pkl")
LOCAL_SPIKE_TIMES = Path(
    "/Users/gabriel/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl"
)
OUT_PATH = Path("/Users/gabriel/lib/ephys/figures/locomotion/niell_style_fs_rs.pdf")
OUT_PATH_SHARED_STAT_BASELINE = Path(
    "/Users/gabriel/lib/ephys/figures/locomotion/niell_style_fs_rs_shared_stat_baseline.pdf"
)
OUT_PATH_LOG = Path(
    "/Users/gabriel/lib/ephys/figures/locomotion/niell_style_fs_rs_log.pdf"
)
OUT_PATH_SHARED_STAT_BASELINE_LOG = Path(
    "/Users/gabriel/lib/ephys/figures/locomotion/niell_style_fs_rs_shared_stat_baseline_log.pdf"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FS_RS_BOUNDARY_MS = 0.4
COL_FS = "#1f77b4"
COL_RS = "#2ca02c"


@dataclass
class SessionInputs:
    subject: str
    session: str
    unit_ids: list[int]
    spike_times: list[np.ndarray]
    spike_duration_ms: np.ndarray
    trial_ts: pd.DataFrame


@dataclass
class ModeResult:
    subject: str
    mode_label: str
    baseline_label: str
    pk_stat: np.ndarray
    pk_move: np.ndarray
    fs_mask: np.ndarray
    rs_mask: np.ndarray
    missing_mask: np.ndarray
    n_stat_events: int
    n_move_events: int


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


def fetch_spike_duration_ms(
    subject: str, session: str, unit_ids: list[int]
) -> np.ndarray:
    from labdata.schema import EphysRecording, SpikeSorting, UnitCount, UnitMetrics

    sess_q = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()
    good_ids = (
        sess_q * (UnitCount.Unit & "unit_criteria_id = 1" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)
    metric_rows = pd.DataFrame(
        ((SpikeSorting.Unit & good_ids) * UnitMetrics).fetch(
            "unit_id", "spike_duration", as_dict=True
        )
    )
    if metric_rows.empty:
        return np.full(len(unit_ids), np.nan, dtype=float)

    srate = float((EphysRecording.ProbeSetting() & sess_q).fetch("sampling_rate")[0])
    med = metric_rows["spike_duration"].dropna()
    if len(med) and med.median() > 100:
        metric_rows["spike_duration_ms"] = (
            metric_rows["spike_duration"] / srate * 1000.0
        )
    else:
        metric_rows["spike_duration_ms"] = metric_rows["spike_duration"]
    dur_map = dict(
        zip(
            metric_rows["unit_id"].astype(int).tolist(),
            metric_rows["spike_duration_ms"].astype(float).tolist(),
        )
    )
    return np.array([dur_map.get(int(uid), np.nan) for uid in unit_ids], dtype=float)


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


def load_db_behavior(subject: str, session: str) -> pd.DataFrame:
    from ephys.src.utils.utils_IO import fetch_session_events, fetch_trial_metadata

    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}")
    trial_df = enrich_trial_df(trial_df)
    return build_trial_stim_classification(align_ev, trial_df).reset_index(drop=True)


def load_grb006() -> SessionInputs:
    subject, session = "GRB006", "20240821_121447"
    print(f"\nLoading hybrid session: {subject} {session}")
    trial_df = fetch_chipmunk_session_trials(subject, session)
    trial_ts = pd.read_pickle(LOCAL_TRIAL_TS).reset_index(drop=True).copy()
    trial_ts["trial_idx"] = align_local_trials_to_full_trial_df(trial_ts, trial_df)
    unit_ids, spike_times = load_local_spike_times(LOCAL_SPIKE_TIMES)
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(trial_df)}  Paired rows: {len(trial_ts)}"
    )
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        spike_duration_ms=spike_duration_ms,
        trial_ts=trial_ts,
    )


def load_grb058() -> SessionInputs:
    from ephys.src.utils.utils_IO import fetch_good_units

    subject, session = "GRB058", "20260312_134952"
    print(f"\nLoading DB session: {subject} {session}")
    trial_ts = load_db_behavior(subject, session)
    st_per_unit = fetch_good_units(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    spike_duration_ms = fetch_spike_duration_ms(subject, session, unit_ids)
    print(f"  Units: {len(unit_ids)}  Paired rows: {len(trial_ts)}")
    return SessionInputs(
        subject=subject,
        session=session,
        unit_ids=unit_ids,
        spike_times=spike_times,
        spike_duration_ms=spike_duration_ms,
        trial_ts=trial_ts,
    )


def mean_psth_and_baseline(
    peth: np.ndarray, bc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base_mask = (bc >= BASELINE_WINDOW[0]) & (bc < BASELINE_WINDOW[1])
    mean_psth = peth.mean(axis=1)
    baseline = mean_psth[:, base_mask].mean(axis=1, keepdims=True)
    return mean_psth, baseline


def peak_from_baseline_corrected_mean(
    mean_psth: np.ndarray, bc: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    resp_mask = (bc >= RESP_WINDOW[0]) & (bc < RESP_WINDOW[1])
    resp = mean_psth[:, resp_mask] - baseline
    return resp.max(axis=1)


def mode_event_times(
    trial_ts: pd.DataFrame, mode: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if mode == "first_paired":
        has_stat = trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
        has_move = trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
        paired_trials = trial_ts[has_stat & has_move]
        stat_times = np.array(
            [stims[0] for stims in paired_trials["stationary_stims"]], dtype=float
        )
        move_times = np.array(
            [stims[0] for stims in paired_trials["movement_stims"]], dtype=float
        )
        return stat_times, move_times, len(stat_times), len(move_times)

    if mode == "paired":
        has_stat = trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
        has_move = trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
        paired_trials = trial_ts[has_stat & has_move]
        stat_times = np.array(
            [stims[-1] for stims in paired_trials["stationary_stims"]], dtype=float
        )
        move_times = np.array(
            [stims[0] for stims in paired_trials["movement_stims"]], dtype=float
        )
        return stat_times, move_times, len(stat_times), len(move_times)

    raise ValueError(f"Unknown mode: {mode}")


def analyze_mode(
    inputs: SessionInputs,
    mode: str,
    mode_label: str,
    *,
    shared_stat_baseline: bool,
) -> ModeResult:
    stat_times, move_times, n_stat_events, n_move_events = mode_event_times(
        inputs.trial_ts, mode
    )
    if len(stat_times) == 0 or len(move_times) == 0:
        raise RuntimeError(f"No events found for {inputs.subject} mode={mode}")

    peth_stat, _, bc = compute_population_peth(
        inputs.spike_times, stat_times, **PETH_KWARGS
    )
    peth_move, _, _ = compute_population_peth(
        inputs.spike_times, move_times, **PETH_KWARGS
    )

    mean_stat, stat_baseline = mean_psth_and_baseline(peth_stat, bc)
    mean_move, move_baseline = mean_psth_and_baseline(peth_move, bc)
    move_ref_baseline = stat_baseline if shared_stat_baseline else move_baseline

    pk_stat = peak_from_baseline_corrected_mean(mean_stat, bc, stat_baseline)
    pk_move = peak_from_baseline_corrected_mean(mean_move, bc, move_ref_baseline)

    finite = np.isfinite(inputs.spike_duration_ms)
    fs_mask = finite & (inputs.spike_duration_ms <= FS_RS_BOUNDARY_MS)
    rs_mask = finite & (inputs.spike_duration_ms > FS_RS_BOUNDARY_MS)
    missing_mask = ~finite

    return ModeResult(
        subject=inputs.subject,
        mode_label=mode_label,
        baseline_label=(
            "Shared stationary baseline"
            if shared_stat_baseline
            else "Condition-specific baseline"
        ),
        pk_stat=pk_stat,
        pk_move=pk_move,
        fs_mask=fs_mask,
        rs_mask=rs_mask,
        missing_mask=missing_mask,
        n_stat_events=n_stat_events,
        n_move_events=n_move_events,
    )


def summary_line(pk_stat: np.ndarray, pk_move: np.ndarray) -> str:
    if len(pk_stat) == 0:
        return "no units"
    pct = 100 * (pk_move > pk_stat).mean()
    delta = float(np.median(pk_move - pk_stat))
    try:
        _, p = wilcoxon(pk_move, pk_stat)
    except ValueError:
        p = 1.0
    return (
        f"{pct:.0f}% above diag  Δmedian={delta:+.2f} sp/s  p={p:.3f}  n={len(pk_stat)}"
    )


def print_mode_summary(result: ModeResult) -> None:
    keep = result.fs_mask | result.rs_mask
    print(f"\n{result.subject}  {result.mode_label}")
    print(f"  stationary events: {result.n_stat_events}")
    print(f"  movement events:   {result.n_move_events}")
    print(
        f"  waveform classes: FS={int(result.fs_mask.sum())}  "
        f"RS={int(result.rs_mask.sum())}  missing={int(result.missing_mask.sum())}"
    )
    print(f"  baseline: {result.baseline_label}")
    print(f"  overall: {summary_line(result.pk_stat[keep], result.pk_move[keep])}")
    print(
        f"  FS:      {summary_line(result.pk_stat[result.fs_mask], result.pk_move[result.fs_mask])}"
    )
    print(
        f"  RS:      {summary_line(result.pk_stat[result.rs_mask], result.pk_move[result.rs_mask])}"
    )


def plot_panel(ax, result: ModeResult, *, log_scale: bool) -> None:
    keep = result.fs_mask | result.rs_mask
    all_pos = np.concatenate([result.pk_stat[keep], result.pk_move[keep]])
    if log_scale:
        plot_stat = np.maximum(result.pk_stat, 0) + 0.1
        plot_move = np.maximum(result.pk_move, 0) + 0.1
        all_plot = np.concatenate([plot_stat[keep], plot_move[keep]])
        lim_lo = 0.1
        lim_hi = np.percentile(all_plot, 99) * 1.05 if all_plot.size else 100.0
        lim_hi = max(1.0, float(lim_hi))
    else:
        plot_stat = result.pk_stat
        plot_move = result.pk_move
        lim_lo = 0.0
        lim_hi = np.percentile(all_pos, 99) * 1.05 if all_pos.size else 5.0
        lim_hi = max(5.0, float(lim_hi))

    ax.scatter(
        plot_stat[result.fs_mask],
        plot_move[result.fs_mask],
        s=18,
        alpha=0.55,
        color=COL_FS,
        label=f"FS (n={int(result.fs_mask.sum())})",
    )
    ax.scatter(
        plot_stat[result.rs_mask],
        plot_move[result.rs_mask],
        s=18,
        alpha=0.55,
        color=COL_RS,
        label=f"RS (n={int(result.rs_mask.sum())})",
    )
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, lw=0.8)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
    ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
    ax.set_title(f"{result.subject}  {result.mode_label}", fontsize=10)
    ax.text(
        0.97,
        0.03,
        "\n".join(
            [
                f"stat ev={result.n_stat_events}",
                f"move ev={result.n_move_events}",
                f"missing waveform={int(result.missing_mask.sum())}",
                result.baseline_label,
                summary_line(
                    result.pk_stat[result.fs_mask | result.rs_mask],
                    result.pk_move[result.fs_mask | result.rs_mask],
                ),
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, lw=0.0),
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def make_figure(
    results: list[ModeResult], baseline_label: str, *, log_scale: bool
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, result in zip(axes.flat, results):
        plot_panel(ax, result, log_scale=log_scale)
    fig.suptitle(
        "Niell-style locomotion approximation for flash responses\n"
        f"{baseline_label}; FS <= 0.4 ms (blue), RS > 0.4 ms (green)"
        + ("; log scale" if log_scale else ""),
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.07, top=0.90, wspace=0.28, hspace=0.28
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-stat-baseline",
        action="store_true",
        help="Use the stationary pre-stim baseline for both stationary and movement peaks.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Save a log-scale scatter version to a separate output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shared_stat_baseline and args.log_scale:
        out_path = OUT_PATH_SHARED_STAT_BASELINE_LOG
    elif args.shared_stat_baseline:
        out_path = OUT_PATH_SHARED_STAT_BASELINE
    elif args.log_scale:
        out_path = OUT_PATH_LOG
    else:
        out_path = OUT_PATH
    baseline_label = (
        "Shared stationary baseline"
        if args.shared_stat_baseline
        else "Condition-specific baseline"
    )

    grb006 = load_grb006()
    grb058 = load_grb058()

    results = [
        analyze_mode(
            grb006,
            mode="first_paired",
            mode_label="First stat vs first move",
            shared_stat_baseline=args.shared_stat_baseline,
        ),
        analyze_mode(
            grb058,
            mode="first_paired",
            mode_label="First stat vs first move",
            shared_stat_baseline=args.shared_stat_baseline,
        ),
        analyze_mode(
            grb006,
            mode="paired",
            mode_label="Last stat vs first move",
            shared_stat_baseline=args.shared_stat_baseline,
        ),
        analyze_mode(
            grb058,
            mode="paired",
            mode_label="Last stat vs first move",
            shared_stat_baseline=args.shared_stat_baseline,
        ),
    ]

    for result in results:
        print_mode_summary(result)

    fig = make_figure(results, baseline_label=baseline_label, log_scale=args.log_scale)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
