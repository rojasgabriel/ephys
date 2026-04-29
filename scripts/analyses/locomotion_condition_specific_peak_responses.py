"""Single-surface locomotion overlay using condition-specific peak responses.

This script asks
whether movement changes visually evoked responses when each condition is
allowed to keep its own response latency.

The current comparison is:
  - paired last stationary vs first movement within the same trial

For each unit and condition:
  - build the mean PSTH across selected events
  - subtract the pre-stim baseline
  - take the peak value in the response window

It writes the compact single-overlay condition-peak scatter. The FS/RS split
remains available through an explicit CLI flag for follow-up reruns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib import colormaps
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys.src.config.locomotion import BASELINE_WINDOW, PETH_KWARGS, RESP_WINDOW
from ephys.src.utils.grb006_data import load_grb006_hybrid_session_inputs
from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table
from ephys.src.utils.unit_metrics import fetch_waveform_durations_ms
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
)

FIGURE_DIR = Path("/Users/gabriel/lib/ephys/figures/locomotion")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FS_RS_BOUNDARY_MS = 0.4
MODE_SPECS = [
    (
        "paired_last_stat_first_move",
        "Last stat vs first move (paired)",
        "condition_peak_paired_last_stat_first_move",
    ),
]

BACKGROUND_DOT_ALPHA = 0.2
MEAN_CI_LEVEL = 0.95


def mean_and_t_ci(values: np.ndarray, *, log_scale: bool) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("mean_and_t_ci requires at least one value.")

    if values.size == 1:
        mean_value = float(values[0])
        return mean_value, mean_value, mean_value

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


def load_db_trial_classification(subject: str, session: str) -> pd.DataFrame:
    from ephys.src.utils.utils_IO import fetch_session_events, fetch_trial_metadata

    aligned_events = fetch_session_events(subject, session)
    trial_table = fetch_trial_metadata(subject, session, aligned_events)
    if trial_table is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}.")
    trial_table = enrich_chipmunk_trial_table(trial_table)
    return build_trial_stim_classification(aligned_events, trial_table).reset_index(
        drop=True
    )


def load_grb006_subject_data() -> dict[str, object]:
    subject, session = "GRB006", "20240821_121447"
    print(f"\nLoading hybrid session: {subject} {session}")
    unit_ids, spike_times_by_unit, chipmunk_trial_table, local_trial_table = (
        load_grb006_hybrid_session_inputs()
    )
    waveform_duration_ms = fetch_waveform_durations_ms(
        subject, session, unit_ids, strict=True
    )
    print(
        f"  Units: {len(unit_ids)}  Trials: {len(chipmunk_trial_table)}  "
        f"Local rows: {len(local_trial_table)}"
    )
    return {
        "subject": subject,
        "session": session,
        "unit_ids": unit_ids,
        "spike_times_by_unit": spike_times_by_unit,
        "waveform_duration_ms": waveform_duration_ms,
        "trial_classification": local_trial_table,
    }


def load_grb058_subject_data() -> dict[str, object]:
    from ephys.src.utils.utils_IO import fetch_good_units

    subject, session = "GRB058", "20260312_134952"
    print(f"\nLoading DB session: {subject} {session}")
    trial_classification = load_db_trial_classification(subject, session)
    spike_times_by_unit_map = fetch_good_units(subject, session)
    unit_ids = list(spike_times_by_unit_map.keys())
    spike_times_by_unit = list(spike_times_by_unit_map.values())
    waveform_duration_ms = fetch_waveform_durations_ms(
        subject, session, unit_ids, strict=True
    )
    print(f"  Units: {len(unit_ids)}  Trial rows: {len(trial_classification)}")
    return {
        "subject": subject,
        "session": session,
        "unit_ids": unit_ids,
        "spike_times_by_unit": spike_times_by_unit,
        "waveform_duration_ms": waveform_duration_ms,
        "trial_classification": trial_classification,
    }


def mean_psth_and_baseline_rate(
    population_psth: np.ndarray, bin_centers_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    baseline_mask = (bin_centers_s >= BASELINE_WINDOW[0]) & (
        bin_centers_s < BASELINE_WINDOW[1]
    )
    mean_rate_by_bin = population_psth.mean(axis=1)
    baseline_rate = mean_rate_by_bin[:, baseline_mask].mean(axis=1, keepdims=True)
    return mean_rate_by_bin, baseline_rate


def peak_response_from_mean_psth(
    mean_rate_by_bin: np.ndarray,
    bin_centers_s: np.ndarray,
    reference_baseline_rate: np.ndarray,
) -> np.ndarray:
    response_mask = (bin_centers_s >= RESP_WINDOW[0]) & (bin_centers_s < RESP_WINDOW[1])
    baseline_corrected_response = (
        mean_rate_by_bin[:, response_mask] - reference_baseline_rate
    )
    return baseline_corrected_response.max(axis=1)


def select_mode_event_times(
    trial_classification: pd.DataFrame, mode_name: str
) -> tuple[np.ndarray, np.ndarray]:
    has_stationary = trial_classification["stationary_stims"].apply(
        lambda x: len(x) > 0
    )
    has_movement = trial_classification["movement_stims"].apply(lambda x: len(x) > 0)

    paired_trial_rows = trial_classification[has_stationary & has_movement]
    if mode_name == "paired_last_stat_first_move":
        stationary_event_times = np.array(
            [stim_times[-1] for stim_times in paired_trial_rows["stationary_stims"]],
            dtype=float,
        )
        movement_event_times = np.array(
            [stim_times[0] for stim_times in paired_trial_rows["movement_stims"]],
            dtype=float,
        )
        return stationary_event_times, movement_event_times

    raise ValueError(f"Unknown mode_name: {mode_name}")


def analyze_subject_mode(
    subject_data: dict[str, object],
    mode_name: str,
    mode_label: str,
    *,
    shared_stationary_baseline: bool,
) -> dict[str, object]:
    stationary_event_times, movement_event_times = select_mode_event_times(
        subject_data["trial_classification"], mode_name
    )
    if len(stationary_event_times) == 0 or len(movement_event_times) == 0:
        raise RuntimeError(
            f"No events found for {subject_data['subject']} mode={mode_name}."
        )

    stationary_population_psth, _, bin_centers_s = compute_population_peth(
        subject_data["spike_times_by_unit"], stationary_event_times, **PETH_KWARGS
    )
    movement_population_psth, _, _ = compute_population_peth(
        subject_data["spike_times_by_unit"], movement_event_times, **PETH_KWARGS
    )

    stationary_mean_rate, stationary_baseline_rate = mean_psth_and_baseline_rate(
        stationary_population_psth, bin_centers_s
    )
    movement_mean_rate, movement_baseline_rate = mean_psth_and_baseline_rate(
        movement_population_psth, bin_centers_s
    )
    movement_reference_baseline = (
        stationary_baseline_rate
        if shared_stationary_baseline
        else movement_baseline_rate
    )

    stationary_peak_response = peak_response_from_mean_psth(
        stationary_mean_rate, bin_centers_s, stationary_baseline_rate
    )
    movement_peak_response = peak_response_from_mean_psth(
        movement_mean_rate, bin_centers_s, movement_reference_baseline
    )

    waveform_duration_ms = np.asarray(subject_data["waveform_duration_ms"], dtype=float)
    if not np.all(np.isfinite(waveform_duration_ms)):
        raise RuntimeError(
            f"Non-finite waveform durations encountered for {subject_data['subject']}."
        )
    fast_spiking_unit_mask = waveform_duration_ms <= FS_RS_BOUNDARY_MS
    regular_spiking_unit_mask = waveform_duration_ms > FS_RS_BOUNDARY_MS
    if not np.all(fast_spiking_unit_mask | regular_spiking_unit_mask):
        raise RuntimeError(
            f"Waveform class assignment failed for {subject_data['subject']}."
        )

    return {
        "subject": subject_data["subject"],
        "mode_label": mode_label,
        "baseline_label": (
            "Shared stationary baseline"
            if shared_stationary_baseline
            else "Condition-specific baseline"
        ),
        "stationary_peak_response": stationary_peak_response,
        "movement_peak_response": movement_peak_response,
        "fast_spiking_unit_mask": fast_spiking_unit_mask,
        "regular_spiking_unit_mask": regular_spiking_unit_mask,
        "n_stationary_events": len(stationary_event_times),
        "n_movement_events": len(movement_event_times),
    }


def print_mode_summary(mode_result: dict[str, object]) -> None:
    print(f"\n{mode_result['subject']}  {mode_result['mode_label']}")
    print(f"  stationary events: {mode_result['n_stationary_events']}")
    print(f"  movement events:   {mode_result['n_movement_events']}")
    print(f"  FS units:          {int(np.sum(mode_result['fast_spiking_unit_mask']))}")
    print(
        f"  RS units:          {int(np.sum(mode_result['regular_spiking_unit_mask']))}"
    )


def plot_panel(
    ax: plt.Axes,
    mode_results: list[dict[str, object]],
    *,
    log_scale: bool,
    split_by_waveform: bool,
) -> None:
    subject_colors = colormaps["Set1"].colors
    plotted_pairs: list[np.ndarray] = []
    for mode_result in mode_results:
        stationary_peak_response = np.asarray(mode_result["stationary_peak_response"])
        movement_peak_response = np.asarray(mode_result["movement_peak_response"])
        if stationary_peak_response.size and movement_peak_response.size:
            plotted_pairs.append(
                np.concatenate([stationary_peak_response, movement_peak_response])
            )

    if log_scale:
        plotted_values = []
        for mode_result in mode_results:
            stationary_peak_response = np.asarray(
                mode_result["stationary_peak_response"]
            )
            movement_peak_response = np.asarray(mode_result["movement_peak_response"])
            plotted_values.append(np.maximum(stationary_peak_response, 0) + 0.1)
            plotted_values.append(np.maximum(movement_peak_response, 0) + 0.1)
        all_peak_values = (
            np.concatenate(plotted_values) if plotted_values else np.array([])
        )
        lower_limit = 0.1
        upper_limit = (
            np.percentile(all_peak_values, 99) * 1.05 if all_peak_values.size else 100.0
        )
        upper_limit = max(1.0, float(upper_limit))
    else:
        all_peak_values = (
            np.concatenate(plotted_pairs) if plotted_pairs else np.array([])
        )
        lower_limit = 0.0
        upper_limit = (
            np.percentile(all_peak_values, 99) * 1.05 if all_peak_values.size else 5.0
        )
        upper_limit = max(5.0, float(upper_limit))

    if split_by_waveform:
        marker_specs = [
            ("regular_spiking_unit_mask", "o", "RS"),
            ("fast_spiking_unit_mask", "^", "FS"),
        ]
    else:
        marker_specs = [("all_unit_mask", "o", "All units")]
    for subject_index, mode_result in enumerate(mode_results):
        color = subject_colors[subject_index]
        stationary_peak_response = np.asarray(mode_result["stationary_peak_response"])
        movement_peak_response = np.asarray(mode_result["movement_peak_response"])
        all_unit_mask = np.ones_like(stationary_peak_response, dtype=bool)
        if log_scale:
            plotted_stationary = np.maximum(stationary_peak_response, 0) + 0.1
            plotted_movement = np.maximum(movement_peak_response, 0) + 0.1
        else:
            plotted_stationary = stationary_peak_response
            plotted_movement = movement_peak_response

        for mask_key, marker, cell_class in marker_specs:
            class_mask = (
                all_unit_mask
                if mask_key == "all_unit_mask"
                else np.asarray(mode_result[mask_key], dtype=bool)
            )
            if not np.any(class_mask):
                continue
            ax.scatter(
                plotted_stationary[class_mask],
                plotted_movement[class_mask],
                s=18,
                alpha=BACKGROUND_DOT_ALPHA,
                color=color,
                marker=marker,
                linewidths=0,
                label="_nolegend_",
                zorder=2,
            )
            mean_x, lower_x, upper_x = mean_and_t_ci(
                plotted_stationary[class_mask], log_scale=log_scale
            )
            mean_y, lower_y, upper_y = mean_and_t_ci(
                plotted_movement[class_mask], log_scale=log_scale
            )
            xerr = np.array([[mean_x - lower_x], [upper_x - mean_x]])
            yerr = np.array([[mean_y - lower_y], [upper_y - mean_y]])
            ax.errorbar(
                mean_x,
                mean_y,
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
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
                label=(
                    f"{mode_result['subject']} {cell_class}"
                    if split_by_waveform
                    else mode_result["subject"]
                ),
            )

    ax.plot(
        [lower_limit, upper_limit], [lower_limit, upper_limit], "k--", alpha=0.4, lw=0.8
    )
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
    ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def make_mode_figure(
    mode_results: list[dict[str, object]], *, log_scale: bool, split_by_waveform: bool
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    plot_panel(
        ax, mode_results, log_scale=log_scale, split_by_waveform=split_by_waveform
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.98)
    return fig


def output_path_for_mode(
    mode_stem: str,
    *,
    shared_stationary_baseline: bool,
    log_scale: bool,
    split_by_waveform: bool,
) -> Path:
    suffix_parts = [mode_stem]
    if shared_stationary_baseline:
        suffix_parts.append("shared_stat_baseline")
    if not split_by_waveform:
        suffix_parts.append("no_waveform_split")
    if log_scale:
        suffix_parts.append("log")
    filename = "_".join(suffix_parts) + ".pdf"
    return FIGURE_DIR / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-specific-baseline",
        action="store_true",
        help="Use condition-specific baselines instead of the default shared stationary baseline.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Save a log-scale scatter version to a separate output path.",
    )
    parser.add_argument(
        "--split-by-waveform",
        action="store_true",
        help="Write the FS/RS-split overlay instead of the default no-waveform-split output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_stationary_baseline = not args.condition_specific_baseline

    grb006_subject_data = load_grb006_subject_data()
    grb058_subject_data = load_grb058_subject_data()

    for mode_name, mode_label, mode_stem in MODE_SPECS:
        mode_results = [
            analyze_subject_mode(
                grb006_subject_data,
                mode_name=mode_name,
                mode_label=mode_label,
                shared_stationary_baseline=shared_stationary_baseline,
            ),
            analyze_subject_mode(
                grb058_subject_data,
                mode_name=mode_name,
                mode_label=mode_label,
                shared_stationary_baseline=shared_stationary_baseline,
            ),
        ]
        for mode_result in mode_results:
            print_mode_summary(mode_result)

        split_by_waveform = args.split_by_waveform
        fig = make_mode_figure(
            mode_results,
            log_scale=args.log_scale,
            split_by_waveform=split_by_waveform,
        )
        output_path = output_path_for_mode(
            mode_stem,
            shared_stationary_baseline=shared_stationary_baseline,
            log_scale=args.log_scale,
            split_by_waveform=split_by_waveform,
        )
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
