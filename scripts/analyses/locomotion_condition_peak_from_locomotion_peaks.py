"""Locomotion condition-peak scatter using the LocomotionPeaks table.

This repeats the GRB006 / GRB058 condition-peak scatter, but uses the
precomputed LocomotionPeaks rows as the peak-response data source instead of
recomputing PETHs inside the script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]


def install_local_ephys_package_alias() -> None:
    if "ephys" in sys.modules:
        return
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package


install_local_ephys_package_alias()
sys.path.insert(0, str(REPO_ROOT))
from ephys.src.utils.analysis_stats import mean_and_t_ci  # noqa: E402
from ephys.src.utils.unit_metrics import fetch_waveform_durations_ms  # noqa: E402

FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
FIGURE_DIR = FIGURE_ROOT / "locomotion"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260312_134952"),
]

UNIT_CRITERIA_ID = 1
MODE_LABEL = "Last stat vs first move (paired)"
FS_RS_BOUNDARY_MS = 0.4
BACKGROUND_DOT_ALPHA = 0.2
MEAN_CI_LEVEL = 0.95


class LocomotionModeResult(TypedDict):
    subject: str
    mode_label: str
    baseline_label: str
    stationary_peak_response: np.ndarray
    movement_peak_response: np.ndarray
    fast_spiking_unit_mask: np.ndarray
    regular_spiking_unit_mask: np.ndarray
    n_stationary_events: int
    n_movement_events: int


def fetch_locomotion_peak_rows(subject: str, session: str) -> pd.DataFrame:
    from labdata_plugin.analysisschema import LocomotionPeaks

    relation = (
        LocomotionPeaks() * LocomotionPeaks.key_source
        & f'subject_name = "{subject}"'
        & f'session_name = "{session}"'
        & f"unit_criteria_id = {UNIT_CRITERIA_ID}"
        & "passes = 1"
    )
    rows = relation.fetch(
        "unit_id",
        "stat_peak",
        "move_peak",
        "stat_latency",
        "move_latency",
        as_dict=True,
    )
    peak_table = pd.DataFrame(rows)
    if peak_table.empty:
        raise RuntimeError(
            f"No LocomotionPeaks rows found for {subject} {session} "
            f"with unit_criteria_id={UNIT_CRITERIA_ID} and passes=1."
        )
    peak_table = peak_table.sort_values("unit_id").reset_index(drop=True)
    for column in ["stat_peak", "move_peak", "stat_latency", "move_latency"]:
        values = peak_table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite {column} values for {subject} {session}.")
    return peak_table


def load_table_mode_result(subject: str, session: str) -> LocomotionModeResult:
    print(f"\nLoading LocomotionPeaks rows: {subject} {session}")
    peak_table = fetch_locomotion_peak_rows(subject, session)
    unit_ids = peak_table["unit_id"].astype(int).tolist()
    waveform_duration_ms = fetch_waveform_durations_ms(
        subject, session, unit_ids, strict=True, unit_criteria_id=UNIT_CRITERIA_ID
    )
    fast_spiking_unit_mask = waveform_duration_ms <= FS_RS_BOUNDARY_MS
    regular_spiking_unit_mask = waveform_duration_ms > FS_RS_BOUNDARY_MS
    if not np.all(fast_spiking_unit_mask | regular_spiking_unit_mask):
        raise RuntimeError(f"Waveform class assignment failed for {subject}.")

    print(f"  Units: {len(peak_table)}")
    print(
        "  Median latency (stat, move): "
        f"{peak_table['stat_latency'].median():.3f}s, "
        f"{peak_table['move_latency'].median():.3f}s"
    )
    return {
        "subject": subject,
        "mode_label": MODE_LABEL,
        "baseline_label": "Shared stationary baseline",
        "stationary_peak_response": peak_table["stat_peak"].to_numpy(dtype=float),
        "movement_peak_response": peak_table["move_peak"].to_numpy(dtype=float),
        "fast_spiking_unit_mask": fast_spiking_unit_mask,
        "regular_spiking_unit_mask": regular_spiking_unit_mask,
        "n_stationary_events": 0,
        "n_movement_events": 0,
    }


def output_path(*, log_scale: bool, split_by_waveform: bool) -> Path:
    suffix_parts = [
        "condition_peak_from_locomotion_peaks",
        "paired_last_stat_first_move",
        "shared_stat_baseline",
    ]
    if not split_by_waveform:
        suffix_parts.append("no_waveform_split")
    if log_scale:
        suffix_parts.append("log")
    return FIGURE_DIR / ("_".join(suffix_parts) + ".pdf")


def plot_panel(
    ax: plt.Axes,
    mode_results: list[LocomotionModeResult],
    *,
    log_scale: bool,
    split_by_waveform: bool,
) -> None:
    subject_colors = sns.color_palette("Set1")
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
            if mask_key == "all_unit_mask":
                class_mask = all_unit_mask
            elif mask_key == "regular_spiking_unit_mask":
                class_mask = np.asarray(
                    mode_result["regular_spiking_unit_mask"], dtype=bool
                )
            elif mask_key == "fast_spiking_unit_mask":
                class_mask = np.asarray(
                    mode_result["fast_spiking_unit_mask"], dtype=bool
                )
            else:
                raise RuntimeError(f"unknown mask_key {mask_key}")
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
                plotted_stationary[class_mask],
                log_scale=log_scale,
                ci_level=MEAN_CI_LEVEL,
                drop_nonfinite=False,
            )
            mean_y, lower_y, upper_y = mean_and_t_ci(
                plotted_movement[class_mask],
                log_scale=log_scale,
                ci_level=MEAN_CI_LEVEL,
                drop_nonfinite=False,
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
    mode_results: list[LocomotionModeResult],
    *,
    log_scale: bool,
    split_by_waveform: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    plot_panel(
        ax, mode_results, log_scale=log_scale, split_by_waveform=split_by_waveform
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.98)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Save a linear-scale scatter version instead of the default log-scale output.",
    )
    parser.add_argument(
        "--split-by-waveform",
        action="store_true",
        help="Write the FS/RS-split overlay instead of the default no-waveform-split output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_results = [
        load_table_mode_result(subject, session)
        for subject, session in SUBJECT_SESSIONS
    ]
    fig = make_mode_figure(
        mode_results,
        log_scale=not args.linear_scale,
        split_by_waveform=args.split_by_waveform,
    )
    save_path = output_path(
        log_scale=not args.linear_scale, split_by_waveform=args.split_by_waveform
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
