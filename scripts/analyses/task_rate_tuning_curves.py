"""First-pass task stimulus-period rate tuning curves.

For each good unit, this analysis measures mean firing rate from the first
15 ms visual flash in a trial to response-port entry, then groups responses
by visual stimulus rate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.analysis_rate_tuning import (  # noqa: E402
    aggregate_tuning_curves,
    build_task_stimulus_windows,
    compute_trial_responses,
    summarize_units,
)
from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata  # noqa: E402
from ephys.src.utils.io_digital_events import fetch_session_events  # noqa: E402
from ephys.src.utils.io_session_units import fetch_good_units  # noqa: E402

FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
FIGURE_DIR = FIGURE_ROOT / "task_rate_tuning"

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
]

UNIT_CRITERIA_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for CSV and PDF outputs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build tables and figures without writing outputs.",
    )
    return parser.parse_args()


def load_session_tables(subject: str, session: str) -> tuple[pd.DataFrame, ...]:
    print(f"\nLoading {subject} {session}")
    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}.")
    spike_times_by_unit = fetch_good_units(
        subject,
        session,
        unit_criteria_id=UNIT_CRITERIA_ID,
    )
    windows = build_task_stimulus_windows(align_ev, trial_df)
    if windows.empty:
        raise RuntimeError(f"No valid task stimulus windows for {subject} {session}.")
    trial_responses = compute_trial_responses(windows, spike_times_by_unit)
    tuning_curves = aggregate_tuning_curves(trial_responses)
    unit_summary = summarize_units(tuning_curves)

    for table in (windows, trial_responses, tuning_curves, unit_summary):
        table.insert(0, "session", session)
        table.insert(0, "subject", subject)

    print(f"  Units: {len(spike_times_by_unit)}")
    print(f"  Valid trials: {len(windows)}")
    print(
        "  Rates: "
        + ", ".join(
            str(int(rate)) for rate in sorted(windows["stim_rate_vision"].unique())
        )
    )
    return windows, trial_responses, tuning_curves, unit_summary


def pivot_tuning(
    tuning_curves: pd.DataFrame,
    value_column: str = "mean_sp_s",
    rates: np.ndarray | None = None,
) -> pd.DataFrame:
    pivot = tuning_curves.pivot_table(
        index=["subject", "session", "unit_id"],
        columns="stim_rate_vision",
        values=value_column,
    )
    if rates is None:
        rates = np.asarray(sorted(tuning_curves["stim_rate_vision"].unique()))
    return pivot.reindex(columns=rates)


def zscore_heatmap_values(
    session_tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Row-wise z-score of mean sp/s tuning, sorted by preferred rate."""
    displayed_rates = np.asarray(sorted(session_tuning["stim_rate_vision"].unique()))
    matrix = pivot_tuning(session_tuning, rates=displayed_rates)
    row_mean = matrix.mean(axis=1)
    row_std = matrix.std(axis=1).replace(0, np.nan)
    values = matrix.sub(row_mean, axis=0).div(row_std, axis=0)

    preferred_rate = matrix.idxmax(axis=1).fillna(np.inf)
    row_mean_for_sort = matrix.mean(axis=1).fillna(0.0)
    sort_index = (
        pd.DataFrame({"preferred_rate": preferred_rate, "row_mean": row_mean_for_sort})
        .sort_values(["preferred_rate", "row_mean"], ascending=[True, False])
        .index
    )
    return values.loc[sort_index], displayed_rates


def choose_example_curve_units(
    unit_summary: pd.DataFrame,
    n_examples: int = 6,
) -> pd.DataFrame:
    """Pick high-modulation examples, spreading preferred rates when possible."""
    if unit_summary.empty:
        return unit_summary.copy()

    summary = unit_summary.sort_values(
        ["tuning_range_sp_s", "mean_sp_s_all_rates"],
        ascending=[False, False],
    )
    selected_rows = []
    selected_rates = set()
    for _, row in summary.iterrows():
        preferred_rate = float(row["preferred_stim_rate"])
        if preferred_rate in selected_rates:
            continue
        selected_rows.append(row)
        selected_rates.add(preferred_rate)
        if len(selected_rows) == n_examples:
            break

    if len(selected_rows) < n_examples:
        selected_keys = {
            (row["subject"], row["session"], int(row["unit_id"]))
            for row in selected_rows
        }
        for _, row in summary.iterrows():
            key = (row["subject"], row["session"], int(row["unit_id"]))
            if key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(key)
            if len(selected_rows) == n_examples:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_example_curves(
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
):
    from matplotlib import pyplot as plt

    examples = choose_example_curve_units(unit_summary, n_examples=6)
    if examples.empty:
        raise RuntimeError("No units available for example curves.")

    rates = np.asarray(sorted(tuning_curves["stim_rate_vision"].unique()))
    curve_rows = []
    for row in examples.itertuples(index=False):
        unit_df = tuning_curves[
            (tuning_curves["subject"] == row.subject)
            & (tuning_curves["session"] == row.session)
            & (tuning_curves["unit_id"] == row.unit_id)
        ].sort_values("stim_rate_vision")
        values = (
            unit_df.set_index("stim_rate_vision")
            .reindex(rates)["mean_sp_s"]
            .to_numpy(dtype=float)
        )
        value_mean = np.nanmean(values)
        value_std = np.nanstd(values, ddof=1)
        z_values = (values - value_mean) / value_std
        curve_rows.append((row, z_values))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 7.0),
        sharex=True,
        sharey=True,
    )
    color = sns.color_palette("Set1", n_colors=1)[0]
    for ax, (row, z_values) in zip(axes.ravel(), curve_rows, strict=False):
        ax.plot(rates, z_values, color=color, marker="o", linewidth=1.5)
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
        ax.grid(False)
        ax.set_xticks(rates)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)

    for ax in axes[:, 0]:
        ax.set_ylabel("z-score")
    for ax in axes[-1, :]:
        ax.set_xlabel("stimulus rate (Hz)")
    fig.tight_layout()
    return fig


def write_pdf(
    output_path: Path,
    windows: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    del windows, unit_summary
    session_groups = list(tuning_curves.groupby(["subject", "session"], sort=False))
    if len(session_groups) != 1:
        raise RuntimeError("The heatmap-only figure expects exactly one session.")

    (_subject, _session), session_tuning = session_groups[0]
    values, displayed_rates = zscore_heatmap_values(session_tuning)
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    sns.heatmap(
        values,
        ax=ax,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z-score within unit"},
        xticklabels=[str(int(rate)) for rate in displayed_rates],
        yticklabels=False,
    )
    ax.set(
        xlabel="stimulus rate (Hz)",
        ylabel="units sorted by preferred rate",
    )
    fig.tight_layout()

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def write_example_curves_pdf(
    output_path: Path,
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_example_curves(tuning_curves, unit_summary)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_rate_response_scatter(tuning_curves: pd.DataFrame):
    from matplotlib import pyplot as plt

    scatter_df = tuning_curves.copy()
    scatter_df["z_mean_sp_s"] = scatter_df.groupby("unit_id")["mean_sp_s"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=1)
    )
    x = scatter_df["stim_rate_vision"].to_numpy(dtype=float)
    y = scatter_df["z_mean_sp_s"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    correlation = np.corrcoef(x, y)[0, 1]
    fit_x = np.linspace(float(x.min()), float(x.max()), 200)
    fit_y = slope * fit_x + intercept

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    line_color = sns.color_palette("Set1", n_colors=1)[0]
    ax.scatter(x, y, s=14, alpha=0.18, color="black", linewidths=0)
    ax.plot(fit_x, fit_y, color=line_color, linewidth=2, linestyle="--")
    ax.grid(False)
    ax.text(
        0.98,
        0.96,
        f"r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.set(
        xlabel="stimulus rate (Hz)",
        ylabel="z-score",
        xticks=sorted(tuning_curves["stim_rate_vision"].unique()),
    )
    fig.tight_layout()
    return fig


def write_rate_scatter_pdf(output_path: Path, tuning_curves: pd.DataFrame) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_rate_response_scatter(tuning_curves)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_frequency_selectivity_box(unit_summary: pd.DataFrame):
    from matplotlib import pyplot as plt

    plot_df = unit_summary.dropna(subset=["frequency_selectivity_index"]).copy()
    if plot_df.empty:
        raise RuntimeError("No finite FSI values available for box plot.")

    plot_df["population"] = (
        plot_df["subject"].astype(str) + "\n" + plot_df["session"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(1.9, 3.9))
    sns.boxplot(
        data=plot_df,
        x="population",
        y="frequency_selectivity_index",
        color="0.8",
        width=0.45,
        linewidth=1.0,
        fliersize=2.0,
        ax=ax,
    )
    ax.set(
        xlabel="population",
        ylabel="FSI",
        ylim=(-0.02, 1.02),
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def write_frequency_selectivity_box_pdf(
    output_path: Path,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_frequency_selectivity_box(unit_summary)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper")

    all_windows = []
    all_trial_responses = []
    all_tuning_curves = []
    all_unit_summaries = []
    for subject, session in SUBJECT_SESSIONS:
        windows, trial_responses, tuning_curves, unit_summary = load_session_tables(
            subject,
            session,
        )
        all_windows.append(windows)
        all_trial_responses.append(trial_responses)
        all_tuning_curves.append(tuning_curves)
        all_unit_summaries.append(unit_summary)

    windows_df = pd.concat(all_windows, ignore_index=True)
    trial_responses_df = pd.concat(all_trial_responses, ignore_index=True)
    tuning_curves_df = pd.concat(all_tuning_curves, ignore_index=True)
    unit_summary_df = pd.concat(all_unit_summaries, ignore_index=True)

    if args.no_save:
        print("\nBuilt rate tuning tables and figures without writing outputs.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_responses_df.to_csv(args.output_dir / "trial_responses.csv", index=False)
    tuning_curves_df.to_csv(args.output_dir / "unit_tuning_curves.csv", index=False)
    unit_summary_df.to_csv(args.output_dir / "unit_summary.csv", index=False)
    pdf_path = args.output_dir / "rate_tuning_curves.pdf"
    write_pdf(pdf_path, windows_df, tuning_curves_df, unit_summary_df)
    example_pdf_path = args.output_dir / "rate_tuning_example_curves.pdf"
    write_example_curves_pdf(example_pdf_path, tuning_curves_df, unit_summary_df)
    scatter_pdf_path = args.output_dir / "rate_tuning_rate_response_scatter.pdf"
    write_rate_scatter_pdf(scatter_pdf_path, tuning_curves_df)
    fsi_pdf_path = args.output_dir / "frequency_selectivity_index_box.pdf"
    write_frequency_selectivity_box_pdf(fsi_pdf_path, unit_summary_df)
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
