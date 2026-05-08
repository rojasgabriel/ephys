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

STIM_RATES = np.arange(4, 21)
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


def plot_population_tuning(tuning_curves: pd.DataFrame):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    palette = sns.color_palette("Set1", n_colors=tuning_curves["subject"].nunique())
    grouped = tuning_curves.groupby(
        ["subject", "session", "stim_rate_vision"], as_index=False
    ).agg(
        mean_sp_s=("mean_sp_s", "mean"),
        sem_units=("mean_sp_s", "sem"),
        n_units=("unit_id", "nunique"),
    )
    grouped["sem_units"] = grouped["sem_units"].fillna(0.0)

    for color, ((subject, session), session_df) in zip(
        palette,
        grouped.groupby(["subject", "session"]),
        strict=False,
    ):
        session_df = session_df.sort_values("stim_rate_vision")
        x = session_df["stim_rate_vision"].to_numpy(dtype=float)
        y = session_df["mean_sp_s"].to_numpy(dtype=float)
        sem = session_df["sem_units"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, label=f"{subject} {session}")
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)

    ax.set(
        xlabel="Visual stimulus rate (Hz)",
        ylabel="Mean firing rate from first flash to response (sp/s)",
        title="Population mean task-period response",
        xticks=STIM_RATES,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_unit_heatmap(tuning_curves: pd.DataFrame, normalization: str):
    from matplotlib import pyplot as plt

    if normalization == "minmax":
        cbar_label = "Row-normalized mean sp/s"
        title = "Unit rate tuning shape"
    elif normalization == "zscore":
        cbar_label = "Within-unit z-score"
        title = "Unit rate tuning z-score"
    else:
        raise ValueError("normalization must be 'minmax' or 'zscore'")

    session_groups = list(tuning_curves.groupby(["subject", "session"], sort=False))
    unit_counts = [group["unit_id"].nunique() for _, group in session_groups]
    height_ratios = [max(1, count) for count in unit_counts]
    fig_height = max(5.0, min(14.0, 1.4 + 0.025 * sum(unit_counts)))
    fig, axes = plt.subplots(
        len(session_groups),
        1,
        figsize=(7.0, fig_height),
        sharex=False,
        sharey=False,
        height_ratios=height_ratios,
    )
    axes_array = np.atleast_1d(axes).ravel()

    for ax, ((subject, session), session_tuning) in zip(
        axes_array,
        session_groups,
        strict=False,
    ):
        displayed_rates = np.asarray(
            sorted(session_tuning["stim_rate_vision"].unique())
        )
        matrix = pivot_tuning(session_tuning, rates=displayed_rates)
        if normalization == "minmax":
            row_max = matrix.max(axis=1).replace(0, np.nan)
            row_min = matrix.min(axis=1)
            values = matrix.sub(row_min, axis=0).div(row_max - row_min, axis=0)
        else:
            row_mean = matrix.mean(axis=1)
            row_std = matrix.std(axis=1).replace(0, np.nan)
            values = matrix.sub(row_mean, axis=0).div(row_std, axis=0)

        preferred_rate = matrix.idxmax(axis=1).fillna(np.inf)
        row_mean = matrix.mean(axis=1).fillna(0.0)
        sort_index = (
            pd.DataFrame({"preferred_rate": preferred_rate, "row_mean": row_mean})
            .sort_values(["preferred_rate", "row_mean"], ascending=[True, False])
            .index
        )
        values = values.loc[sort_index]

        sns.heatmap(
            values,
            ax=ax,
            cmap="viridis" if normalization == "minmax" else "vlag",
            center=0 if normalization == "zscore" else None,
            cbar=ax is axes_array[-1],
            cbar_kws={"label": cbar_label},
            xticklabels=[str(int(rate)) for rate in displayed_rates],
            yticklabels=False,
        )
        ax.set(
            xlabel="Visual stimulus rate (Hz)",
            ylabel="Units sorted by preferred rate",
            title=f"{subject} {session}\n{values.shape[0]} units",
        )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def choose_example_units(unit_summary: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    examples: list[tuple[str, pd.Series]] = []
    summary = unit_summary.copy()
    if summary.empty:
        return examples

    high = summary.sort_values(
        ["preferred_stim_rate", "tuning_range_sp_s"], ascending=[False, False]
    ).iloc[0]
    low = summary.sort_values(
        ["preferred_stim_rate", "tuning_range_sp_s"], ascending=[True, False]
    ).iloc[0]
    strongest = summary.sort_values("tuning_range_sp_s", ascending=False).iloc[0]
    flattest = summary.sort_values("tuning_range_sp_s", ascending=True).iloc[0]

    for label, row in [
        ("Low-rate preferring", low),
        ("High-rate preferring", high),
        ("Largest range", strongest),
        ("Flattest", flattest),
    ]:
        key = (row["subject"], row["session"], row["unit_id"])
        if not any(
            (ex[1]["subject"], ex[1]["session"], ex[1]["unit_id"]) == key
            for ex in examples
        ):
            examples.append((label, row))
    return examples[:4]


def plot_example_units(tuning_curves: pd.DataFrame, unit_summary: pd.DataFrame):
    from matplotlib import pyplot as plt

    examples = choose_example_units(unit_summary)
    n_examples = max(1, len(examples))
    fig, axes = plt.subplots(
        1, n_examples, figsize=(3.2 * n_examples, 3.4), sharey=False
    )
    axes_array = np.atleast_1d(axes)
    for ax, (label, row) in zip(axes_array, examples, strict=False):
        unit_df = tuning_curves[
            (tuning_curves["subject"] == row["subject"])
            & (tuning_curves["session"] == row["session"])
            & (tuning_curves["unit_id"] == row["unit_id"])
        ].sort_values("stim_rate_vision")
        x = unit_df["stim_rate_vision"].to_numpy(dtype=float)
        y = unit_df["mean_sp_s"].to_numpy(dtype=float)
        sem = unit_df["sem_sp_s"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=sem, marker="o", color="black", linewidth=1.2, capsize=2)
        ax.set(
            title=f"{label}\n{row['subject']} {row['session']}\nunit {int(row['unit_id'])}",
            xlabel="Rate (Hz)",
            xticks=[4, 8, 12, 16, 20],
        )
    axes_array[0].set_ylabel("Mean sp/s")
    fig.tight_layout()
    return fig


def plot_diagnostics(windows: pd.DataFrame):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    trial_counts = (
        windows.groupby(["subject", "session", "stim_rate_vision"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
    )
    sns.lineplot(
        data=trial_counts,
        x="stim_rate_vision",
        y="n_trials",
        hue="session",
        style="subject",
        marker="o",
        ax=axes[0],
    )
    axes[0].set(
        xlabel="Visual stimulus rate (Hz)",
        ylabel="Valid trials",
        title="Trial counts",
        xticks=STIM_RATES,
    )
    duration = (
        windows.groupby(["subject", "session", "stim_rate_vision"], as_index=False)
        .agg(mean_duration_s=("window_duration_s", "mean"))
        .sort_values("stim_rate_vision")
    )
    sns.lineplot(
        data=duration,
        x="stim_rate_vision",
        y="mean_duration_s",
        hue="session",
        style="subject",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set(
        xlabel="Visual stimulus rate (Hz)",
        ylabel="Mean window duration (s)",
        title="First flash to response",
        xticks=STIM_RATES,
    )
    sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def heatmap_values(
    session_tuning: pd.DataFrame,
    normalization: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    displayed_rates = np.asarray(sorted(session_tuning["stim_rate_vision"].unique()))
    matrix = pivot_tuning(session_tuning, rates=displayed_rates)
    if normalization == "minmax":
        row_max = matrix.max(axis=1).replace(0, np.nan)
        row_min = matrix.min(axis=1)
        values = matrix.sub(row_min, axis=0).div(row_max - row_min, axis=0)
    elif normalization == "zscore":
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).replace(0, np.nan)
        values = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
    else:
        raise ValueError("normalization must be 'minmax' or 'zscore'")

    preferred_rate = matrix.idxmax(axis=1).fillna(np.inf)
    row_mean = matrix.mean(axis=1).fillna(0.0)
    sort_index = (
        pd.DataFrame({"preferred_rate": preferred_rate, "row_mean": row_mean})
        .sort_values(["preferred_rate", "row_mean"], ascending=[True, False])
        .index
    )
    return values.loc[sort_index], displayed_rates


def plot_population_tuning_on_ax(tuning_curves: pd.DataFrame, ax) -> None:
    palette = sns.color_palette("Set1", n_colors=tuning_curves["subject"].nunique())
    grouped = tuning_curves.groupby(
        ["subject", "session", "stim_rate_vision"], as_index=False
    ).agg(
        mean_sp_s=("mean_sp_s", "mean"),
        sem_units=("mean_sp_s", "sem"),
        n_units=("unit_id", "nunique"),
    )
    grouped["sem_units"] = grouped["sem_units"].fillna(0.0)

    for color, ((subject, session), session_df) in zip(
        palette,
        grouped.groupby(["subject", "session"]),
        strict=False,
    ):
        session_df = session_df.sort_values("stim_rate_vision")
        x = session_df["stim_rate_vision"].to_numpy(dtype=float)
        y = session_df["mean_sp_s"].to_numpy(dtype=float)
        sem = session_df["sem_units"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, label=f"{subject} {session}")
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)

    ax.set(
        xlabel="Visual rate (Hz)",
        ylabel="Mean sp/s",
        title="Population response",
        xticks=STIM_RATES,
    )
    ax.legend(frameon=False, fontsize=6, loc="best")


def plot_diagnostics_on_axes(windows: pd.DataFrame, count_ax, duration_ax) -> None:
    trial_counts = (
        windows.groupby(["subject", "session", "stim_rate_vision"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
    )
    sns.lineplot(
        data=trial_counts,
        x="stim_rate_vision",
        y="n_trials",
        hue="session",
        style="subject",
        marker="o",
        ax=count_ax,
        legend=False,
    )
    count_ax.set(
        xlabel="Visual rate (Hz)",
        ylabel="Valid trials",
        title="Trial counts",
        xticks=STIM_RATES,
    )
    duration = (
        windows.groupby(["subject", "session", "stim_rate_vision"], as_index=False)
        .agg(mean_duration_s=("window_duration_s", "mean"))
        .sort_values("stim_rate_vision")
    )
    sns.lineplot(
        data=duration,
        x="stim_rate_vision",
        y="mean_duration_s",
        hue="session",
        style="subject",
        marker="o",
        ax=duration_ax,
        legend=False,
    )
    duration_ax.set(
        xlabel="Visual rate (Hz)",
        ylabel="Mean duration (s)",
        title="First flash to response",
        xticks=STIM_RATES,
    )


def plot_example_unit_on_ax(
    tuning_curves: pd.DataFrame,
    label: str,
    row: pd.Series,
    ax,
) -> None:
    unit_df = tuning_curves[
        (tuning_curves["subject"] == row["subject"])
        & (tuning_curves["session"] == row["session"])
        & (tuning_curves["unit_id"] == row["unit_id"])
    ].sort_values("stim_rate_vision")
    x = unit_df["stim_rate_vision"].to_numpy(dtype=float)
    y = unit_df["mean_sp_s"].to_numpy(dtype=float)
    sem = unit_df["sem_sp_s"].to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=sem, marker="o", color="black", linewidth=1.0, capsize=2)
    ax.set(
        title=f"{label}\n{row['subject']} unit {int(row['unit_id'])}",
        xlabel="Rate (Hz)",
        xticks=[4, 8, 12, 16, 20],
    )


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

    (subject, session), session_tuning = session_groups[0]
    values, displayed_rates = heatmap_values(session_tuning, "zscore")
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
        xlabel="Visual stimulus rate (Hz)",
        ylabel="Units sorted by preferred rate",
        title=f"{subject} {session}: task-period rate tuning ({values.shape[0]} units)",
    )
    fig.tight_layout()

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
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
