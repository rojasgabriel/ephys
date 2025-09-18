#!/usr/bin/env python3
"""
Population response analysis for ephys data.
Creates a 2x2 grid comparing first stimulus responses and movement vs stationary responses.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join as pjoin
from spks.event_aligned import population_peth
from chiCa.chiCa.visualization_utils import separate_axes
import matplotlib.pyplot as plt
import matplotlib as mpl

new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)
plt.rcParams["font.sans-serif"] = ["Arial"]


def get_balanced_trials(trial_ts, require_both_stim_types=True):
    """Get equal numbers of rewarded and unrewarded trials."""
    # Get valid trials (exclude outcome -1)
    valid_trials = trial_ts[trial_ts.trial_outcome.isin([0, 1])]

    # Optionally require both stim types
    if require_both_stim_types:
        valid_trials = valid_trials[
            (valid_trials.movement_stims.apply(len) > 0)
            & (valid_trials.stationary_stims.apply(len) > 0)
        ]

    # Find minimum number of trials between conditions
    min_trials = min(
        len(valid_trials[valid_trials.trial_outcome == 1]),
        len(valid_trials[valid_trials.trial_outcome == 0]),
    )

    # Sample equal numbers from each condition
    balanced_trials = pd.concat(
        [
            valid_trials[valid_trials.trial_outcome == 1].sample(
                n=min_trials, random_state=42
            ),
            valid_trials[valid_trials.trial_outcome == 0].sample(
                n=min_trials, random_state=42
            ),
        ]
    )

    return balanced_trials, min_trials


def plot_scatter_panel(
    ax, x_data, y_data, xlabel, ylabel, x_err=None, y_err=None, highlight_idx=None
):
    """Plot a scatter comparison with standard formatting and optional error bars and highlights."""
    # Regular scatter plot
    ax.errorbar(
        x_data,
        y_data,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        color="black",
        alpha=0.2,
        ecolor="gray",
        elinewidth=1,
        capsize=2,
    )

    # Add highlights if specified
    if highlight_idx is not None:
        ax.plot(
            x_data[highlight_idx],
            y_data[highlight_idx],
            "o",
            mfc="none",
            mec="red",
            ms=15,
            mew=2,
        )

    min_val = min(np.min(x_data), np.min(y_data))
    max_val = max(np.max(x_data), np.max(y_data))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_aspect("equal")
    separate_axes(ax)


def plot_population_responses(trial_ts, spike_times_per_unit):
    """Creates a 2x4 grid with scatter plots and PSTHs."""
    # Make figure with custom grid
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4)  # Create 2x4 grid

    # First Stimulus Analysis (Top Left)
    balanced_trials, _ = get_balanced_trials(trial_ts)
    balanced_first_stim_ts = np.array(balanced_trials.first_stim_ts).flatten()
    pop_peth, timebin_edges, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_first_stim_ts,
        pre_seconds=0.025,
        post_seconds=0.15,
        binwidth_ms=10,
        pad=0,
        kernel=None,
    )

    # Movement vs Stationary Analysis (Top Right)
    balanced_trials, _ = get_balanced_trials(trial_ts, require_both_stim_types=True)
    balanced_movement_stims = np.array(
        [stims[0] for stims in balanced_trials.movement_stims]
    )
    balanced_stationary_stims = np.array(
        [stims[0] for stims in balanced_trials.stationary_stims]
    )

    pop_peth_move, timebin_edges_move, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_movement_stims,
        pre_seconds=0.025,
        post_seconds=0.15,
        binwidth_ms=10,
        pad=0,
        kernel=None,
    )
    pop_peth_stat, _, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_stationary_stims,
        pre_seconds=0.025,
        post_seconds=0.15,
        binwidth_ms=10,
        pad=0,
        kernel=None,
    )

    # Top Left: Pre vs Post Scatter
    ax1 = fig.add_subplot(gs[0, 0])  # Top row, first column
    pre_window = (timebin_edges[:-1] >= -0.1) & (timebin_edges[:-1] < 0)
    post_window = (timebin_edges[:-1] >= 0.04) & (timebin_edges[:-1] < 0.1)

    mean_peth = np.mean(pop_peth, axis=1)
    pre_activity = np.mean(mean_peth[:, pre_window], axis=1)
    post_activity = np.mean(mean_peth[:, post_window], axis=1)
    pre_sem = np.std(pop_peth[:, :, pre_window], axis=1) / np.sqrt(pop_peth.shape[1])
    post_sem = np.std(pop_peth[:, :, post_window], axis=1) / np.sqrt(pop_peth.shape[1])
    pre_sem = np.mean(pre_sem, axis=1)
    post_sem = np.mean(post_sem, axis=1)

    # Calculate response magnitude for next top responders (4-6 instead of 1-3)
    response_magnitude = post_activity - pre_activity
    top_6_idx = np.argsort(response_magnitude)[-6:][::-1]  # Get top 6
    next_3_idx = top_6_idx[3:6]  # Take indices 4-6

    # Plot scatter panels
    plot_scatter_panel(
        ax1,
        pre_activity,
        post_activity,
        "pre-stimulus activity (sp/s)",
        "post-stimulus activity (sp/s)",
        x_err=pre_sem,
        y_err=post_sem,
        highlight_idx=next_3_idx,
    )

    # Top Right: Movement vs Stationary Scatter
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom row, first column
    stimulus_window = (timebin_edges_move[:-1] >= 0.04) & (
        timebin_edges_move[:-1] <= 0.1
    )

    mean_peth_move = np.mean(pop_peth_move, axis=1)
    mean_peth_stat = np.mean(pop_peth_stat, axis=1)
    move_stimulus_response = np.mean(mean_peth_move[:, stimulus_window], axis=1)
    stat_stimulus_response = np.mean(mean_peth_stat[:, stimulus_window], axis=1)
    move_sem = np.std(pop_peth_move[:, :, stimulus_window], axis=1) / np.sqrt(
        pop_peth_move.shape[1]
    )
    stat_sem = np.std(pop_peth_stat[:, :, stimulus_window], axis=1) / np.sqrt(
        pop_peth_stat.shape[1]
    )
    move_sem = np.mean(move_sem, axis=1)
    stat_sem = np.mean(stat_sem, axis=1)

    plot_scatter_panel(
        ax2,
        stat_stimulus_response,
        move_stimulus_response,
        "stationary stimulus activity (sp/s)",
        "movement stimulus activity (sp/s)",
        x_err=stat_sem,
        y_err=move_sem,
        highlight_idx=next_3_idx,
    )

    # Get PETH with finer time bins for visualization
    pop_peth_fine, timebin_edges_fine, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_first_stim_ts,
        pre_seconds=0.2,
        post_seconds=0.2,
        binwidth_ms=5,
        pad=0,
        kernel=None,
    )

    # Get movement and stationary PETHs for the same units
    pop_peth_move_fine, timebin_edges_move_fine, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_movement_stims,
        pre_seconds=0.2,
        post_seconds=0.2,
        binwidth_ms=5,
        pad=0,
        kernel=None,
    )

    pop_peth_stat_fine, _, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_stationary_stims,
        pre_seconds=0.2,
        post_seconds=0.2,
        binwidth_ms=5,
        pad=0,
        kernel=None,
    )

    # Plot PSTHs for next 3 responders
    time_axis = timebin_edges_fine[:-1] + np.diff(timebin_edges_fine) / 2
    for i, idx in enumerate(next_3_idx):
        # First stimulus PSTH
        ax = fig.add_subplot(gs[0, i + 1])  # Top row, columns 1-3

        # Calculate mean and SEM for first stim
        unit_peth = pop_peth_fine[idx]
        mean_response = np.mean(unit_peth, axis=0)
        sem_response = np.std(unit_peth, axis=0) / np.sqrt(unit_peth.shape[0])

        # Plot PSTH with matching aesthetics
        ax.fill_between(
            time_axis,
            mean_response - sem_response,
            mean_response + sem_response,
            alpha=0.2,
            color="black",
        )
        ax.plot(time_axis, mean_response, "k-", linewidth=2)

        # Add stimulus onset line
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)

        # Set x-axis ticks and ensure y starts at 0
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_ylim(bottom=0)

        # Format
        ax.set_xlabel("Time from stimulus (s)", fontsize=18)
        if i == 0:
            ax.set_ylabel("Firing rate (sp/s)", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        separate_axes(ax)

        # Movement vs Stationary PSTH
        ax = fig.add_subplot(gs[1, i + 1])  # Bottom row, columns 1-3

        # Calculate mean and SEM for movement and stationary
        unit_peth_move = pop_peth_move_fine[idx]
        unit_peth_stat = pop_peth_stat_fine[idx]
        mean_move = np.mean(unit_peth_move, axis=0)
        mean_stat = np.mean(unit_peth_stat, axis=0)
        sem_move = np.std(unit_peth_move, axis=0) / np.sqrt(unit_peth_move.shape[0])
        sem_stat = np.std(unit_peth_stat, axis=0) / np.sqrt(unit_peth_stat.shape[0])

        # Plot both PSTHs with new colors
        ax.fill_between(
            time_axis,
            mean_move - sem_move,
            mean_move + sem_move,
            alpha=0.2,
            color="green",
        )
        ax.plot(time_axis, mean_move, "g-", linewidth=2, label="Movement")

        ax.fill_between(
            time_axis,
            mean_stat - sem_stat,
            mean_stat + sem_stat,
            alpha=0.2,
            color="magenta",
        )
        ax.plot(time_axis, mean_stat, "m-", linewidth=2, label="Stationary")

        # Add stimulus onset line and legend
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=12)

        # Set x-axis ticks and ensure y starts at 0
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_ylim(bottom=0)

        # Format
        ax.set_xlabel("Time from stimulus (s)", fontsize=18)
        if i == 0:
            ax.set_ylabel("Firing rate (sp/s)", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        separate_axes(ax)

    plt.tight_layout()
    return fig, next_3_idx


def main():
    # """Main function to load data and create plots."""
    # parser = argparse.ArgumentParser(description="Process ephys data for specified mouse and sessions.")
    # parser.add_argument('--data_path', type=str, required=True, help="Path to the data directory.")
    # parser.add_argument('--mouse', type=str, required=True, help="Specify a particular mouse ID (e.g., GRB123).")
    # parser.add_argument('--sessions', type=str, nargs='+', required=True, help="Specify one or more session IDs (e.g., 20230101_123456).")

    # args = parser.parse_args()

    # data_dir = args.data_path
    # animal = args.mouse
    # sessions = args.sessions
    data_dir = "/Users/gabriel/data"
    animal = "GRB006"
    sessions = ["20240723_142451"]

    for session in sessions:
        try:
            print(f"Processing session: {session}")
            trial_ts = pd.read_pickle(
                pjoin(data_dir, animal, session, "pre_processed", "trial_ts.pkl")
            )
            spike_times_per_unit = np.load(
                pjoin(
                    data_dir,
                    animal,
                    session,
                    "pre_processed",
                    "spike_times_per_unit.npy",
                ),
                allow_pickle=True,
            )

            # Create and save the figure
            fig, next_3_idx = plot_population_responses(trial_ts, spike_times_per_unit)
            print(next_3_idx)

            # Save the figure
            session_dir = Path(data_dir) / animal / session
            analysis_dir = session_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(analysis_dir / f"visual_scatters_{session}.svg", format="svg")
            fig.savefig(analysis_dir / f"visual_scatters_{session}.png", format="png")
            plt.close(fig)
            print(f"Successfully processed and saved figure for session: {session}")
        except Exception as e:
            print(f"Error processing session {session}: {e}")
            continue


if __name__ == "__main__":
    main()
