#!/usr/bin/env python3
"""
Population response analysis for ephys data.
Creates a 2x2 grid comparing first stimulus responses and movement vs stationary responses.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spks.event_aligned import population_peth
from chiCa.chiCa.visualization_utils import separate_axes
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
plt.rcParams['font.sans-serif'] = ['Arial'] 

def get_balanced_trials(trial_ts, require_both_stim_types=True):
    """Get equal numbers of rewarded and unrewarded trials."""
    # Get valid trials (exclude outcome -1)
    valid_trials = trial_ts[trial_ts.trial_outcome.isin([0,1])]
    
    # Optionally require both stim types
    if require_both_stim_types:
        valid_trials = valid_trials[
            (valid_trials.movement_stims.apply(len) > 0) & 
            (valid_trials.stationary_stims.apply(len) > 0)
        ]
    
    # Find minimum number of trials between conditions
    min_trials = min(
        len(valid_trials[valid_trials.trial_outcome == 1]),
        len(valid_trials[valid_trials.trial_outcome == 0])
    )
    
    # Sample equal numbers from each condition
    balanced_trials = pd.concat([
        valid_trials[valid_trials.trial_outcome == 1].sample(n=min_trials, random_state=42),
        valid_trials[valid_trials.trial_outcome == 0].sample(n=min_trials, random_state=42)
    ])
    
    return balanced_trials, min_trials

def plot_peth_panel(ax, timebin_edges, pop_peth, color="black", label=None):
    """Plot a PETH with standard formatting."""
    mean_peth = np.mean(pop_peth, axis=1)
    pop_mean = np.mean(mean_peth, axis=0)
    pop_sem = np.std(mean_peth, axis=0) / np.sqrt(mean_peth.shape[0])
    
    ax.plot(timebin_edges[:-1], pop_mean, color=color, linewidth=2, label=label)
    ax.fill_between(
        timebin_edges[:-1],
        pop_mean - pop_sem,
        pop_mean + pop_sem,
        color=color,
        alpha=0.2,
    )
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("time from stimulus (s)", fontsize=18)
    ax.set_ylabel("sp/s", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    separate_axes(ax)
    
    return mean_peth

def plot_scatter_panel(ax, x_data, y_data, xlabel, ylabel, color="black"):
    """Plot a scatter comparison with standard formatting."""
    ax.scatter(x_data, y_data, alpha=0.5, color=color)
    min_val = min(np.min(x_data), np.min(y_data))
    max_val = max(np.max(x_data), np.max(y_data))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_aspect("equal")
    separate_axes(ax)

def plot_population_responses(trial_ts, spike_times_per_unit):
    """
    Creates a 2x2 grid of population response analyses:
    - Left column: First stimulus onset analysis
    - Right column: Movement vs Stationary comparison
    Top row shows PETHs, bottom row shows scatter comparisons
    """
    fig = plt.figure(figsize=(16, 16))
    
    # First Stimulus Analysis (Left Column)
    balanced_trials, min_trials = get_balanced_trials(trial_ts)
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
    
    # Movement vs Stationary Analysis (Right Column)
    balanced_trials, _ = get_balanced_trials(trial_ts, require_both_stim_types=True)
    balanced_movement_stims = np.array([stims[0] for stims in balanced_trials.movement_stims])
    balanced_stationary_stims = np.array([stims[0] for stims in balanced_trials.stationary_stims])
    
    pop_peth_move, timebin_edges_move, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_movement_stims,
        pre_seconds=0.025,
        post_seconds=0.15,
        binwidth_ms=10,
        pad=0,
        kernel=None,
    )
    pop_peth_stat, timebin_edges_stat, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=balanced_stationary_stims,
        pre_seconds=0.025,
        post_seconds=0.15,
        binwidth_ms=10,
        pad=0,
        kernel=None,
    )
    
    # Top Left: First Stim PETH
    ax1 = fig.add_subplot(221)
    mean_peth = plot_peth_panel(ax1, timebin_edges, pop_peth, color="black")
    ax1.legend(handles=[
        plt.Line2D([], [], color='black', label=f"n={len(spike_times_per_unit)} neurons\n{min_trials} trials per outcome", linestyle='None')
    ],
    labelcolor='linecolor',
    handlelength=0,
    handletextpad=0,
    fontsize=14,
    loc='upper right',
    frameon=False)
    
    # Top Right: Movement vs Stationary PETH
    ax2 = fig.add_subplot(222)
    mean_peth_move = plot_peth_panel(ax2, timebin_edges_move, pop_peth_move, 
                                    color="green", label=f"movement stims (n={len(balanced_movement_stims)})")
    mean_peth_stat = plot_peth_panel(ax2, timebin_edges_stat, pop_peth_stat, 
                                    color="magenta", label=f"stationary stims (n={len(balanced_stationary_stims)})")
    ax2.legend(handles=[
        plt.Line2D([], [], color='green', label=f"movement stims (n={len(balanced_movement_stims)})", linestyle='None'),
        plt.Line2D([], [], color='magenta', label=f"stationary stims (n={len(balanced_stationary_stims)})", linestyle='None')
    ],
    labelcolor='linecolor',
    handlelength=0,
    handletextpad=0,
    fontsize=14,
    loc='upper right',
    frameon=False)
    
    # Bottom Left: Pre vs Post Scatter
    ax3 = fig.add_subplot(223)
    pre_window = (timebin_edges[:-1] >= -0.1) & (timebin_edges[:-1] < 0)
    post_window = (timebin_edges[:-1] >= 0) & (timebin_edges[:-1] < 0.1)
    pre_activity = np.mean(mean_peth[:, pre_window], axis=1)
    post_activity = np.mean(mean_peth[:, post_window], axis=1)
    
    # Calculate percentages and colors, excluding points on diagonal
    colors = []
    for x,y in zip(post_activity, pre_activity):
        if x == y:
            colors.append('none')
        elif x > y:
            colors.append('red')
        else:
            colors.append('blue')
            
    pct_excited = (np.array(colors) == 'red').mean() * 100
    pct_inhibited = (np.array(colors) == 'blue').mean() * 100
    
    plot_scatter_panel(ax3, post_activity, pre_activity, 
                      "post-stimulus activity (sp/s)", "pre-stimulus activity (sp/s)",
                      color=colors)
    
    ax3.legend(handles=[
        plt.Line2D([], [], color='red', label=f"excited ({pct_excited:.1f}%)", linestyle='None'),
        plt.Line2D([], [], color='blue', label=f"inhibited ({pct_inhibited:.1f}%)", linestyle='None')
    ],
    labelcolor='linecolor',
    handlelength=0,
    handletextpad=0,
    fontsize=14,
    loc='upper right',
    frameon=False)  # removes box outline
    
    # Bottom Right: Movement vs Stationary Scatter
    ax4 = fig.add_subplot(224)
    stimulus_window = (timebin_edges_move[:-1] >= 0) & (timebin_edges_move[:-1] <= 0.1)
    move_stimulus_response = np.mean(mean_peth_move[:, stimulus_window], axis=1)
    stat_stimulus_response = np.mean(mean_peth_stat[:, stimulus_window], axis=1)
    
    # Calculate percentages and colors, excluding points on diagonal
    colors = []
    for x,y in zip(move_stimulus_response, stat_stimulus_response):
        if x == y:
            colors.append('none')
        elif x > y:
            colors.append('green') 
        else:
            colors.append('magenta')
            
    pct_green = (np.array(colors) == 'green').mean() * 100
    pct_magenta = (np.array(colors) == 'magenta').mean() * 100
    
    plot_scatter_panel(ax4, move_stimulus_response, stat_stimulus_response,
                      "movement stimulus activity (sp/s)", "stationary stimulus activity (sp/s)",
                      color=colors)
    
    ax4.legend(handles=[
        plt.Line2D([], [], color='green', label=f"movement-preferring ({pct_green:.1f}%)", linestyle='None'),
        plt.Line2D([], [], color='magenta', label=f"stationary-preferring ({pct_magenta:.1f}%)", linestyle='None')
    ],
    labelcolor='linecolor',
    handlelength=0,
    handletextpad=0,
    fontsize=14,
    loc='upper right',
    frameon=False)  # removes box outline
    plt.tight_layout()
    return fig

def main():
    """Main function to load data and create plots."""
    # Load the data
    save_dir = Path('processed_data')
    animal = 'GRB006'  # example animal
    session = '20240723_142451'  # example session
    
    data_dir = save_dir / animal / session
    trial_ts = pd.read_pickle(data_dir / 'trial_ts.pkl')
    spike_times_per_unit = np.load(data_dir / 'spike_times_per_unit.npy', allow_pickle=True)
    
    # Create and save the figure
    fig = plot_population_responses(trial_ts, spike_times_per_unit)
    
    # Save the figure
    fig_dir = Path('figures') / animal / session
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / 'fig3_neural_activity.svg', format='svg')
    plt.close(fig)

if __name__ == '__main__':
    main()


