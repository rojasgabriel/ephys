#%% Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
from os.path import join as pjoin


import matplotlib.pyplot as plt
import matplotlib as mpl
from spks.event_aligned import population_peth
from spks.utils import alpha_function

plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams['backend'] = 'macosx'


#%% Load data
animal = 'GRB006'  # example animal
session = '20240723_142451'  # example session

data_dir = '/Users/gabriel/data'
trial_ts = pd.read_pickle(pjoin(data_dir, animal, session, "pre_processed", "trial_ts.pkl"))
spike_times_per_unit = np.load(pjoin(data_dir, animal, session, "pre_processed", "spike_times_per_unit.npy"), allow_pickle=True)

#%% Pre-process variables
trial_ts = trial_ts[
    trial_ts['stationary_stims'].apply(lambda x: len(x) > 0) &
    trial_ts['movement_stims'].apply(lambda x: len(x) > 0) &
    trial_ts['center_port_entries'].apply(lambda x: len(x) > 0)
].copy()

stim_lists = trial_ts.stationary_stims.tolist()
n_stims = 4

tmp = [
    np.array([stims[i] for stims in stim_lists if len(stims) > i])
    for i in range(n_stims)
]

first_stim, second_stim, third_stim, fourth_stim = tmp

first_movement_stim = np.array([stims[0] for stims in trial_ts.movement_stims.tolist()])

#%% Define constants
binwidth_ms = 25
pre_seconds = binwidth_ms/1000
post_seconds = 0.15 + pre_seconds

t_decay = 0.025
t_rise = 0.001
decay = t_decay / (binwidth_ms/1000)
kern = alpha_function(int(decay * 15), t_rise=t_rise, t_decay=decay, srate=1./(binwidth_ms/1000))

#%% Compute peak responses 0.4–1.0 s after each flash, for each unit
stimuli = [first_stim, second_stim, third_stim, fourth_stim, first_movement_stim]
peak_responses = []
stim_window_start = 0.04
stim_window_end = 0.1

for stim_times in stimuli:
    peth, tb, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=stim_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        kernel=kern,
        pad=0
    )
    # select timebins between 0.4 and 1.0 s
    stimulus_window_bool = (tb[:-1] >= stim_window_start) & (tb[:-1] <= stim_window_end)
    peak_response = np.max(np.mean(peth[:, :, stimulus_window_bool], axis=1), axis=1)
    peak_responses.append(peak_response)

# stack into (n_units, 5) and wrap in a DataFrame
peak_matrix = np.stack(peak_responses, axis=1)
colnames = ['peak_first_flash',
            'peak_second_flash',
            'peak_third_flash',
            'peak_fourth_flash',
            'peak_movement_flash']
peak_df = pd.DataFrame(peak_matrix, columns=colnames)

# Plot each unit’s peak responses across the five stimuli
peak_subtracted_df = peak_df.subtract(peak_df['peak_first_flash'], axis=0)
stimuli = peak_subtracted_df.columns
x = np.arange(len(stimuli))

fig1, ax1 = plt.subplots(1, figsize=(5, 5))

# Plot individual units as dots
for vals in peak_subtracted_df.values:
    ax1.plot(x, vals, 'o', color='lightgray', alpha=0.3, markersize=3)

# Plot mean ± SEM
means = peak_subtracted_df.mean()
sems = peak_subtracted_df.sem()
ax1.errorbar(x, means, yerr=sems, fmt='o-', color='red', linewidth=2, 
           markersize=8, capsize=5, label=f'mean ± SEM\n{len(peak_subtracted_df)} units')

ax1.set_xticks(x)
ax1.set_xticklabels(stimuli, rotation=45, ha='right')
ax1.set_ylabel('peak response relative to 1st flash (sp/s)')
ax1.legend()
ax1.set_ylim([-10, 10])
fig1.tight_layout()

melt_peak = peak_subtracted_df.melt(var_name='stimulus', value_name='response')
fig2, ax2 = plt.subplots(1, figsize=(5, 5))
sns.boxplot(data=melt_peak, x='stimulus', y='response', color='red', showfliers=False, fill=False, ax=ax2)

# before setting labels, fix the tick positions
ax2.set_xticks(range(len(stimuli)))
ax2.set_xticklabels(stimuli, rotation=45, ha='right')

ax2.set_ylabel('peak response relative to 1st flash (sp/s)')
ax2.set_ylim([-25, 25])
fig2.tight_layout()

#%% Split trials by response side
left_trials = trial_ts[trial_ts['response_side'] == 0].copy()
right_trials = trial_ts[trial_ts['response_side'] == 1].copy()

print(f"Left response trials: {len(left_trials)}")
print(f"Right response trials: {len(right_trials)}")

# Extract stimulus times for each response side
def extract_stims_by_side(trials_df, n_stims=4):
    stim_lists = trials_df.stationary_stims.tolist()
    stims = [
        np.array([stims[i] for stims in stim_lists if len(stims) > i])
        for i in range(n_stims)
    ]
    movement_stim = np.array([stims[0] for stims in trials_df.movement_stims.tolist()])
    return stims + [movement_stim]

left_stims = extract_stims_by_side(left_trials)
right_stims = extract_stims_by_side(right_trials)

# Calculate peak responses for left and right trials separately
def calculate_peak_responses(stim_list):
    peak_responses = []
    for stim_times in stim_list:
        peth, tb, _ = population_peth(
            all_spike_times=spike_times_per_unit,
            alignment_times=stim_times,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            binwidth_ms=binwidth_ms,
            kernel=kern,
            pad=0
        )
        # select timebins between 0.04 and 0.1 s
        stimulus_window_bool = (tb[:-1] >= stim_window_start) & (tb[:-1] <= stim_window_end)
        peak_response = np.max(np.mean(peth[:, :, stimulus_window_bool], axis=1), axis=1)
        peak_responses.append(peak_response)
    return peak_responses

# Get peak responses for each side
left_peak_responses = calculate_peak_responses(left_stims)
right_peak_responses = calculate_peak_responses(right_stims)

# Create DataFrames
colnames = ['peak_first_flash', 'peak_second_flash', 'peak_third_flash', 
           'peak_fourth_flash', 'peak_movement_flash']

left_peak_matrix = np.stack(left_peak_responses, axis=1)
right_peak_matrix = np.stack(right_peak_responses, axis=1)

left_peak_df = pd.DataFrame(left_peak_matrix, columns=colnames)
right_peak_df = pd.DataFrame(right_peak_matrix, columns=colnames)

# Subtract first flash response for each side
left_peak_subtracted_df = left_peak_df.subtract(left_peak_df['peak_first_flash'], axis=0)
right_peak_subtracted_df = right_peak_df.subtract(right_peak_df['peak_first_flash'], axis=0)

# Plot comparison
stimuli = left_peak_subtracted_df.columns
x = np.arange(len(stimuli))

fig3, ax3 = plt.subplots(1, figsize=(5, 5))

# Plot individual units as dots for both sides
for vals in left_peak_subtracted_df.values:
    ax3.plot(x, vals, 'o', color='blue', alpha=0.1, markersize=2)
for vals in right_peak_subtracted_df.values:
    ax3.plot(x, vals, 'o', color='orange', alpha=0.1, markersize=2)

# Plot means ± SEM for both sides
left_means = left_peak_subtracted_df.mean()
left_sems = left_peak_subtracted_df.sem()
right_means = right_peak_subtracted_df.mean()
right_sems = right_peak_subtracted_df.sem()

ax3.errorbar(x - 0.1, left_means, yerr=left_sems, fmt='o-', color='blue', linewidth=2, 
           markersize=8, capsize=5, label=f'Left trials (n={len(left_trials)})')
ax3.errorbar(x + 0.1, right_means, yerr=right_sems, fmt='o-', color='orange', linewidth=2, 
           markersize=8, capsize=5, label=f'Right trials (n={len(right_trials)})')

ax3.set_xticks(x)
ax3.set_xticklabels(stimuli, rotation=45, ha='right')
ax3.set_ylabel('Peak response relative to 1st flash (sp/s)')
ax3.set_ylim([-10, 10])
ax3.legend()
fig3.tight_layout()

# Melt the DataFrames for plotting
left_melted = left_peak_subtracted_df.melt(var_name='stimulus', value_name='response')
left_melted['side'] = 'Left'

right_melted = right_peak_subtracted_df.melt(var_name='stimulus', value_name='response')
right_melted['side'] = 'Right'

combined_df = pd.concat([left_melted, right_melted], ignore_index=True)

fig4, ax4 = plt.subplots(1, figsize=(5, 5))

sns.boxplot(data=combined_df, x='stimulus', y='response', hue='side', palette=['blue', 'orange'], showfliers=False, fill=False, ax=ax4)

# similarly for ax4
ax4.set_xticks(range(len(stimuli)))
ax4.set_xticklabels(stimuli, rotation=45, ha='right')

ax4.set_ylabel('Peak response relative to 1st flash (sp/s)')
ax4.set_ylim([-25, 25])
ax4.legend(title='Response Side')
fig4.tight_layout()

figures_dir = '/Users/gabriel/Desktop/figures'
os.makedirs(figures_dir, exist_ok=True)

fig1.savefig(os.path.join(figures_dir, 'adaptation_mean.svg'), format='svg', dpi=300)
fig2.savefig(os.path.join(figures_dir, 'adaptation_boxplot.svg'), format='svg', dpi=300)
fig3.savefig(os.path.join(figures_dir, 'adaptation_by_side_mean.svg'), format='svg', dpi=300)
fig4.savefig(os.path.join(figures_dir, 'adaptation_by_side_boxplot.svg'), format='svg', dpi=300)

# plt.show()