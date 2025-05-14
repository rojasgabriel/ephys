#!/Users/gabriel/miniconda3/bin/python
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
from ephys.utils_analysis import (calculate_stim_offsets, 
                                  find_unique_cross_trial_offset_pairs, 
                                  compute_stim_response_for_trial_subset)
from ephys.viz import plot_scatter_panel

# new_rc_params = {'text.usetex': False,
# "svg.fonttype": 'none'
# }
# mpl.rcParams.update(new_rc_params)
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100

#%% Load data
animal = 'GRB006'  # example animal
session = '20240723_142451'  # example session

data_dir = '/Users/gabriel/data'
trial_ts = pd.read_pickle(pjoin(data_dir, animal, session, "pre_processed", "trial_ts.pkl"))
spike_times_per_unit = np.load(pjoin(data_dir, animal, session, "pre_processed", "spike_times_per_unit.npy"), allow_pickle=True)

trial_ts = trial_ts[
    trial_ts['stationary_stims'].apply(lambda x: len(x) > 0) &
    trial_ts['movement_stims'].apply(lambda x: len(x) > 0) &
    trial_ts['center_port_entries'].apply(lambda x: len(x) > 0)
].copy()

#%% Main analysis

#DEFINE VARIABLES TO BE USED EVERYWHERE
pre_seconds = 0.025
post_seconds = 0.15
binwidth_ms = 25
stim_window_start = 0.04
stim_window_end = 0.06
wiggle = 0.01 #10ms

data = trial_ts.copy()

stims_offset_df = calculate_stim_offsets(data, trial_start_col='center_port_entries')
matched_unique_pairs_df = find_unique_cross_trial_offset_pairs(stims_offset_df, wiggle_room=wiggle)
if matched_unique_pairs_df.empty:
    raise ValueError("No matched stationary/movement stimulus pairs found.")

offset_range_ms = np.round(matched_unique_pairs_df.offset_diff.max() - matched_unique_pairs_df.offset_diff.min(), 2)

# Extract alignment times for the matched pairs
stat_alignment_times = matched_unique_pairs_df['stat_stim_time'].values
move_alignment_times = matched_unique_pairs_df['move_stim_time'].values

# Define alpha kernel to convolve PETHs with alpha function kernel
t_decay = 0.025
t_rise = 0.001
decay = t_decay / (binwidth_ms/1000)
kern = alpha_function(int(decay * 15), t_rise=t_rise, t_decay=decay, srate=1./(binwidth_ms/1000))

# Calculate PETH for stationary stimuli in the pairs
pop_peth_stat_matched, timebin_edges_stat, _ = population_peth(
    all_spike_times=spike_times_per_unit,
    alignment_times=stat_alignment_times,
    pre_seconds=pre_seconds,
    post_seconds=post_seconds,
    binwidth_ms=binwidth_ms,
    pad=0,
    kernel=kern,
)

# Calculate PETH for movement stimuli in the pairs
pop_peth_move_matched, timebin_edges_move, _ = population_peth(
    all_spike_times=spike_times_per_unit,
    alignment_times=move_alignment_times,
    pre_seconds=pre_seconds,
    post_seconds=post_seconds,
    binwidth_ms=binwidth_ms,
    pad=0,
    kernel=kern,
)

# Define the stimulus response window
stimulus_window_bool = (timebin_edges_stat[:-1] >= stim_window_start) & (timebin_edges_stat[:-1] <= stim_window_end)

# Calculate mean response per neuron across the matched pairs
# Shape of pop_peth is (n_neurons, n_trials/pairs, n_bins)
stat_response_per_neuron = np.mean(pop_peth_stat_matched[:, :, stimulus_window_bool], axis=(1, 2))
move_response_per_neuron = np.mean(pop_peth_move_matched[:, :, stimulus_window_bool], axis=(1, 2))

# Calculate SEM per neuron across the matched pairs
n_pairs = pop_peth_stat_matched.shape[1]
stat_sem_per_neuron = np.std(np.mean(pop_peth_stat_matched[:, :, stimulus_window_bool], axis=2), axis=1) / np.sqrt(n_pairs)
move_sem_per_neuron = np.std(np.mean(pop_peth_move_matched[:, :, stimulus_window_bool], axis=2), axis=1) / np.sqrt(n_pairs)

# --- Plotting ---
fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))

plot_scatter_panel(ax1, 
                    stat_response_per_neuron, 
                    move_response_per_neuron,
                    "stationary stimulus activity (sp/s)", 
                    "movement stimulus activity (sp/s)",
                    x_err=stat_sem_per_neuron, 
                    y_err=move_sem_per_neuron)

ax1.set_title(f'n = {n_pairs} pairs of stimuli\noffset_range_ms is {offset_range_ms} ms', fontsize=10)

fig1.tight_layout()
save_dir = '/Users/gabriel/figures/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig1.savefig(pjoin(save_dir, f"matched_stim_responses_{offset_range_ms}s_offset.svg"), format='svg', dpi=300, bbox_inches='tight')

#%% Now to compare slow vs. fast response times
rt = []
idx = []
for itrial, data in trial_ts.iterrows():
    rt.append(data.response - data.center_port_exits[0])
    idx.append(itrial)

response_times = np.array(rt)
trial_indices = np.array(idx)

response_times_df = pd.DataFrame({
    'trial_idx': trial_indices,
    'response_times': np.array(response_times)
})

response_times_df = response_times_df.dropna() #remove early withdrawals and no choice trials if any remained lol
bottom = 0.25
top = 0.75
quantile_25 = response_times_df.response_times.quantile([bottom]).values[0]
quantile_75 = response_times_df.response_times.quantile([top]).values[0]

response_times_df['quantile_25'] = response_times_df.response_times <= quantile_25
response_times_df['quantile_75'] = response_times_df.response_times >= quantile_75

trials_idx_25 = response_times_df[response_times_df.quantile_25 == True].trial_idx.values
trials_idx_75 = response_times_df[response_times_df.quantile_75 == True].trial_idx.values

slow_rt_trial_ts = trial_ts[trial_ts.index.isin(trials_idx_25)].copy()
fast_rt_trial_ts = trial_ts[trial_ts.index.isin(trials_idx_75)].copy()

slow_responses, n_slow_pairs = compute_stim_response_for_trial_subset(spike_times_per_unit=spike_times_per_unit,
                                                                      trial_subset=slow_rt_trial_ts, 
                                                                      pre_seconds=pre_seconds,
                                                                      post_seconds=post_seconds,
                                                                      binwidth_ms=binwidth_ms,
                                                                      stim_window_start=stim_window_start,
                                                                      stim_window_end=stim_window_end,
                                                                      wiggle_room=wiggle)

fast_responses, n_fast_pairs = compute_stim_response_for_trial_subset(spike_times_per_unit=spike_times_per_unit,
                                                                      trial_subset=fast_rt_trial_ts, 
                                                                      pre_seconds=pre_seconds,
                                                                      post_seconds=post_seconds,
                                                                      binwidth_ms=binwidth_ms,
                                                                      stim_window_start=stim_window_start,
                                                                      stim_window_end=stim_window_end,
                                                                      wiggle_room=wiggle)

n_neurons = slow_responses["Stationary"].shape[0]
rows = []
for neuron in np.arange(n_neurons):
    rows.append({
        "neuron_id": neuron,
        "activity": slow_responses["Stationary"][neuron],
        "condition": "stationary slow"
    })
    rows.append({
        "neuron_id": neuron,
        "activity": slow_responses["Movement"][neuron],
        "condition": "running slow"
    })
    rows.append({
        "neuron_id": neuron,
        "activity": fast_responses["Stationary"][neuron],
        "condition": "stationary fast"
    })
    rows.append({
        "neuron_id": neuron,
        "activity": fast_responses["Movement"][neuron],
        "condition": "running fast"
    })
    
results_df = pd.DataFrame(rows)

condition_order = [
    "stationary slow", 
    "running slow",
    "stationary fast", 
    "running fast"
]

fig2, ax2 = plt.subplots(1, 1, figsize=(4, 6))
sns.boxenplot(data=results_df, 
              x='condition', 
              y='activity', 
              order=condition_order, 
              hue='condition', 
              palette="Paired", 
              saturation=0.75, 
              showfliers=False,
              width_method="exponential", 
              k_depth="trustworthy", 
              trust_alpha=0.01,
              ax = ax2)
ax2.set_ylabel(f'sp/s\n({int(results_df.shape[0]/4)} units)')
ax2.tick_params(axis='x', labelrotation=45)
# ax2.figtext(0.15, 0.90, f"slow pairs: {n_slow_pairs}, fast pairs: {n_fast_pairs}, offset: {wiggle*1000:.0f}ms", fontsize=10)
fig2.tight_layout()

plt.show()