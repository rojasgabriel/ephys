#!/Users/gabriel/miniconda3/bin/python
# %% Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
from spks.event_aligned import population_peth  # type: ignore
from spks.utils import alpha_function  # type: ignore
from ephys.utils.utils_analysis import (
    calculate_stim_offsets,
    find_unique_cross_trial_offset_pairs,
    compute_stim_response_for_trial_subset,
)  # type: ignore
from ephys.viz import plot_scatter_panel  # type: ignore

plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

# %% Load data
animal = "GRB006"  # example animal
session = "20240723_142451"  # example session

data_dir = "/Users/gabriel/data"
trial_ts = pd.read_pickle(
    pjoin(data_dir, animal, session, "pre_processed", "trial_ts.pkl")
)
spike_times_per_unit = np.load(
    pjoin(data_dir, animal, session, "pre_processed", "spike_times_per_unit.npy"),
    allow_pickle=True,
)

trial_ts = trial_ts[
    trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["center_port_entries"].apply(lambda x: len(x) > 0)
].copy()

# %% Main analysis

# DEFINE VARIABLES TO BE USED EVERYWHERE
pre_seconds = 0.025
post_seconds = 0.15
binwidth_ms = 25
stim_window_start = 0.04
stim_window_end = 0.1
wiggle = 0.010


stims_offset_df = calculate_stim_offsets(
    trial_ts, trial_start_col="center_port_entries"
)
matched_unique_pairs_df = find_unique_cross_trial_offset_pairs(
    stims_offset_df, wiggle_room=wiggle
)
if matched_unique_pairs_df.empty:
    raise ValueError("No matched stationary/movement stimulus pairs found.")

offset_range_ms = np.round(
    matched_unique_pairs_df.offset_diff.max()
    - matched_unique_pairs_df.offset_diff.min(),
    2,
)

# Extract alignment times for the matched pairs
stat_alignment_times = matched_unique_pairs_df["stat_stim_time"].values
move_alignment_times = matched_unique_pairs_df["move_stim_time"].values

# Define alpha kernel to convolve PETHs with alpha function kernel
t_decay = 0.025
t_rise = 0.001
decay = t_decay / (binwidth_ms / 1000)
kern = alpha_function(
    int(decay * 15), t_rise=t_rise, t_decay=decay, srate=1.0 / (binwidth_ms / 1000)
)

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
stimulus_window_bool = (timebin_edges_stat[:-1] >= stim_window_start) & (
    timebin_edges_stat[:-1] <= stim_window_end
)

# Calculate mean response per neuron across the matched pairs
# Shape of pop_peth is (n_neurons, n_trials/pairs, n_bins)
stat_response_per_neuron = np.max(
    np.mean(pop_peth_stat_matched[:, :, stimulus_window_bool], axis=1), axis=1
)  # np.mean(pop_peth_stat_matched[:, :, stimulus_window_bool], axis=(1, 2))
move_response_per_neuron = np.max(
    np.mean(pop_peth_move_matched[:, :, stimulus_window_bool], axis=1), axis=1
)  # np.mean(pop_peth_move_matched[:, :, stimulus_window_bool], axis=(1, 2))

# Calculate SEM per neuron across the matched pairs
n_pairs = pop_peth_stat_matched.shape[1]
stat_sem_per_neuron = np.std(
    np.max(pop_peth_stat_matched[:, :, stimulus_window_bool], axis=2), axis=1
) / np.sqrt(n_pairs)  # np.std(stat_response_per_neuron, axis=1) / np.sqrt(n_pairs)
move_sem_per_neuron = np.std(
    np.max(pop_peth_move_matched[:, :, stimulus_window_bool], axis=2), axis=1
) / np.sqrt(n_pairs)  # np.std(move_response_per_neuron, axis=1) / np.sqrt(n_pairs)

# --- Plotting ---
fig1, ax1 = plt.subplots(1, figsize=(5, 5))

plot_scatter_panel(
    ax1,
    stat_response_per_neuron,
    move_response_per_neuron,
    "stationary stimulus activity (sp/s)",
    "running stimulus activity (sp/s)",
    x_err=stat_sem_per_neuron,
    y_err=move_sem_per_neuron,
    plot_unity=True,
)

ax1.set_title(
    f"n = {n_pairs} pairs of stimuli\noffset_range_ms is {offset_range_ms} ms",
    fontsize=10,
)

fig1.tight_layout()
save_dir = "/Users/gabriel/figures/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig1.savefig(
    pjoin(save_dir, f"stim_responses_{offset_range_ms}s_offset.svg"),
    format="svg",
    dpi=300,
    bbox_inches="tight",
)

# %% Now to compare slow vs. fast response times
rt = []
idx = []
for itrial, data in trial_ts.iterrows():
    rt.append(data.response - data.center_port_exits[0])
    idx.append(itrial)

response_times = np.array(rt)
trial_indices = np.array(idx)

response_times_df = pd.DataFrame(
    {"trial_idx": trial_indices, "response_times": np.array(response_times)}
)

response_times_df = (
    response_times_df.dropna()
)  # remove early withdrawals and no choice trials if any remained lol
bottom = 0.25
top = 0.75
quantile_25 = response_times_df.response_times.quantile([bottom]).values[0]
quantile_75 = response_times_df.response_times.quantile([top]).values[0]

response_times_df["quantile_25"] = response_times_df.response_times <= quantile_25
response_times_df["quantile_75"] = response_times_df.response_times >= quantile_75

trials_idx_25 = response_times_df[response_times_df.quantile_25].trial_idx.values
trials_idx_75 = response_times_df[response_times_df.quantile_75].trial_idx.values

slow_rt_trial_ts = trial_ts[trial_ts.index.isin(trials_idx_25)].copy()
fast_rt_trial_ts = trial_ts[trial_ts.index.isin(trials_idx_75)].copy()

slow_responses, n_slow_pairs = compute_stim_response_for_trial_subset(
    spike_times_per_unit=spike_times_per_unit,
    trial_subset=slow_rt_trial_ts,
    pre_seconds=pre_seconds,
    post_seconds=post_seconds,
    binwidth_ms=binwidth_ms,
    stim_window_start=stim_window_start,
    stim_window_end=stim_window_end,
    wiggle_room=wiggle,
)

fast_responses, n_fast_pairs = compute_stim_response_for_trial_subset(
    spike_times_per_unit=spike_times_per_unit,
    trial_subset=fast_rt_trial_ts,
    pre_seconds=pre_seconds,
    post_seconds=post_seconds,
    binwidth_ms=binwidth_ms,
    stim_window_start=stim_window_start,
    stim_window_end=stim_window_end,
    wiggle_room=wiggle,
)


fig2, ax2 = plt.subplots(1, figsize=(5, 5))
plot_scatter_panel(
    ax2,
    slow_responses["stationary"],
    slow_responses["running"],
    "stationary stimulus activity (sp/s)",
    "running stimulus activity (sp/s)",
    x_err=slow_responses["stationary_sem"],
    y_err=slow_responses["running_sem"],
    c="blue",
    plot_unity=False,
)
plot_scatter_panel(
    ax2,
    fast_responses["stationary"],
    fast_responses["running"],
    "stationary stimulus activity (sp/s)",
    "running stimulus activity (sp/s)",
    x_err=fast_responses["stationary_sem"],
    y_err=fast_responses["running_sem"],
    c="red",
    plot_unity=True,
)

# filter out NaNs or invalid entries before fitting
mask_slow = np.isfinite(slow_responses["stationary"]) & np.isfinite(
    slow_responses["running"]
)
mask_fast = np.isfinite(fast_responses["stationary"]) & np.isfinite(
    fast_responses["running"]
)
x_slow = slow_responses["stationary"][mask_slow]
y_slow = slow_responses["running"][mask_slow]
x_fast = fast_responses["stationary"][mask_fast]
y_fast = fast_responses["running"][mask_fast]

# simple linear fits
slow_coef = np.polyfit(
    x_slow, y_slow, 1
)  # the first value is the 1st degree coef and the second is the constant
fast_coef = np.polyfit(x_fast, y_fast, 1)

# plot the fit lines
min_val = min(x_slow.min(), x_fast.min())
max_val = max(x_slow.max(), x_fast.max())
x_fit = np.linspace(min_val, max_val, 100)
ax2.plot(
    x_fit, slow_coef[0] * x_fit + slow_coef[1], "b-", label="slow fit"
)  # here we are doing `y = mx + b`
ax2.plot(x_fit, fast_coef[0] * x_fit + fast_coef[1], "r-", label="fast fit")

ax2.legend(frameon=False)

fig2.tight_layout()
save_dir = "/Users/gabriel/figures/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig2.savefig(
    pjoin(save_dir, f"stim_responses_fast_vs_slow_{offset_range_ms}s_offset.svg"),
    format="svg",
    dpi=300,
    bbox_inches="tight",
)

# %% Avg firing rate quartiles
mod_idx = (move_response_per_neuron - stat_response_per_neuron) / (
    move_response_per_neuron + stat_response_per_neuron
)

# estimate each unit’s overall firing rate (spikes/sec)
# here we take the full span of recorded spike times
starts = [st.min() if len(st) > 0 else 0 for st in spike_times_per_unit]
stops = [st.max() if len(st) > 0 else 0 for st in spike_times_per_unit]
t_start = min(starts)
t_stop = max(stops)
dur = t_stop - t_start
fr_units = np.array([len(st) / dur for st in spike_times_per_unit])

# assign quartiles
quartiles = pd.qcut(fr_units, 3, labels=["low", "medium", "high"])

# build a DataFrame
df_q = pd.DataFrame(
    {"firing_rate": fr_units, "mod_idx": mod_idx, "quartile": quartiles}
)

# plot modulation index by firing‐rate quartile
fig3, ax3 = plt.subplots(1, figsize=(5, 4))
sns.boxplot(
    x="quartile",
    y="mod_idx",
    data=df_q,
    hue="quartile",
    palette="hls",
    ax=ax3,
    legend=False,
    fill=False,
    showfliers=False,
)
sns.stripplot(
    x="quartile",
    y="mod_idx",
    data=df_q,
    hue="quartile",
    palette="hls",
    size=4,
    jitter=True,
    ax=ax3,
)
ax3.set_xlabel("avg. session firing rate")
ax3.set_ylabel("movement index\n(move-stat)/(move+stat)")
fig3.tight_layout()

# %% Modulation index
slow_idx = (slow_responses["running"] - slow_responses["stationary"]) / (
    slow_responses["running"] + slow_responses["stationary"]
)
fast_idx = (fast_responses["running"] - fast_responses["stationary"]) / (
    fast_responses["running"] + fast_responses["stationary"]
)

# TODO: this df and df_q are redundant. i should construct just one of them with all the data at the beginning before doing all the plotting
df_comp = pd.DataFrame(
    {"quartile": quartiles, "slow_idx": slow_idx, "fast_idx": fast_idx}
)

n = len(df_comp["slow_idx"])
x_slow = np.zeros(n)
x_fast = np.ones(n)

# add a little horizontal jitter
jitter = 0.1
x_slow_j = x_slow + np.random.uniform(-jitter, jitter, size=n)
x_fast_j = x_fast + np.random.uniform(-jitter, jitter, size=n)

df_comp.insert(loc=0, column="x_slow_j", value=x_slow_j, allow_duplicates=False)
df_comp.insert(loc=0, column="x_fast_j", value=x_fast_j, allow_duplicates=False)

fig4, ax4 = plt.subplots(1, figsize=(4, 6))

quartile_categories = df_comp["quartile"].cat.categories.tolist()
palette_hls = sns.color_palette("hls", len(quartile_categories))[::-1]
quartile_color_map = {
    category: color for category, color in zip(quartile_categories, palette_hls)
}

# Draw lines connecting each unit’s slow->fast, colored by quartile
for i in range(n):  # n is the number of units
    quartile_value = df_comp["quartile"].iloc[i]
    line_color = quartile_color_map[quartile_value]
    ax4.plot(
        [x_slow_j[i], x_fast_j[i]],
        [df_comp["slow_idx"].iloc[i], df_comp["fast_idx"].iloc[i]],
        color=line_color,
        alpha=0.5,
        linewidth=0.7,
        zorder=1,  # Draw lines behind scatter points
    )

sns.scatterplot(
    data=df_comp,
    x=x_slow_j,
    y="slow_idx",
    hue="quartile",
    hue_order=quartile_categories,
    palette=palette_hls,
    ax=ax4,
    legend=False,
    s=50,
    alpha=0.8,
    zorder=2,
)
sns.scatterplot(
    data=df_comp,
    x=x_fast_j,
    y="fast_idx",
    hue="quartile",
    hue_order=quartile_categories,
    palette=palette_hls,
    ax=ax4,
    legend=True,
    s=50,
    alpha=0.8,
    zorder=2,  # Generate legend from this plot
)

ax4.set_xticks([0, 1])
ax4.set_xticklabels(["slow", "fast"])
ax4.set_ylabel("movement index\n(move-stat)/(move+stat)")
ax4.set_xlabel("time to choice")

handles, labels = ax4.get_legend_handles_labels()
ax4.legend(bbox_to_anchor=(1.05, 1), loc="best", frameon=False)
fig4.tight_layout()


slope = df_comp["fast_idx"] - df_comp["slow_idx"]
df_comp.insert(0, "slope", slope)

fig5, ax5 = plt.subplots(1, figsize=(6, 4))
sns.histplot(
    data=df_comp,
    x="slope",
    hue="quartile",
    hue_order=quartile_categories,
    palette=palette_hls[::-1],
    ax=ax5,
    bins=20,
    kde=True,
    alpha=0.5,
    linewidth=0.5,
    line_kws={"linewidth": 2},
    legend=False,
)

ax5.set_xlabel("slope\n(fast-slow)")
ax5.legend(labels=quartile_categories, loc="best", frameon=False)
fig5.tight_layout()


plt.show()
