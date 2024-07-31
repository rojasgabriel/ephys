# Useful functions for my ephys analyses
# Gabriel Rojas Bowe - Jun 2024

import numpy as np
import matplotlib.pyplot as plt
from spks.event_aligned import compute_firing_rate
from utils import get_cluster_spike_times, compute_mean_sem, suppress_print

def plot_psth(mean_sem_func, pre_seconds, post_seconds, binwidth_ms, xlabel, ylabel, fig_title=None, data_label=None, color='b', ax = None, tight = True):
    mean, sem = mean_sem_func
    x = np.arange(-pre_seconds, post_seconds, binwidth_ms/1000)

    if not ax:
        ax = plt.gca()

    ax.plot(x, mean, color=color, alpha=0.5, label=data_label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3)
    ax.vlines(0, ymin=mean.min(), ymax=mean.max(), color='k', linestyles='dashed', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(fig_title, fontsize=10)
    if tight:
        plt.tight_layout()

def individual_psth_viewer(event_times, single_unit_timestamps, pre_seconds, post_seconds, binwidth_ms, save_dir, fig_title = None):
    from ipywidgets import IntSlider, Button, HBox, VBox
    from IPython.display import display
    import matplotlib.pyplot as plt
    import os

    # Create the slider and buttons
    ax = plt.gca()
    slider = IntSlider(min=0, max=len(single_unit_timestamps) - 1, step=1, value=0)
    next_button = Button(description="Next")
    prev_button = Button(description="Previous")
    save_button = Button(description="Save")

    # Define button click event handlers
    def on_next_button_clicked(b):
        slider.value = min(slider.value + 1, slider.max)

    def on_prev_button_clicked(b):
        slider.value = max(slider.value - 1, slider.min)

    def on_save_button_clicked(b):
        filename = f"{fig_title}_unit_{slider.value}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)

    # Attach event handlers to buttons
    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)
    save_button.on_click(on_save_button_clicked)

    # Update plot when slider value changes
    def on_slider_value_change(change):
        ax.clear()
        # plot_psth(change['new'], event_times = event_times, spike_times = spike_times, pre = pre, post = post, binw = binw, kernel_width = kernel_width)
        with suppress_print():
            psth, _ = compute_firing_rate(event_times, single_unit_timestamps[slider.value], pre_seconds, post_seconds, binwidth_ms)
        plot_psth(compute_mean_sem(psth), pre_seconds, post_seconds, binwidth_ms, 'time from event (s)', 'spike rate (Hz)', f'{fig_title}\niunit: {slider.value}')
        plt.draw()  # Ensure the plot is updated

    slider.observe(on_slider_value_change, names='value')

    # Display buttons and slider
    display(VBox([HBox([prev_button, next_button, save_button]), slider]))

    # Initial plot
    with suppress_print():
        psth, _ = compute_firing_rate(event_times, single_unit_timestamps[slider.value], pre_seconds, post_seconds, binwidth_ms)
    plot_psth(compute_mean_sem(psth), pre_seconds, post_seconds, binwidth_ms, 'time from event (s)', 'spike rate (Hz)', f'{fig_title}\niunit: {slider.value}')
    plt.show()


# #%% old functions

# #%% Visualizing Kilosort cluster data

# def plot_cluster_info_histograms(clu):
#     # Setup
#     fig, axs = plt.subplots(3, 2, tight_layout=True, figsize=(10, 6))
#     n_bins = 50

#     # Calculate the total number of neurons
#     total_neurons = len(clu.cluster_info)

#     # Calculate the excluded samples
#     excluded_peak_amplitude = np.sum(np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude) <= 50)
#     excluded_isi_contamination = np.sum(clu.cluster_info.isi_contamination > 0.1)
#     excluded_spike_duration = np.sum(clu.cluster_info.spike_duration <= 0.1)
#     excluded_presence_ratio = np.sum(clu.cluster_info.presence_ratio <= 0.6)
#     excluded_amplitude_cutoff = np.sum(clu.cluster_info.amplitude_cutoff > 0.1)

#     # Set the figure title
#     fig.suptitle(f'Total clusters: {total_neurons}\n Shaded area indicates excluded clusters by metric', fontsize=14)

#     # ------- Plot histograms ------- #

#     # spike amplitude
#     axs[0, 0].hist(np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude), bins=n_bins, alpha=0.5, color='b')
#     axs[0, 0].axvline(x=50, color='b', linestyle='--')
#     axs[0, 0].fill_betweenx(y=axs[0, 0].get_ylim(), x1=max(axs[0, 0].get_xlim()[0], 0), x2=50, color='b', alpha=0.1)
#     axs[0, 0].set_xlabel('spike amplitude')
#     axs[0, 0].set_ylabel('counts')
#     axs[0, 0].set_title(f'n = {excluded_peak_amplitude}')

#     # ISI contamination
#     axs[1, 0].hist(clu.cluster_info.isi_contamination, bins=n_bins, alpha=0.5, color='g')
#     axs[1, 0].axvline(x=0.1, color='g', linestyle='--')
#     axs[1, 0].fill_betweenx(y=axs[1, 0].get_ylim(), x1=0.1, x2=min(axs[1, 0].get_xlim()[1], axs[1, 0].get_xlim()[1]), color='g', alpha=0.1)
#     axs[1, 0].set_xlabel('isi_contamination')
#     axs[1, 0].set_ylabel('counts')
#     axs[1, 0].set_title(f'n = {excluded_isi_contamination}')

#     # spike duration
#     axs[2, 0].hist(clu.cluster_info.spike_duration, bins=n_bins, alpha=0.5, color='r')
#     axs[2, 0].axvline(x=0.1, color='r', linestyle='--')
#     axs[2, 0].fill_betweenx(y=axs[2, 0].get_ylim(), x1=max(axs[2, 0].get_xlim()[0], 0), x2=0.1, color='r', alpha=0.1)
#     axs[2, 0].set_xlabel('spike_duration')
#     axs[2, 0].set_ylabel('counts')
#     axs[2, 0].set_title(f'n = {excluded_spike_duration}')

#     # presence ratio
#     axs[0, 1].hist(clu.cluster_info.presence_ratio, bins=n_bins, alpha=0.5, color='orange')
#     axs[0, 1].axvline(x=0.6, color='orange', linestyle='--')
#     axs[0, 1].fill_betweenx(y=axs[0, 1].get_ylim(), x1=max(axs[0, 1].get_xlim()[0], 0), x2=0.6, color='orange', alpha=0.1)
#     axs[0, 1].set_xlabel('presence_ratio')
#     axs[0, 1].set_ylabel('counts')
#     axs[0, 1].set_title(f'n = {excluded_presence_ratio}')

#     # amplitude cutoff
#     axs[1, 1].hist(clu.cluster_info.amplitude_cutoff, bins=n_bins, alpha=0.5, color='purple')
#     axs[1, 1].axvline(x=0.1, color='purple', linestyle='--')
#     axs[1, 1].fill_betweenx(y=axs[1, 1].get_ylim(), x1=0.1, x2=min(axs[1, 1].get_xlim()[1], axs[1, 1].get_xlim()[1]), color='purple', alpha=0.1)
#     axs[1, 1].set_xlabel('amplitude_cutoff')
#     axs[1, 1].set_ylabel('counts')
#     axs[1, 1].set_title(f'n = {excluded_amplitude_cutoff}')

#     # depth - not used for filtering out clusters, but a helpful viz nonetheless
#     axs[2, 1].hist(clu.cluster_info.depth, bins=n_bins, alpha=0.5, color='c')
#     axs[2, 1].set_xlabel('depth')
#     axs[2, 1].set_ylabel('counts')

# #%% Individual neuron related functions

# def individual_raster_viewer(trig_ts, tpre=0.1, tpost=1):
#     from ipywidgets import IntSlider, Button, HBox, VBox
#     from IPython.display import display
#     # Create the figure and axis
#     fig, ax = plt.subplots(1, 1, figsize=(8, 5))

#     # Define the plotting function
#     def plot_neuron(iunit):
#         ax.clear()
#         for i, ss in enumerate(trig_ts[iunit]):
#             ax.vlines(ss, i, i + 1, color='k')
#         ax.set_xlim([-tpre, tpost])
#         ax.set_ylabel('Trial #')
#         ax.set_xlabel('Time from first event onset (s)')
#         fig.canvas.draw_idle()  # Update the plot without blocking

#     # Create the slider and buttons
#     slider = IntSlider(min=0, max=len(trig_ts) - 1, step=1, value=0)
#     next_button = Button(description="Next")
#     prev_button = Button(description="Previous")

#     # Define button click event handlers
#     def on_next_button_clicked(b):
#         slider.value = min(slider.value + 1, slider.max)

#     def on_prev_button_clicked(b):
#         slider.value = max(slider.value - 1, slider.min)

#     # Attach event handlers to buttons
#     next_button.on_click(on_next_button_clicked)
#     prev_button.on_click(on_prev_button_clicked)

#     # Update plot when slider value changes
#     def on_slider_value_change(change):
#         plot_neuron(change['new'])

#     slider.observe(on_slider_value_change, names='value')

#     # Display buttons and slider
#     display(VBox([HBox([prev_button, next_button]), slider]))

#     # Initial plot
#     plot_neuron(slider.value)

# #%% PSTH related functions

# def plot_psth(iunit, event_times, spike_times, pre=0.5, post=1, binw=0.01, use_kernel=False, kernel_width=2):
#     """
#     Plots the Peri-Stimulus Time Histogram (PSTH) with SEM shading.
    
#     Parameters:
#     - event_times: array-like, times of the stimulus events.
#     - spike_times: array-like, times of the spikes.
#     - pre: float, time window before the stimulus event.
#     - post: float, time window after the stimulus event.
#     - binw: float, bin width for the histogram.
#     - kernel_width: float, width of the Gaussian kernel for smoothing.
#     """
#     if use_kernel:
#         psth_matrix, event_index = compute_firing_rate(
#             event_times=event_times,
#             spike_times=spike_times[iunit],
#             pre_seconds=pre,
#             post_seconds=post,
#             binwidth_ms=int(binw * 1000),
#             kernel=gaussian_function(kernel_width)
#         )

#     psth_matrix, event_index = compute_firing_rate(
#     event_times=event_times,
#     spike_times=spike_times[iunit],
#     pre_seconds=pre,
#     post_seconds=post,
#     binwidth_ms=int(binw * 1000),
#     kernel=None
# )
    
#     trial_avg_psth = np.mean(psth_matrix, axis=0)
#     trial_sem_psth = np.std(psth_matrix, axis=0) / np.sqrt(psth_matrix.shape[0])

#     x_nums = np.arange(-pre, post, binw)
#     plt.plot(x_nums[:-1], trial_avg_psth, label='Average PSTH')
#     plt.fill_between(x_nums[:-1], trial_avg_psth - trial_sem_psth, trial_avg_psth + trial_sem_psth, alpha=0.3, label='SEM')
#     plt.vlines([0], ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed', color='gray')
#     plt.xlabel('Time from first stimulus event (s)')
#     plt.ylabel('Firing rate (Hz)')
#     plt.title(f'Trial avg PSTH with SEM; iunit: {iunit}')

#     ax = plt.gca()
#     separate_axes(ax)  # Assuming separate_axes is defined elsewhere

# def individual_psth_viewer(event_times, spike_times, pre=0.5, post=1, binw=0.01, use_kernel=False, kernel_width=2):
#     from ipywidgets import IntSlider, Button, HBox, VBox
#     from IPython.display import display
#     # Create the slider and buttons
#     ax = plt.gca()
#     slider = IntSlider(min=0, max=len(spike_times) - 1, step=1, value=0)
#     next_button = Button(description="Next")
#     prev_button = Button(description="Previous")

#     # Define button click event handlers
#     def on_next_button_clicked(b):
#         slider.value = min(slider.value + 1, slider.max)

#     def on_prev_button_clicked(b):
#         slider.value = max(slider.value - 1, slider.min)

#     # Attach event handlers to buttons
#     next_button.on_click(on_next_button_clicked)
#     prev_button.on_click(on_prev_button_clicked)

#     # Update plot when slider value changes
#     def on_slider_value_change(change):
#         ax.clear()
#         plot_psth(change['new'], event_times = event_times, spike_times = spike_times, pre = pre, post = post, binw = binw, kernel_width = kernel_width)

#     slider.observe(on_slider_value_change, names='value')

#     # Display buttons and slider
#     display(VBox([HBox([prev_button, next_button]), slider]))

#     # Initial plot
#     plot_psth(iunit = slider.value, event_times = event_times, spike_times = spike_times, pre = pre, post = post, binw = binw, use_kernel=use_kernel, kernel_width = kernel_width)

# # ---------- Plotting utils ----------
# def separate_axes(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     yti = ax.get_yticks()
#     yti = yti[(yti >= ax.get_ylim()[0]) & (yti <= ax.get_ylim()[1]+10**-3)] #Add a small value to cover for some very tiny added values
#     ax.spines['left'].set_bounds([yti[0], yti[-1]])
#     xti = ax.get_xticks()
#     xti = xti[(xti >= ax.get_xlim()[0]) & (xti <= ax.get_xlim()[1]+10**-3)]
#     ax.spines['bottom'].set_bounds([xti[0], xti[-1]])
#     return