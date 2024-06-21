# Useful functions for my ephys analyses
# Gabriel Rojas Bowe - Jun 2024

import numpy as np
import matplotlib.pyplot as plt

def plot_cluster_info_histograms(clu):
    # Setup
    fig, axs = plt.subplots(3, 2, tight_layout=True, figsize=(10, 6))
    n_bins = 50

    # Calculate the total number of neurons
    total_neurons = len(clu.cluster_info)

    # Calculate the excluded samples
    excluded_peak_amplitude = np.sum(np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude) <= 50)
    excluded_isi_contamination = np.sum(clu.cluster_info.isi_contamination > 0.1)
    excluded_spike_duration = np.sum(clu.cluster_info.spike_duration <= 0.1)
    excluded_presence_ratio = np.sum(clu.cluster_info.presence_ratio <= 0.6)
    excluded_amplitude_cutoff = np.sum(clu.cluster_info.amplitude_cutoff > 0.1)

    # Set the figure title
    fig.suptitle(f'Total clusters: {total_neurons}\n Shaded area indicates excluded clusters by metric', fontsize=14)

    # ------- Plot histograms ------- #

    # spike amplitude
    axs[0, 0].hist(np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude), bins=n_bins, alpha=0.5, color='b')
    axs[0, 0].axvline(x=50, color='b', linestyle='--')
    axs[0, 0].fill_betweenx(y=axs[0, 0].get_ylim(), x1=max(axs[0, 0].get_xlim()[0], 0), x2=50, color='b', alpha=0.1)
    axs[0, 0].set_xlabel('spike amplitude')
    axs[0, 0].set_ylabel('counts')
    axs[0, 0].set_title(f'n = {excluded_peak_amplitude}')

    # ISI contamination
    axs[1, 0].hist(clu.cluster_info.isi_contamination, bins=n_bins, alpha=0.5, color='g')
    axs[1, 0].axvline(x=0.1, color='g', linestyle='--')
    axs[1, 0].fill_betweenx(y=axs[1, 0].get_ylim(), x1=0.1, x2=min(axs[1, 0].get_xlim()[1], axs[1, 0].get_xlim()[1]), color='g', alpha=0.1)
    axs[1, 0].set_xlabel('isi_contamination')
    axs[1, 0].set_ylabel('counts')
    axs[1, 0].set_title(f'n = {excluded_isi_contamination}')

    # spike duration
    axs[2, 0].hist(clu.cluster_info.spike_duration, bins=n_bins, alpha=0.5, color='r')
    axs[2, 0].axvline(x=0.1, color='r', linestyle='--')
    axs[2, 0].fill_betweenx(y=axs[2, 0].get_ylim(), x1=max(axs[2, 0].get_xlim()[0], 0), x2=0.1, color='r', alpha=0.1)
    axs[2, 0].set_xlabel('spike_duration')
    axs[2, 0].set_ylabel('counts')
    axs[2, 0].set_title(f'n = {excluded_spike_duration}')

    # presence ratio
    axs[0, 1].hist(clu.cluster_info.presence_ratio, bins=n_bins, alpha=0.5, color='orange')
    axs[0, 1].axvline(x=0.6, color='orange', linestyle='--')
    axs[0, 1].fill_betweenx(y=axs[0, 1].get_ylim(), x1=max(axs[0, 1].get_xlim()[0], 0), x2=0.6, color='orange', alpha=0.1)
    axs[0, 1].set_xlabel('presence_ratio')
    axs[0, 1].set_ylabel('counts')
    axs[0, 1].set_title(f'n = {excluded_presence_ratio}')

    # amplitude cutoff
    axs[1, 1].hist(clu.cluster_info.amplitude_cutoff, bins=n_bins, alpha=0.5, color='purple')
    axs[1, 1].axvline(x=0.1, color='purple', linestyle='--')
    axs[1, 1].fill_betweenx(y=axs[1, 1].get_ylim(), x1=0.1, x2=min(axs[1, 1].get_xlim()[1], axs[1, 1].get_xlim()[1]), color='purple', alpha=0.1)
    axs[1, 1].set_xlabel('amplitude_cutoff')
    axs[1, 1].set_ylabel('counts')
    axs[1, 1].set_title(f'n = {excluded_amplitude_cutoff}')

    # depth - not used for filtering out clusters, but a helpful viz nonetheless
    axs[2, 1].hist(clu.cluster_info.depth, bins=n_bins, alpha=0.5, color='c')
    axs[2, 1].set_xlabel('depth')
    axs[2, 1].set_ylabel('counts')

def individual_raster_viewer(trig_ts, tpre=0.1, tpost=1):
    from ipywidgets import IntSlider, Button, HBox, VBox
    from IPython.display import display
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Define the plotting function
    def plot_neuron(iunit):
        ax.clear()
        for i, ss in enumerate(trig_ts[iunit]):
            ax.vlines(ss, i, i + 1, color='k')
        ax.set_xlim([-tpre, tpost])
        ax.set_ylabel('Trial #')
        ax.set_xlabel('Time from first event onset (s)')
        fig.canvas.draw_idle()  # Update the plot without blocking

    # Create the slider and buttons
    slider = IntSlider(min=0, max=len(trig_ts) - 1, step=1, value=0)
    next_button = Button(description="Next")
    prev_button = Button(description="Previous")

    # Define button click event handlers
    def on_next_button_clicked(b):
        slider.value = min(slider.value + 1, slider.max)

    def on_prev_button_clicked(b):
        slider.value = max(slider.value - 1, slider.min)

    # Attach event handlers to buttons
    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)

    # Update plot when slider value changes
    def on_slider_value_change(change):
        plot_neuron(change['new'])

    slider.observe(on_slider_value_change, names='value')

    # Display buttons and slider
    display(VBox([HBox([prev_button, next_button]), slider]))

    # Initial plot
    plot_neuron(slider.value)

def separate_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    yti = ax.get_yticks()
    yti = yti[(yti >= ax.get_ylim()[0]) & (yti <= ax.get_ylim()[1]+10**-3)] #Add a small value to cover for some very tiny added values
    ax.spines['left'].set_bounds([yti[0], yti[-1]])
    xti = ax.get_xticks()
    xti = xti[(xti >= ax.get_xlim()[0]) & (xti <= ax.get_xlim()[1]+10**-3)]
    ax.spines['bottom'].set_bounds([xti[0], xti[-1]])
    return