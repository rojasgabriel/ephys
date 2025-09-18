# Useful functions for my ephys analyses
# Gabriel Rojas Bowe - Jun 2024

import numpy as np
import matplotlib.pyplot as plt


def plot_psth(
    mean_sem_func,
    pre_seconds,
    post_seconds,
    binwidth_ms,
    window_ms=None,
    xlabel=None,
    ylabel=None,
    fig_title=None,
    data_label=None,
    color="b",
    ax=None,
    tight=True,
    vline=True,
):
    mean, sem = mean_sem_func
    x = np.arange(-pre_seconds, post_seconds, binwidth_ms / 1000)

    if window_ms is not None:
        window_size_bins = int(window_ms / binwidth_ms)
        if window_size_bins >= len(x):
            raise ValueError(
                "Smoothing window is too large compared to the data length."
            )
        if window_size_bins > 1:
            x = x[(window_size_bins - 1) // 2 : -(window_size_bins // 2)]

    if len(x) != len(mean):
        raise ValueError(
            f"x and mean must have the same length, but got {len(x)} and {len(mean)}."
        )

    if not ax:
        ax = plt.gca()

    ax.plot(x, mean, color=color, alpha=0.5, label=data_label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3)

    if vline:
        ax.vlines(
            0,
            ymin=mean.min(),
            ymax=mean.max(),
            color="k",
            linestyles="dashed",
            alpha=0.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(fig_title, fontsize=10)

    if tight:
        plt.tight_layout()

    if data_label:
        ax.legend()


def plot_cluster_info_histograms(clu):
    """Plots quality control metrics for Kilosort results. Works with the spks.clusters.Clusters object"""
    # Setup
    fig, axs = plt.subplots(3, 2, tight_layout=True, figsize=(10, 6))
    n_bins = 50

    # Calculate the total number of neurons
    total_neurons = len(clu.cluster_info)

    # Calculate the excluded samples
    excluded_peak_amplitude = np.sum(
        np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude)
        <= 50
    )
    excluded_isi_contamination = np.sum(clu.cluster_info.isi_contamination > 0.1)
    excluded_spike_duration = np.sum(clu.cluster_info.spike_duration <= 0.1)
    excluded_presence_ratio = np.sum(clu.cluster_info.presence_ratio <= 0.6)
    excluded_amplitude_cutoff = np.sum(clu.cluster_info.amplitude_cutoff > 0.1)

    # Set the figure title
    fig.suptitle(
        f"Total clusters: {total_neurons}\n Shaded area indicates excluded clusters by metric",
        fontsize=14,
    )

    # ------- Plot histograms ------- #

    # spike amplitude
    axs[0, 0].hist(
        np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude),
        bins=n_bins,
        alpha=0.5,
        color="b",
    )
    axs[0, 0].axvline(x=50, color="b", linestyle="--")
    axs[0, 0].fill_betweenx(
        y=axs[0, 0].get_ylim(),
        x1=max(axs[0, 0].get_xlim()[0], 0),
        x2=50,
        color="b",
        alpha=0.1,
    )
    axs[0, 0].set_xlabel("spike amplitude")
    axs[0, 0].set_ylabel("counts")
    axs[0, 0].set_title(f"n = {excluded_peak_amplitude}")

    # ISI contamination
    axs[1, 0].hist(
        clu.cluster_info.isi_contamination, bins=n_bins, alpha=0.5, color="g"
    )
    axs[1, 0].axvline(x=0.1, color="g", linestyle="--")
    axs[1, 0].fill_betweenx(
        y=axs[1, 0].get_ylim(),
        x1=0.1,
        x2=min(axs[1, 0].get_xlim()[1], axs[1, 0].get_xlim()[1]),
        color="g",
        alpha=0.1,
    )
    axs[1, 0].set_xlabel("isi_contamination")
    axs[1, 0].set_ylabel("counts")
    axs[1, 0].set_title(f"n = {excluded_isi_contamination}")

    # spike duration
    axs[2, 0].hist(clu.cluster_info.spike_duration, bins=n_bins, alpha=0.5, color="r")
    axs[2, 0].axvline(x=0.1, color="r", linestyle="--")
    axs[2, 0].fill_betweenx(
        y=axs[2, 0].get_ylim(),
        x1=max(axs[2, 0].get_xlim()[0], 0),
        x2=0.1,
        color="r",
        alpha=0.1,
    )
    axs[2, 0].set_xlabel("spike_duration")
    axs[2, 0].set_ylabel("counts")
    axs[2, 0].set_title(f"n = {excluded_spike_duration}")

    # presence ratio
    axs[0, 1].hist(
        clu.cluster_info.presence_ratio, bins=n_bins, alpha=0.5, color="orange"
    )
    axs[0, 1].axvline(x=0.6, color="orange", linestyle="--")
    axs[0, 1].fill_betweenx(
        y=axs[0, 1].get_ylim(),
        x1=max(axs[0, 1].get_xlim()[0], 0),
        x2=0.6,
        color="orange",
        alpha=0.1,
    )
    axs[0, 1].set_xlabel("presence_ratio")
    axs[0, 1].set_ylabel("counts")
    axs[0, 1].set_title(f"n = {excluded_presence_ratio}")

    # amplitude cutoff
    axs[1, 1].hist(
        clu.cluster_info.amplitude_cutoff, bins=n_bins, alpha=0.5, color="purple"
    )
    axs[1, 1].axvline(x=0.1, color="purple", linestyle="--")
    axs[1, 1].fill_betweenx(
        y=axs[1, 1].get_ylim(),
        x1=0.1,
        x2=min(axs[1, 1].get_xlim()[1], axs[1, 1].get_xlim()[1]),
        color="purple",
        alpha=0.1,
    )
    axs[1, 1].set_xlabel("amplitude_cutoff")
    axs[1, 1].set_ylabel("counts")
    axs[1, 1].set_title(f"n = {excluded_amplitude_cutoff}")

    # depth - not used for filtering out clusters
    axs[2, 1].hist(clu.cluster_info.depth, bins=n_bins, alpha=0.5, color="c")
    axs[2, 1].set_xlabel("depth")
    axs[2, 1].set_ylabel("counts")


def plot_scatter_panel(
    ax,
    x_data,
    y_data,
    xlabel,
    ylabel,
    x_err=None,
    y_err=None,
    c="black",
    highlight_idx=None,
    plot_unity=False,
):
    """Plot a scatter comparison with standard formatting and optional error bars and highlights."""
    # Regular scatter plot
    ax.errorbar(
        x_data,
        y_data,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        color=c,
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

    if plot_unity:
        min_val = min(np.min(x_data), np.min(y_data))
        max_val = max(np.max(x_data), np.max(y_data))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.set_aspect("equal")
