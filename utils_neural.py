import numpy as np


def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    """
    Get the spike times for each cluster individually.

    Args:
        spike_times (np.array): one dimensional array of spike times for all units
        spike_clusters (np.array): one dimensional array of cluster ids for all spike times. must be of same length as spike_times
        good_unit_ids (bool): good unit ids mask obtained from get_good_units function

    Returns:
        np.array: spike times for each cluster of n_units x n_spikes
    """
    return [
        spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu]
        for uclu in np.unique(spike_clusters[good_unit_ids])
    ]


def get_good_units(clusters_obj, spike_clusters):
    mask = (
        (
            np.abs(
                clusters_obj.cluster_info.trough_amplitude
                - clusters_obj.cluster_info.peak_amplitude
            )
            > 50
        )
        & (clusters_obj.cluster_info.amplitude_cutoff < 0.1)
        & (clusters_obj.cluster_info.isi_contamination < 0.1)
        & (clusters_obj.cluster_info.presence_ratio >= 0.6)
        & (clusters_obj.cluster_info.spike_duration > 0.1)
        & (clusters_obj.cluster_info.firing_rate > 2)
    )  # added this filter myself. changed from 1 to 2 sp/s on 4/16/25
    # TODO:add n_active_channels < 100 filter to eliminate multi channel noise

    good_unit_ids = np.isin(
        spike_clusters, clusters_obj.cluster_info[mask].cluster_id.values
    )
    n_units = len(clusters_obj.cluster_info[mask])

    return good_unit_ids, n_units
