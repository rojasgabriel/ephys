import numpy as np
import contextlib
import io

def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    """ get the spike times for each cluster individually """
    return [spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu] for uclu in np.unique(spike_clusters[good_unit_ids])]

def get_good_units(clusters_obj, spike_clusters):
    """ filter Kilosort results using criteria specified in Melin et al. 2024.

    Parameters
    ----------
    clusters_obj : object from spks.clusters.Clusters
        object to access spike sorting results
    spike_clusters : ndarray
        spike clusters output from Kilosort (i.e. "../spike_clusters.npy")

    Returns
    ----------
    good_unit_ids : boolean ndarray
        boolean array where True means that a recorded spike corresponded to a filtered spike cluster
    n_units : int
        number of single units filtered
    """
    mask = ((np.abs(clusters_obj.cluster_info.trough_amplitude - clusters_obj.cluster_info.peak_amplitude) > 50)
            & (clusters_obj.cluster_info.amplitude_cutoff < 0.1) 
            & (clusters_obj.cluster_info.isi_contamination < 0.1)
            & (clusters_obj.cluster_info.presence_ratio >= 0.6)
            & (clusters_obj.cluster_info.spike_duration > 0.1))

    good_unit_ids = np.isin(spike_clusters,clusters_obj.cluster_info[mask].cluster_id.values)
    n_units = len(clusters_obj.cluster_info[mask])

    return good_unit_ids, n_units

def compute_mean_sem(psth):
    return np.mean(psth, axis=0), np.std(psth, axis=0) / np.sqrt(psth.shape[0])

def get_nth_element(x, i):
    if isinstance(x, np.ndarray) and len(x) > i and not np.isnan(x[0]):
        return x[i]
    return np.nan

@contextlib.contextmanager
def suppress_print():
    """ suppress print statements """
    with contextlib.redirect_stdout(io.StringIO()):
        yield