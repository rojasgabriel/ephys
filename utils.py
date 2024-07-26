import numpy as np
import contextlib
import io

def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    # this separates the spike times for each cluster
    # used for computing spike rates for individual units
    return [spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu] for uclu in np.unique(spike_clusters[good_unit_ids])]

def compute_mean_sem(psth):
    return np.mean(psth, axis=0), np.std(psth, axis=0) / np.sqrt(psth.shape[0])

@contextlib.contextmanager
def suppress_print():
    """Context manager to suppress print statements."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield