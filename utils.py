import numpy as np
import pandas as pd
import contextlib
import io
from spks.event_aligned import compute_firing_rate

@contextlib.contextmanager
def suppress_print():
    """ Suppress print statements when not needed """
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def detect_stim_events(time_vector, srate, analog_signal, amp_threshold = 5000, time_threshold = 0.04):
    """ Detect visual stimulus events from the nidaq analog signal """
    ii = np.where(np.diff(analog_signal>amp_threshold)==1)[0]
    return time_vector[ii[np.diff(np.hstack([0,ii]))>time_threshold*srate]]

def get_trial_ts(trial_starts : np.array, stim_ts : np.array, behavior_data : pd.DataFrame, port_events : dict) -> pd.DataFrame:
    """ Get a DataFrame with all the task events divided by trials """
    trial_ts = []

    for ti in range(len(trial_starts) - 1):
        start_time = trial_starts[ti]
        end_time = trial_starts[ti + 1]

        # get all stim ts in a given trial
        stim_events_in_interval = stim_ts[np.logical_and(stim_ts > start_time, stim_ts < end_time)]

        # get the first stim ts
        first_stim_ts = np.nan
        if len(stim_events_in_interval) > 0:
            first_stim_ts = stim_events_in_interval[0]
        
        trial_data = {
            "trial_rate": len(behavior_data.stimulus_event_timestamps[ti]),
            "detected_events": len(stim_events_in_interval),
            "trial_start": trial_starts[ti],
            "stim_ts": stim_events_in_interval,
            "first_stim_ts": first_stim_ts,
            "trial_outcome": behavior_data.outcome_record[ti]
        }

        for port_name, events in port_events.items():
            trial_data[f"{port_name}_entries"] = events["entries"][np.logical_and(events["entries"] > start_time, events["entries"] < end_time)]
            trial_data[f"{port_name}_exits"] = events["exits"][np.logical_and(events["exits"] > start_time, events["exits"] < end_time)]
        
        trial_ts.append(trial_data)
    
    return pd.DataFrame(trial_ts)

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

# def get_population_firing_rate(event_times, spike_times, tpre, tpost, binwidth_ms, kernel=None):
#     unit_fr = []
#     with suppress_print():
#         for i in range(len(spike_times)):
#             try:
#                 unit_fr.append(compute_firing_rate(event_times, spike_times[i], tpre, tpost, binwidth_ms, kernel=kernel)[0])
#             except:
#                 unit_fr.append(np.nan)
#     psth = np.mean(unit_fr, axis = 0)

#     return psth

def get_population_firing_rate(event_times, spike_times, tpre, tpost, binwidth_ms, kernel=None, window_ms=None):
    unit_fr = []
    with suppress_print():
        for i in range(len(spike_times)):
            try:
                unit_fr.append(compute_firing_rate(event_times, spike_times[i], tpre, tpost, binwidth_ms, kernel=kernel)[0])
            except:
                unit_fr.append(np.nan)
    
    psth = np.mean(unit_fr, axis=0)

    if window_ms:
        window_size_bins = int(window_ms / binwidth_ms)
        if psth.ndim == 1:
            psth = moving_average(psth, window_size_bins)
        elif psth.ndim == 2:
            psth = np.array([moving_average(row, window_size_bins) for row in psth])

        unit_fr = np.array([moving_average(np.mean(row, axis = 0), window_size_bins) for row in unit_fr])

    return psth, unit_fr

def compute_mean_sem(psth : np.array):
    return np.mean(psth, axis=0), np.std(psth, axis=0) / np.sqrt(psth.shape[0])

def get_nth_element(x, i):
    """ Used for getting the n-th event in an array 
    
    Example usage:
    for i, ax in enumerate(axs):
        for outcome, c in zip(np.unique(stim_ts_per_trial.trial_outcome), ['b', 'k', 'r', 'y']):
            ts = np.hstack(stim_ts_per_trial[stim_ts_per_trial.trial_outcome == outcome].stim_ts.apply(lambda x: get_nth_element(x, i)))
    """
    if len(x) > i and not np.isnan(x[0]):
        return x[i]
    return np.nan

def get_response_ts(row):
    """ Get the timestamp of when the animal responded, regardless of which side it was 
    Example usage:
    trial_ts.insert(trial_ts.shape[1], 'response', trial_ts.apply(get_response_ts, axis=1))
    """
    w = row['center_port_exits'][-1]  # withdrawal time
    left = [entry for entry in row['left_port_entries'] if entry > w]
    right = [entry for entry in row['right_port_entries'] if entry > w]
    
    # get the first valid value or None if empty
    left = left[0] if left else None
    right = right[0] if right else None
    
    # return the first valid (non-NaN) value
    if pd.notna(left):
        return left
    elif pd.notna(right):
        return right
    else:
        return None

def get_stationary_stims(row, max_tseconds = 0.4):
    return row.stim_ts[row.stim_ts < row.first_stim_ts + max_tseconds]

def get_movement_stims(row, max_tseconds = 0.4):
    return row.stim_ts[np.logical_and(row.center_port_exits[-1] < row.stim_ts, row.stim_ts < row.center_port_exits[-1] + max_tseconds)]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size