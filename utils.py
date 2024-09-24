import numpy as np
import pandas as pd
import contextlib
import io
from spks.event_aligned import compute_firing_rate

@contextlib.contextmanager
def suppress_print():
    """ 
    Suppress print statements when not needed.
    
    This context manager redirects stdout to a StringIO object,
    effectively suppressing any print statements within its scope.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def detect_stim_events(time_vector, srate, analog_signal, amp_threshold=5000, time_threshold=0.04):
    """ 
    Detect visual stimulus events from the nidaq analog signal.
    
    Parameters:
    -----------
    time_vector : array-like
        The time points corresponding to the analog signal.
    srate : float
        Sampling rate of the analog signal.
    analog_signal : array-like
        The analog signal containing stimulus events.
    amp_threshold : float, optional
        Amplitude threshold for detecting events (default is 5000).
    time_threshold : float, optional
        Time threshold in seconds for considering consecutive events (default is 0.04).
    
    Returns:
    --------
    array-like
        Timestamps of detected stimulus events.
    """
    ii = np.where(np.diff(analog_signal>amp_threshold)==1)[0]
    return time_vector[ii[np.diff(np.hstack([0,ii]))>time_threshold*srate]]

def get_trial_ts(trial_starts: np.array, stim_ts: np.array, behavior_data: pd.DataFrame, port_events: dict) -> pd.DataFrame:
    """ 
    Get a DataFrame with all the task events divided by trials.
    
    Parameters:
    -----------
    trial_starts : np.array
        Timestamps of the trial starts.
    stim_ts : np.array
        Timestamps of the stimulus events.
    behavior_data : pd.DataFrame
        DataFrame with the behavior data.
    port_events : dict
        Dictionary with the port events.

    Returns:
    --------
    pd.DataFrame
        DataFrame with the trial events.
    """
    trial_data = []
    
    # Vectorized operations for stim events
    stim_events = np.searchsorted(stim_ts, trial_starts)
    
    for ti in range(len(trial_starts) - 1):
        start_time, end_time = trial_starts[ti], trial_starts[ti + 1]
        
        stim_start, stim_end = stim_events[ti], stim_events[ti + 1]
        stim_events_in_interval = stim_ts[stim_start:stim_end]
        
        trial_dict = {
            "trial_rate": len(behavior_data.stimulus_event_timestamps[ti]),
            "detected_events": stim_end - stim_start,
            "trial_start": start_time,
            "stim_ts": stim_events_in_interval,
            "first_stim_ts": stim_events_in_interval[0] if len(stim_events_in_interval) > 0 else np.nan,
            "trial_outcome": behavior_data.outcome_record[ti]
        }
        
        # Vectorized operations for port events
        for port_name, events in port_events.items():
            for event_type in ['entries', 'exits']:
                event_times = events[event_type]
                mask = (event_times > start_time) & (event_times < end_time)
                trial_dict[f"{port_name}_{event_type}"] = event_times[mask]
        
        trial_data.append(trial_dict)
    
    return pd.DataFrame(trial_data)

def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    """ 
    Get the spike times for each cluster individually.
    
    Parameters:
    -----------
    spike_times : array-like
        Array of all spike times.
    spike_clusters : array-like
        Array of cluster IDs for each spike.
    good_unit_ids : array-like
        Array of IDs for good units.
    
    Returns:
    --------
    list
        List of spike times for each good cluster.
    """
    return [spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu] for uclu in np.unique(spike_clusters[good_unit_ids])]

def get_good_units(clusters_obj, spike_clusters):
    """ 
    Filter Kilosort results using criteria specified in Melin et al. 2024.

    Parameters:
    -----------
    clusters_obj : object from spks.clusters.Clusters
        Object to access spike sorting results.
    spike_clusters : ndarray
        Spike clusters output from Kilosort (i.e. "../spike_clusters.npy").

    Returns:
    --------
    good_unit_ids : boolean ndarray
        Boolean array where True means that a recorded spike corresponded to a filtered spike cluster.
    n_units : int
        Number of single units filtered.
    """
    mask = ((np.abs(clusters_obj.cluster_info.trough_amplitude - clusters_obj.cluster_info.peak_amplitude) > 50)
            & (clusters_obj.cluster_info.amplitude_cutoff < 0.1) 
            & (clusters_obj.cluster_info.isi_contamination < 0.1)
            & (clusters_obj.cluster_info.presence_ratio >= 0.6)
            & (clusters_obj.cluster_info.spike_duration > 0.1))

    good_unit_ids = np.isin(spike_clusters,clusters_obj.cluster_info[mask].cluster_id.values)
    n_units = len(clusters_obj.cluster_info[mask])

    return good_unit_ids, n_units

def get_population_firing_rate(event_times, spike_times, tpre, tpost, binwidth_ms, kernel=None, window_ms=None):
    """
    Calculate population firing rate and optionally apply smoothing.

    Parameters:
    -----------
    event_times : array-like
        Timestamps of events.
    spike_times : list of array-like
        List of spike times for each unit.
    tpre : float
        Time before event to include in calculation.
    tpost : float
        Time after event to include in calculation.
    binwidth_ms : float
        Width of time bins in milliseconds.
    kernel : array-like, optional
        Kernel for smoothing (default is None).
    window_ms : float, optional
        Window size for moving average in milliseconds (default is None).

    Returns:
    --------
    psth : array-like
        Population firing rate (PSTH).
    unit_fr : array-like
        Firing rates for individual units.
    """
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
            psth = centered_moving_average(psth, window_size_bins)
        elif psth.ndim == 2:
            psth = np.array([centered_moving_average(row, window_size_bins) for row in psth])

        unit_fr = np.array([centered_moving_average(np.mean(row, axis=0), window_size_bins) for row in unit_fr])

    return psth, unit_fr

def compute_mean_sem(psth : np.array):
    """
    Compute mean and standard error of the mean for a given PSTH.

    Parameters:
    -----------
    psth : np.array
        Population spike time histogram.

    Returns:
    --------
    tuple
        Mean and standard error of the mean of the PSTH.
    """
    return np.mean(psth, axis=0), np.std(psth, axis=0) / np.sqrt(psth.shape[0])

def get_nth_element(x, i):
    """ 
    Get the n-th event in an array.
    
    Parameters:
    -----------
    x : array-like
        Input array.
    i : int
        Index of the element to retrieve.
    
    Returns:
    --------
    float or np.nan
        The i-th element of x if it exists, otherwise np.nan.
    
    Example usage:
    --------------
    for i, ax in enumerate(axs):
        for outcome, c in zip(np.unique(stim_ts_per_trial.trial_outcome), ['b', 'k', 'r', 'y']):
            ts = np.hstack(stim_ts_per_trial[stim_ts_per_trial.trial_outcome == outcome].stim_ts.apply(lambda x: get_nth_element(x, i)))
    """
    if len(x) > i and not np.isnan(x[0]):
        return x[i]
    return np.nan

def get_response_ts(row):
    """ 
    Get the timestamp of when the animal responded, regardless of which side it was.
    
    Parameters:
    -----------
    row : pd.Series
        A row from a DataFrame containing port entry and exit times.
    
    Returns:
    --------
    float or None
        Timestamp of the animal's response, or None if no valid response.
    
    Example usage:
    --------------
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
    """
    Get stimulus timestamps that occur during the stationary period of a trial.

    Parameters:
    -----------
    row : pd.Series
        A row from a DataFrame containing stimulus timestamps and first stimulus time.
    max_tseconds : float, optional
        Maximum time in seconds to consider for stationary period (default is 0.4).

    Returns:
    --------
    array-like
        Stimulus timestamps within the stationary period.
    """
    return row.stim_ts[row.stim_ts < row.first_stim_ts + max_tseconds]

def get_movement_stims(row, max_tseconds = 0.4):
    """
    Get stimulus timestamps that occur during the movement period of a trial.

    Parameters:
    -----------
    row : pd.Series
        A row from a DataFrame containing stimulus timestamps and center port exit times.
    max_tseconds : float, optional
        Maximum time in seconds to consider for movement period (default is 0.4).

    Returns:
    --------
    array-like
        Stimulus timestamps within the movement period.
    """
    return row.stim_ts[np.logical_and(row.center_port_exits[-1] < row.stim_ts, row.stim_ts < row.center_port_exits[-1] + max_tseconds)]

def moving_average(data, window_size):
    """
    Calculate the moving average of a 1D array.

    Parameters:
    -----------
    data : array-like
        Input data.
    window_size : int
        Size of the moving window.

    Returns:
    --------
    array-like
        Moving average of the input data.
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def centered_moving_average(data, window_size):
    """
    Calculate the centered moving average of a 1D array.

    Parameters:
    -----------
    data : array-like
        Input data.
    window_size : int
        Size of the moving window.

    Returns:
    --------
    array-like
        Centered moving average of the input data.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size