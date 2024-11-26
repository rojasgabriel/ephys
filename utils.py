import numpy as np
import pandas as pd
import contextlib
import io
import spks
import chiCa.chiCa as chiCa
from os.path import join as pjoin
import scipy

### 1. CORE DATA LOADING ###
def load_sync_data(sessionpath, sync_port=0):
    print('Loading nisync data...')
    (nionsets,nioffsets),(nisync,nimeta),(apsyncdata) = spks.sync.load_ni_sync_data(sessionpath=sessionpath)
    aponsets = apsyncdata[0]['file0_sync_onsets'][6]

    corrected_onsets = {}
    corrected_offsets = {}
    for k in nionsets.keys():
        corrected_onsets[k] = spks.sync.interp1d(nionsets[sync_port],aponsets,fill_value='extrapolate')(nionsets[k]).astype('uint64')
        corrected_offsets[k] = spks.sync.interp1d(nionsets[sync_port],aponsets,fill_value='extrapolate')(nioffsets[k]).astype('uint64')
    del k

    nitime = spks.sync.interp1d(nionsets[sync_port],aponsets,fill_value='extrapolate')(np.arange(len(nisync)))
    srate = apsyncdata[0]['sampling_rate']
    t = nitime/srate
    frame_rate = scipy.stats.mode(1/(np.diff(corrected_onsets[1])/srate))
    analog_signal = nisync[:, 0]
    print('Success!\n-----')
    
    return corrected_onsets, corrected_offsets, t, srate, analog_signal

def process_port_events(corrected_onsets, corrected_offsets, srate):
    print('Loading nidaq events...')
    trial_starts = corrected_onsets[2]/srate
    
    if len(corrected_onsets.keys()) <= 3:
        print('No port events registered in this session. Proceeding without them...')
        return trial_starts, None
        
    print('Port events found. Proceeding with extracting them...')
    port_events = {
        "center_port": {
            "entries": corrected_onsets[4]/srate,
            "exits": corrected_offsets[4]/srate
        },
        "left_port": {
            "entries": corrected_onsets[3]/srate,
            "exits": corrected_offsets[3]/srate
        },
        "right_port": {
            "entries": corrected_onsets[5]/srate,
            "exits": corrected_offsets[5]/srate
        }
    }
    
    return trial_starts, port_events

def process_trial_data(sessionpath, trial_starts, t, srate, analog_signal, port_events, animal, session):
    behavior_data = chiCa.load_trialdata(pjoin(sessionpath, f'chipmunk/{animal}_{session}_chipmunk_DemonstratorAudiTask.mat'))
    
    if port_events is None:
        trial_ts = get_trial_ts(trial_starts, detect_stim_events(t, srate, analog_signal, amp_threshold=5000), behavior_data)
    else:
        trial_ts = get_trial_ts(trial_starts, detect_stim_events(t, srate, analog_signal, amp_threshold=5000), behavior_data, port_events)
        trial_ts.insert(trial_ts.shape[1], 'response', trial_ts.apply(get_response_ts, axis=1))
        
    trial_ts.insert(trial_ts.shape[1], 'stationary_stims', trial_ts.apply(get_stationary_stims, axis=1))
    trial_ts.insert(trial_ts.shape[1], 'movement_stims', trial_ts.apply(get_movement_stims, axis=1))

    trial_ts.insert(0, 'category', trial_ts.apply(lambda x: 'left' if x.trial_rate < 12 else ('right' if x.trial_rate > 12 else 'boundary'), axis=1))
    
    print('Success!')
    return behavior_data, trial_ts

### 2. TRIAL & EVENT PROCESSING ###
def get_trial_ts(trial_starts, stim_ts, behavior_data, port_events=None):
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
            "stimulus_modality": behavior_data.stimulus_modality[ti],
            "response_side": behavior_data.response_side[ti],
            "correct_side": behavior_data.correct_side[ti],
            "trial_outcome": behavior_data.outcome_record[ti]
        }

        if port_events is not None:
            # Vectorized operations for port events
            for port_name, events in port_events.items():
                for event_type in ['entries', 'exits']:
                    event_times = events[event_type]
                    mask = (event_times > start_time) & (event_times < end_time)
                    trial_dict[f"{port_name}_{event_type}"] = event_times[mask]
        
        trial_data.append(trial_dict)
    
    return pd.DataFrame(trial_data)

def get_balanced_trials(trial_ts, require_both_stim_types=True):
    # Get valid trials (exclude early withdrawals)
    valid_trials = trial_ts[trial_ts.trial_outcome.isin([0,1])]
    
    # Optionally require both stim types
    if require_both_stim_types:
        valid_trials = valid_trials[
            (valid_trials.movement_stims.apply(len) > 0) & 
            (valid_trials.stationary_stims.apply(len) > 0)
        ]
    
    # Find minimum number of trials between conditions
    min_trials = min(
        len(valid_trials[valid_trials.trial_outcome == 1]),
        len(valid_trials[valid_trials.trial_outcome == 0])
    )
    
    # Sample equal numbers from each condition
    balanced_trials = pd.concat([
        valid_trials[valid_trials.trial_outcome == 1].sample(n=min_trials, random_state=42),
        valid_trials[valid_trials.trial_outcome == 0].sample(n=min_trials, random_state=42)
    ])
    
    return balanced_trials, min_trials

def detect_stim_events(time_vector, srate, analog_signal, amp_threshold=5000, time_threshold=0.04):
    ii = np.where(np.diff(analog_signal>amp_threshold)==1)[0]
    return time_vector[ii[np.diff(np.hstack([0,ii]))>time_threshold*srate]]

def get_response_ts(row):
    try:
        w = row['center_port_exits'][-1]  # withdrawal time
    except:
        w = None

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

def get_stationary_stims(row, max_tseconds=0.4):
    return row.stim_ts[row.stim_ts < row.first_stim_ts + max_tseconds]

def get_movement_stims(row, max_tseconds=0.4):
    if 'center_port_exits' in row.index:
        if row.center_port_exits.size != 0:
            return row.stim_ts[np.logical_and(row.center_port_exits[-1] < row.stim_ts, row.stim_ts < row.center_port_exits[-1] + max_tseconds)]
    else:
        return row.stim_ts[np.logical_and(row.first_stim_ts + 0.5 < row.stim_ts, row.stim_ts < row.first_stim_ts + 0.5 + max_tseconds)]

### 3. NEURAL DATA PROCESSING ###
def get_cluster_spike_times(spike_times, spike_clusters, good_unit_ids):
    return [spike_times[good_unit_ids][spike_clusters[good_unit_ids] == uclu] for uclu in np.unique(spike_clusters[good_unit_ids])]

def get_good_units(clusters_obj, spike_clusters):
    mask = ((np.abs(clusters_obj.cluster_info.trough_amplitude - clusters_obj.cluster_info.peak_amplitude) > 50)
            & (clusters_obj.cluster_info.amplitude_cutoff < 0.1) 
            & (clusters_obj.cluster_info.isi_contamination < 0.1)
            & (clusters_obj.cluster_info.presence_ratio >= 0.6)
            & (clusters_obj.cluster_info.spike_duration > 0.1)
            & (clusters_obj.cluster_info.firing_rate > 1)) #added this filter myself

    good_unit_ids = np.isin(spike_clusters,clusters_obj.cluster_info[mask].cluster_id.values)
    n_units = len(clusters_obj.cluster_info[mask])

    return good_unit_ids, n_units

### 4. GENERAL UTILITIES ###
@contextlib.contextmanager
def suppress_print():
    """ 
    Suppress print statements when not needed.
    
    This context manager redirects stdout to a StringIO object,
    effectively suppressing any print statements within its scope.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield

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
