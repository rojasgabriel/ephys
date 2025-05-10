import numpy as np
import pandas as pd
import contextlib
import io
import spks
import chiCa.chiCa as chiCa
from os.path import join as pjoin
import scipy
from glob import glob

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
    behavior_data = chiCa.load_trialdata(glob(pjoin(sessionpath, f'chipmunk/{animal}_{session}_chipmunk_*.mat'))[0]) #TODO: test now with glob instead of hardcoding the DemonAudiTask name
    
    if port_events is None:
        raise Warning('No port events found. Proceeding without them...')
        trial_ts = get_trial_ts(trial_starts, detect_stim_events(t, srate, analog_signal, amp_threshold=5000), behavior_data)
    else:
        trial_ts = get_trial_ts(trial_starts, detect_stim_events(t, srate, analog_signal, amp_threshold=5000), behavior_data, port_events)
        trial_ts = trial_ts[trial_ts.trial_outcome != -1].copy()

        for itrial, data in trial_ts.iterrows():
            if len(data.center_port_entries) == 0:
                continue
            mask = (data.center_port_entries < data.first_stim_ts) #true if entry is before first stim onset. sometimes mice will briefly poke again in the center before reporting their choice
            true_entry = data.center_port_entries[mask][-1] #get the last entry before stim onset since sometimes the first poke will not be long enough to be considered a valid entry
            trial_ts.loc[itrial, 'center_port_entries'] = [true_entry] #replace the entries with the last entry before stim onset

            mask = (data.center_port_exits > true_entry) #true if exit is after the last entry before stim onset
            true_exit = data.center_port_exits[mask][0] #get the first exit after stim onset
            trial_ts.loc[itrial, 'center_port_exits'] = [true_exit] #replace the exits with the first exit after stim onset
        trial_ts.insert(trial_ts.shape[1], 'response', trial_ts.apply(get_response_ts, axis=1))
        
    trial_ts.insert(trial_ts.shape[1], 'stationary_stims', trial_ts.apply(get_stationary_stims, axis=1))
    trial_ts.insert(trial_ts.shape[1], 'movement_stims', trial_ts.apply(get_movement_stims, axis=1))

    trial_ts.insert(0, 'category', trial_ts.apply(lambda x: 'left' if x.trial_rate < 12 else ('right' if x.trial_rate > 12 else 'boundary'), axis=1))
    
    print('Success!')
    return behavior_data, trial_ts

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

def get_stationary_stims(row):
    return row.stim_ts[row.stim_ts < row.center_port_exits[0]]

def get_movement_stims(row, max_tseconds=0.4):
    if 'center_port_exits' in row.index: #this check might be redundant 4/16/25
        if len(row.center_port_exits) != 0:
            return row.stim_ts[row.stim_ts > row.center_port_exits[0]]
    else:
        return row.stim_ts[np.logical_and(row.first_stim_ts + 0.5 < row.stim_ts, row.stim_ts < row.first_stim_ts + 0.5 + max_tseconds)]

