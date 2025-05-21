import pandas as pd
import numpy as np

def calculate_stim_offsets(trial_ts, trial_start_col='center_port_entries'):
    """
    Calculates the offset of each stationary and movement stimulus relative 
    to the specified trial start time column for each trial.
    
    Returns a DataFrame with trial_idx, stim_time, offset, and movement_status.
    """
    records = []
    for idx, row in trial_ts.iterrows():
        trial_start_time = (row[trial_start_col])[0]
        
        # Handle potential None values in stim lists
        stationary_stims = row['stationary_stims'][1:] if row['stationary_stims'] is not None else [] #exclude the first stim
        movement_stims = row['movement_stims'] if row['movement_stims'] is not None else []

        for st in stationary_stims:
            records.append({
                'trial_idx': idx,
                'stim_time': st,
                'offset': st - trial_start_time,
                'movement_status': 0  # 0 for stationary
            })
            
        for mt in movement_stims:
            records.append({
                'trial_idx': idx,
                'stim_time': mt,
                'offset': mt - trial_start_time,
                'movement_status': 1  # 1 for movement
            })

    stims_offset_df = pd.DataFrame(records)
    return stims_offset_df

def find_unique_cross_trial_offset_pairs(stims_offset_df, wiggle_room=0.1):
    """
    Finds pairs of stationary (movement_status=0) and movement (movement_status=1) 
    stimuli from *different* trials whose offsets are within wiggle_room of each other.
    Ensures that each stimulus (stationary or movement) is used in at most one pair.
    
    Args:
        stims_offset_df (pd.DataFrame): DataFrame from calculate_stim_offsets.
        wiggle_room (float): Maximum allowed absolute difference between offsets.
        
    Returns:
        pd.DataFrame: DataFrame containing uniquely matched pairs.
    """
    
    # Use original index for tracking usage
    stims_offset_df = stims_offset_df.reset_index().rename(columns={'index': 'original_index'})
    
    stationary_stims = stims_offset_df[stims_offset_df['movement_status'] == 0].copy()
    movement_stims = stims_offset_df[stims_offset_df['movement_status'] == 1].copy()
    
    matched_pairs = []
    used_stationary_indices = set()
    used_movement_indices = set()
    
    # Sort stationary stimuli by offset to potentially make matching more consistent (optional)
    # stationary_stims = stationary_stims.sort_values('offset')

    # Iterate through each stationary stimulus
    for _, stat_row in stationary_stims.iterrows():
        stat_idx = stat_row['original_index']
        
        # Skip if this stationary stimulus has already been used
        if stat_idx in used_stationary_indices:
            continue
            
        # Find potential movement stimuli in DIFFERENT trials with close offsets
        potential_matches = movement_stims[
            (movement_stims['trial_idx'] != stat_row['trial_idx']) &
            (np.abs(movement_stims['offset'] - stat_row['offset']) <= wiggle_room) &
            (~movement_stims['original_index'].isin(used_movement_indices)) # Only consider unused movement stims
        ].copy()
        
        # Optional: Sort potential matches by offset difference to pick the closest one first
        potential_matches['offset_diff'] = np.abs(potential_matches['offset'] - stat_row['offset'])
        potential_matches = potential_matches.sort_values('offset_diff')

        # If there are any available matches, take the first one (closest offset diff if sorted)
        if not potential_matches.empty:
            move_row = potential_matches.iloc[0]
            move_idx = move_row['original_index']
            
            # Record the pair
            matched_pairs.append({
                'stat_trial_idx': stat_row['trial_idx'],
                'stat_stim_time': stat_row['stim_time'],
                'stat_offset': stat_row['offset'],
                'move_trial_idx': move_row['trial_idx'],
                'move_stim_time': move_row['stim_time'],
                'move_offset': move_row['offset'],
                'offset_diff': move_row['offset_diff'] # Use calculated diff
            })
            
            # Mark both stimuli as used
            used_stationary_indices.add(stat_idx)
            used_movement_indices.add(move_idx)
            
    matched_df = pd.DataFrame(matched_pairs)
    print(f"Found {len(matched_df)} unique cross-trial pairs with offset difference <= {wiggle_room}s.")
    return matched_df

def compute_stim_response_for_trial_subset(spike_times_per_unit,
                                           trial_subset, 
                                           pre_seconds,
                                           post_seconds,
                                           binwidth_ms,
                                           stim_window_start,
                                           stim_window_end,
                                           wiggle_room=0.010):
    """
    Given a trial subset (slow or fast RT), compute the mean population response
    for stationary and movement conditions using matched unique pairs.
    Returns a dict with keys 'Stationary' and 'Movement'.
    """
    #TODO: make this function general for comparing any two subsets of trials
    from spks.event_aligned import population_peth

    data = trial_subset.copy()
    stims_offset_df = calculate_stim_offsets(data, trial_start_col='center_port_entries')
    matched_pairs_df = find_unique_cross_trial_offset_pairs(stims_offset_df, wiggle_room=wiggle_room)
    n_matched_stims = len(matched_pairs_df)
    if matched_pairs_df.empty:
        print("No matched pairs found!")
        return None

    # Alignment times for stationary and movement stims
    stat_alignment_times = matched_pairs_df['stat_stim_time'].values
    move_alignment_times = matched_pairs_df['move_stim_time'].values

    pop_peth_stat, timebin_edges_stat, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=stat_alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        pad=0,
        kernel=None,
    )
    pop_peth_move, _, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=move_alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        pad=0,
        kernel=None,
    )
    
    # Define response window based on timebins (use stationary timebins as reference)
    stimulus_window_bool = (timebin_edges_stat[:-1] >= stim_window_start) & (timebin_edges_stat[:-1] <= stim_window_end)

    stationary_response = np.max(np.mean(pop_peth_stat[:, :, stimulus_window_bool], axis=1), axis=1) #mean across trials and then max of the stimulus window
    running_response = np.max(np.mean(pop_peth_move[:, :, stimulus_window_bool], axis=1), axis=1)
    
    n_neurons = pop_peth_stat.shape[1]
    stationary_sem = np.std(np.max(pop_peth_stat[:, :, stimulus_window_bool], axis=2), axis=1) / np.sqrt(n_neurons)
    running_sem = np.std(np.max(pop_peth_move[:, :, stimulus_window_bool], axis=2), axis=1) / np.sqrt(n_neurons)
    return {"stationary": stationary_response, 
            "stationary_sem" : stationary_sem,
            "running": running_response,
            "running_sem" : running_sem}, n_matched_stims


#%% General functions
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