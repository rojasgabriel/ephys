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