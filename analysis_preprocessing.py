import numpy as np
import os
import re
from os.path import join as pjoin
from glob import glob
import spks
from utils import load_sync_data, process_port_events, process_trial_data, get_good_units, get_cluster_spike_times
from tqdm import tqdm

data_path = '/Volumes/grb_ephys/data'
local_data_path = '/Users/gabriel/data'
mice_paths = glob(pjoin(data_path, "GRB*"))

# Ask the user if they want to run the script in overwrite mode
overwrite_mode = input("Do you want to run the script in overwrite mode? (y/n): ").strip().lower() == 'y'

# Ask the user if they want to specify a particular mouse
specific_mouse = input("Do you want to specify a particular mouse? (y/n): ").strip().lower() == 'y'

selected_mouse = None
selected_session = None

if specific_mouse:
    selected_mouse = input("Enter the mouse ID (e.g., GRB123): ").strip()
    # Ask the user if they want to specify a particular session only if a mouse is specified
    specific_session = input("Do you want to specify a particular session? (y/n): ").strip().lower() == 'y'
    if specific_session:
        selected_session = input("Enter the session ID (e.g., 20230101_123456): ").strip()
else:
    specific_session = None

print("Finding mice directories...")
mice = []
sessions = {}

for path in mice_paths:
    mouse_id = re.search(r"GRB\d+", os.path.basename(path)).group(0)
    if specific_mouse and mouse_id != selected_mouse:
        continue
    mice.append(mouse_id)
    print(f"Found mouse: {mouse_id}")
    session_paths = glob(f"{path}/*")
    sessions[mouse_id] = []

    for session_path in session_paths:
        session_id = re.search(r"\d{8}_\d{6}", os.path.basename(session_path)).group(0)
        if specific_session and session_id != selected_session:
            continue
        sessions[mouse_id].append(session_id)
        print(f"Found session: {session_id}")

print("Processing data for each mouse and session...")
auto_mode = input(f"Do you want to run in auto mode? (y/n): ").strip().lower() == 'y'
save_locally = input(f"Do you want to save the processed data in the home data folder? (y/n): ").strip().lower() == 'y' 

for mouse in mice:
    for session in tqdm(sessions[mouse], desc=f"Processing sessions for {mouse}"):
        base_path = local_data_path if save_locally else data_path
        save_dir = pjoin(base_path, mouse, session, 'pre_processed')

        if not auto_mode:
            process_session = input(f"Do you want to process session {session}? (y/n): ").strip().lower() == 'y'
            if not process_session:
                print(f"Skipping session {session}.")
                continue
        
        try:
            session_path = pjoin(data_path, mouse, session)


            spike_times_file = pjoin(save_dir, 'spike_times_per_unit.npy')
            trial_ts_file = pjoin(save_dir, 'trial_ts.pkl')

            if not overwrite_mode and os.path.exists(spike_times_file) and os.path.exists(trial_ts_file):
                print(f"Pre-processed files already exist for session {session}. Skipping processing.")
                continue

            print(f"Loading sync data for session: {session}")
            corrected_onsets, corrected_offsets, t, srate, analog_signal = load_sync_data(session_path)
            
            print("Processing port events...")
            trial_starts, port_events = process_port_events(corrected_onsets, corrected_offsets, srate)
            
            print("Processing trial data...")
            behavior_data, trial_ts = process_trial_data(session_path, trial_starts, t, srate, analog_signal, port_events, mouse, session)

            kilosort_path = pjoin(session_path, 'kilosort2.5/imec0/')
            print("Loading spike data...")
            sc = np.load(pjoin(kilosort_path, 'spike_clusters.npy')) #KS clusters
            ss = np.load(pjoin(kilosort_path, 'spike_times.npy')) #KS spikes (in samples)
            st = ss/srate #conversion from spike samples to spike times

            print("Getting good units...")
            clu = spks.clusters.Clusters(folder=kilosort_path, get_waveforms=False, get_metrics=True, load_template_features=True)
            good_units_mask, n_units = get_good_units(clusters_obj=clu, spike_clusters=sc)
            print(f"Number of good units: {n_units}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Created directory: {save_dir}")

            print("Saving spike times per unit...")
            spike_times_per_unit = get_cluster_spike_times(spike_times=st, spike_clusters=sc, good_unit_ids=good_units_mask)
            spike_times_array = np.array(spike_times_per_unit, dtype=object)

            np.save(spike_times_file, spike_times_array, allow_pickle=True)
            trial_ts.to_pickle(trial_ts_file)
            print(f"Data saved for session: {session}")
        except Exception as e:
            import traceback
            print(f"Error processing session {session}: {e}")
            traceback.print_exc()
            continue