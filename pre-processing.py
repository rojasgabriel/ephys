#%% Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode
# import chiCa
from spks import *
from spks.sync import load_ni_sync_data,interp1d

# %matplotlib qt #to make figures pop out if running an interactive session

#%% Load npx sync data
sessionpath = Path('/home/data/GRB006/20240429_174359/')
sync_port = 0 # this is where the SMA of the probe is connected

(nionsets,nioffsets),(nisync,nimeta),(apsyncdata) = load_ni_sync_data(sessionpath=sessionpath)
aponsets = apsyncdata[0]['file0_sync_onsets'][6] # this should be the same for you, its where the sync is on the probe

corrected_onsets = {} # This is a dictionary with the digital events that were connected to the breakout box.
for k in nionsets.keys():
    corrected_onsets[k] = interp1d(nionsets[sync_port],aponsets,fill_value='extrapolate')(nionsets[k]).astype('uint64')

# if you need analog channels those are in "nisync"
nitime = interp1d(nionsets[sync_port],aponsets,fill_value='extrapolate')(np.arange(len(nisync)))

# everything is in samples, use this sampling rate
srate = apsyncdata[0]['sampling_rate']  

frame_rate = mode(1/(np.diff(corrected_onsets[1])/srate)) #corrected_onsets[1] are the frame samples, [2] are the trial start samples
trial_start_times = corrected_onsets[2][:-1]/srate

analog_signal = nisync[:, 0] # analog stim signal
t = nitime/srate #nidaq clock time vector in seconds
threshold = 15000
ii = np.where(np.diff(analog_signal>threshold)==1)[0]
stim_events_nidaq = t[ii[np.diff(np.hstack([0,ii]))>0.04*srate]] #just getting this for now. eventually i will extract other task variables here as well

#%% Load Kilosort data
kilosort_path = Path('/home/data/GRB006/20240429_174359/kilosort2.5/imec0/') #hardcoded to the session I'm prototyping stuff with

clu = Clusters(folder = kilosort_path, get_waveforms=False, get_metrics=True, load_template_features=True) #an object from the spks library

# I'm now curating units using the criteria Max and Joao used in their holder paper
# ---------- this gets the row indices ---------- #
single_unit_idx = np.where((np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude) > 50)
            & (clu.cluster_info.amplitude_cutoff < 0.1) 
            & (clu.cluster_info.isi_contamination < 0.1)
            & (clu.cluster_info.presence_ratio >= 0.6)
            & (clu.cluster_info.spike_duration > 0.1))[0]

# ---------- and this get the cluster_id values ---------- #
mask = ((np.abs(clu.cluster_info.trough_amplitude - clu.cluster_info.peak_amplitude) > 50)
            & (clu.cluster_info.amplitude_cutoff < 0.1) 
            & (clu.cluster_info.isi_contamination < 0.1)
            & (clu.cluster_info.presence_ratio >= 0.6)
            & (clu.cluster_info.spike_duration > 0.1))


single_unit_ids = clu.cluster_info[mask].cluster_id.values

#%% Plotting population PSTH response to all stimulus onsets
sc = np.load('/home/data/GRB006/20240429_174359/kilosort2.5/imec0/spike_clusters.npy') #KS clusters
ss = np.load('/home/data/GRB006/20240429_174359/kilosort2.5/imec0/spike_times.npy') #KS spikes (in samples)

st = ss/srate #conversion from spike samples to spike times

selection = np.isin(sc,single_unit_ids)

binsize = 0.005 # 10ms binsize
edges = np.arange(0,np.max(st[selection]),binsize)

pop_rate,_ = np.histogram(st[selection],edges)
pop_rate = pop_rate/binsize
pop_rate_time = edges[:-1]+np.diff(edges[:2])/2

psth = []
tpre = 0.025
tpost = 0.2

for onset in stim_events_nidaq:#stim_events_nidaq:
    if ~np.isnan(onset):
       psth.append(pop_rate[(pop_rate_time>= onset -tpre) & (pop_rate_time< onset +tpost)])
psth = np.stack(psth)
kernel_fig = plt.figure(figsize=(4,4))
plt.imshow(psth,aspect='auto',extent=[-tpre,tpost,0,len(psth)],cmap = 'RdBu_r',clim = [0,2500])
plt.vlines(x = 0, ymin = plt.ylim()[0], ymax = plt.ylim()[1], linestyles = 'dashed', color = 'k')
plt.colorbar(label='Population rate (Hz)')
plt.xlabel('time from stim event')
plt.ylabel('Number of stim events')
kernel_fig.tight_layout()
plt.show() #needed if running this script from the terminal