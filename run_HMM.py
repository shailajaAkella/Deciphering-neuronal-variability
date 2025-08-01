import numpy as np
import os
from state_utils import HMM
import lfp_utils as lu
import data_utils as du
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import matplotlib.pyplot as plt

data_dir = "D:/ecephys__project_cache/" # path to manifest.json
stim = 'natural_movie_one_more_repeats'
session_id = 767871931

manifest_path = os.path.join(data_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session = cache.get_session_data(session_id)
stim_table = du.trials_table(session, stim)


cortex = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISam', 'VISpm']
Fs_lfp = 1250
n_tr = len(stim_table)
T = (np.round(np.mean(stim_table.duration.values))*Fs_lfp).astype(int)
n_bands = 4

# Process LFPs
probe_ids = session.probes.index.values
probes = session.probes.description.values
lfp = {probe: [] for probe in probes}
not_found = []
for p_id, probe in zip(probe_ids, probes):
    try:
        lfp[probe] = session.get_lfp(p_id)
    except ValueError:
        print('LFP NWB not found for ' + probe)
        not_found.append(probe)

hilbert_matrix = {probe: [] for probe in probes if probe not in not_found}
channel_ids = {probe: [] for probe in probes if probe not in not_found}
for probe, probe_id in zip(probes, probe_ids):
    if probe in not_found:
            continue
    channels_CRTX = session.channels[(session.channels.probe_id == probe_id) & \
                                         (session.channels.ecephys_structure_acronym.isin(cortex))].index.values
    if len(channels_CRTX) == 0:
        print(f'No channels found for {probe} in cortex')
        not_found.append(probe)
        continue
    lfp_channels = np.array(lfp[probe].channel.values)
    num_CRTX = len(np.where((lfp_channels <= np.max(channels_CRTX)) & (lfp_channels >= np.min(channels_CRTX)))[0]) if len(channels_CRTX) > 0 else 0
    
    
    fields = lfp[probe].sel(channel=slice(np.min(channels_CRTX), np.max(channels_CRTX)))
    hilbert_fields = lu.hilbert_transform(lu.car(fields), Fs_lfp)

    hilbert_cortex = np.zeros((num_CRTX, n_tr, n_bands, T))
    presentation_ids = stim_table.index.values
    for trial, pres_id in enumerate(presentation_ids):
        pres_row = stim_table.loc[pres_id]
        start = pres_row['Start']        
        start_time_index = np.where(fields.time.values >= start)[0][0]
        end_time_index = start_time_index + T
        hilbert_cortex[:, trial, :, :] = np.transpose(hilbert_fields[start_time_index:end_time_index])
            
    hilbert_matrix[probe] = hilbert_cortex
    channel_ids[probe] = fields.channel.values

# bin LFP
fs = 30
binned_lfp_events = {probe: [] for probe in probes if probe not in not_found}
for probe in probes:
    if probe in not_found:
        continue
    select_channels, channel_to_layers = lu.get_layers(session, probe, channel_ids[probe])
    if np.sum(select_channels) == 0:
        not_found.append(probe)
        print(f'No channels selected for {probe}')
        continue
    channel_nos = np.concatenate([np.where(channel_ids[probe] == idx)[0] for idx in select_channels])
    binned_lfp_events[probe] = du.bin_data(hilbert_matrix[probe][channel_nos], Fs_lfp, fs)


# Make observations for HMM
observations_matrix = []
binned_hilbert_lfp3 = {probe: [] for probe in probes}
for n, probe in enumerate(probes):
    if probe in not_found:
        continue
    observations_matrix.append(binned_lfp_events[probe])
observations_matrix = np.nan_to_num(np.concatenate(observations_matrix))

hmm = HMM(num_states = 3)
hmm = hmm.fit(observations_matrix)

states = hmm.states
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
states = states.reshape(n_tr, -1)
plt.imshow(states, aspect='auto', cmap='viridis')
plt.colorbar(label='States')
ax.set_title('HMM States')
ax.set_ylabel('Trials')
ax.set_xticks(np.arange(0, states.shape[1], fs))
ax.set_xticklabels(np.arange(0, states.shape[1] // fs))
ax.set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

# state definition
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
freq = ['3-8 Hz', '10-30 Hz', '30-50 Hz', '50-80 Hz']
for i in range(hmm.num_states):
    ax.plot(freq, hmm.state_definition[:, i], marker='o', label=f'State {i}')
ax.set_xlabel('Frequency Bands')
ax.set_ylabel('Normalized activity')
ax.set_title('State Definition')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
    




    

    
    




