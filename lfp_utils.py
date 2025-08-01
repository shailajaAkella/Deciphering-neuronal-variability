import os
import numpy as np
import pandas as pd
import numpy.matlib
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from sklearn.impute import SimpleImputer
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import sosfiltfilt, hilbert, butter

data_dir = "D:/ecephys__project_cache/"


def lfps(session_id, stim, type=None):
    probes = ['probeC', 'probeD', 'probeF', 'probeE', 'probeB', 'probeA']
    lfp = {probe: [] for probe in probes}
    channel_ids = {probe: [] for probe in probes}

    session_directory = os.path.join(data_dir + '/session_' + str(session_id))
    for probe in probes:
        probe_path = session_directory + "/MATLAB_files" + '/' + probe
        if type == 'car':
            lfp_mat_path = probe_path + '/' + stim + "_lfp_car.mat"
        else:
            lfp_mat_path = probe_path + '/' + stim + "_lfp.mat"
        if os.path.exists(lfp_mat_path):
            lfp[probe] = sio.loadmat(lfp_mat_path)['data'][:, :, :37500]
            channel_ids[probe] = sio.loadmat(lfp_mat_path)['channels'].reshape(-1)
    return lfp, channel_ids


def hilbert_transform(lfp_matrix, Fs):

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos

    bands = [
        (3, 8),    # theta
        (10, 30),  # beta
        (30, 50),  # gamma1
        (50, 80),  # gamma2
    ]
    n_bands = len(bands)
    T, n_ch = lfp_matrix.shape
    lfp_hilbert = np.zeros((T, n_bands, n_ch))

    # Precompute filters
    sos_filters = [butter_bandpass(low, high, Fs, 11) for (low, high) in bands]

    # Vectorized filtering and Hilbert transform per band
    for b, sos in enumerate(sos_filters):
        # Replace NaNs with zeros for filtering
        data = np.nan_to_num(lfp_matrix)
        # Apply filter to all channels at once (axis=0: time, axis=1: channels)
        filtered = sosfiltfilt(sos, data, axis=0)
        # Hilbert transform (analytic signal) for all channels at once
        lfp_hilbert[:, b, :] = np.abs(hilbert(filtered, axis=0))

    return lfp_hilbert


def get_csd(session_id, stim='flashes', plot=True):
    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    print(session_id)

    raw_lfp, channels = lfps(session_id, stim)
    session = cache.get_session_data(session_id)
    probes = ['probeC', 'probeD', 'probeF', 'probeE', 'probeB', 'probeA']
    csd_all = {probe: [] for probe in probes}
    if plot:
        fig, ax = plt.subplots(1, 6, figsize=(18, 6))
    for p_no, probe in enumerate(probes):
        print(probe)

        probe_id = session.probes[session.probes.description == probe].index.values[0]
        if raw_lfp[probe].shape[0] == 0:
            continue
        csd = session.get_current_source_density(probe_id)

        filtered_csd = gaussian_filter(csd.data, sigma=(5, 1))
        if plot:
            _ = ax[p_no].pcolor(csd["time"], csd["vertical_position"], filtered_csd, vmin=-3e4, vmax=3e4)

            _ = ax[p_no].set_xlabel("time relative to stimulus onset (s)")
            _ = ax[p_no].set_ylabel("vertical position (um)")
            plt.tight_layout()

        map = get_cortical_layer(probe_id)
        map_probe = map.loc[np.intersect1d(map.index.values, csd['vertical_position'].values)]
        layer_ends = map_probe.iloc[np.where(np.diff(map_probe['layer'].values) < 0)[0]].index.values
        layers = map_probe.loc[layer_ends]['layer'].values

        pos = np.array([session.channels.loc[ch].probe_vertical_position
                        if ch in session.channels.index.values else np.nan for ch in channels[probe]])
        impute = SimpleImputer(missing_values=np.nan, strategy='mean')
        pos = impute.fit_transform(pos.reshape(1, -1)).astype(int)

        if plot:
            for end, label in zip(layer_ends, layers):
                _ = ax[p_no].plot(csd["time"], end * np.ones(len(csd["time"])), color='white')
                _ = ax[p_no].text(0.01, end - 30, 'L' + str(label))
            _ = ax[p_no].text(0.01, np.max(pos) - 30, 'L2/3')
            _ = ax[p_no].set_ylim(np.min(pos), np.max(pos))
            _ = ax[p_no].set_xlim(0, 0.1)

        layer_annot = np.concatenate((layer_ends.reshape(-1, 1), layers.reshape(-1, 1)), axis=1)
        st = np.where(csd["vertical_position"] == np.min(pos))[0][0]
        end = np.where(csd["vertical_position"] == np.max(pos))[0][0]
        dict = {"time": csd["time"], "vert_pos": csd["vertical_position"][st:end],
                "filtered_csd": filtered_csd[st:end], "csd": csd[st:end], "has_lfp": pos,
                "layer_info": layer_annot}
        # layer_info - positions below the demarcated position is the corresponding layer
        csd_all[probe] = dict

    return csd_all


def get_cortical_layer(probe_id):
    df = pd.read_csv('data/unit_table.csv')
    pos_ = df[df.ecephys_probe_id.values == probe_id].probe_vertical_position.values
    layer_ = df[df.ecephys_probe_id.values == probe_id].cortical_layer.values
    ind = np.where((layer_ <= 6) & (layer_ > 0))
    map = pd.DataFrame()
    map['vertical_position'] = pos_[ind]
    map['layer'] = layer_[ind]
    map = map.drop_duplicates()  # check for duplicates in dataframe
    map = map.set_index('vertical_position')

    # check for duplicates in indexes
    map_final = pd.DataFrame()
    map_final['vertical_position'] = map.index.values[np.where(~map.index.duplicated(keep='first'))[0]]
    map_final['layer'] = map['layer'].values[np.where(~map.index.duplicated(keep='first'))[0]]
    map_final = map_final.set_index('vertical_position')
    return map_final


def get_layers(session, probe, lfp_channel_ids):
    map_channel_pos = pd.DataFrame()
    map_channel_pos['channel_id'] = lfp_channel_ids
    probe_id = session.units[session.units.probe_description == probe].probe_id.iloc[0]
    map_pos_layer = get_cortical_layer(probe_id)
    map_channel_pos['vertical_position'] = np.array([session.channels.loc[ch].probe_vertical_position
                                                     if ch in session.channels.index.values else np.nan for ch in
                                                     lfp_channel_ids])
    map_channel_pos = map_channel_pos.set_index('vertical_position')
    map_channel_layer = map_channel_pos.join(map_pos_layer)
    layer_vals = map_channel_layer.layer.values
    if 2 in layer_vals or 3 in layer_vals:
        s = map_channel_layer.channel_id[map_channel_layer.layer.eq(2) |
                                         map_channel_layer.layer.eq(3)].sample(random_state=0).values
    else:
        s = [map_channel_layer.channel_id.values[-1]]

    if 4 in layer_vals:
        m = map_channel_layer.channel_id[map_channel_layer.layer.eq(4)].sample(random_state=0).values
    else:
        m = [np.nan]

    if 5 in layer_vals or 6 in layer_vals:
        d = map_channel_layer.channel_id[map_channel_layer.layer.eq(5) |
                                         map_channel_layer.layer.eq(6)].sample(random_state=0).values
    else:
        d = [map_channel_layer.channel_id.values[0]]
    if np.sum([s, m, d]) > 0:
        return np.concatenate([s, m, d]), map_channel_layer
    else:
        return [0, 0, 0], map_channel_layer


def car(data):
    """
    This function calculates and returns the common average reference (CAR) of a 2D matrix 'data'.
    :param data: 2D NumPy array
    :return: 2D NumPy array with CAR applied
    """

    # Convert to double (float in Python)
    data = data.astype(float)

    transflag = False
    if data.shape[0] < data.shape[1]:
        data = data.T
        transflag = True

    num_chans = data.shape[1]

    # Create a CAR spatial filter matrix
    spatfiltmatrix = -np.ones((num_chans, num_chans))
    np.fill_diagonal(spatfiltmatrix, num_chans - 1)
    spatfiltmatrix = spatfiltmatrix / num_chans

    # Perform spatial filtering
    if spatfiltmatrix.size != 0:
        print('Spatial filtering')
        data = np.dot(data, spatfiltmatrix)
        if data.shape[1] != spatfiltmatrix.shape[0]:
            print('The first dimension in the spatial filter matrix has to equal the second dimension in the data')

    # If the data was transposed, transpose it back
    if transflag:
        data = data.T

    return data
