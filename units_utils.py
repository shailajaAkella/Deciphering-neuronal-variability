import os
import numpy as np
import pandas as pd
import random
from einops import rearrange
import sys
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from scipy import stats
import warnings
import math
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
import data_utils as du
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_dir = "D:/ecephys__project_cache/"
sys.path += [r"D:/ecephys__project_cache/", r"C:\Users\shailaja.akella\Dropbox ("
                                            r"Personal)\AI_neural_variability\code\python codes"]


def neural_activity(session_id, probe, stim, responsiveness=False, QC=False, area=[]):
    Fs_lfp = 1250
    session = du.get_nwb_session(session_id)
    trials = du.trials_table(session, stim)
    session_directory = os.path.join(data_dir + '/session_' + str(session_id))
    if not QC:
        session = EcephysSession.from_nwb_path(os.path.join(session_directory, 'session_' + str(session_id) + '.nwb'),
                                               api_kwargs={
                                                   "amplitude_cutoff_maximum": np.inf,
                                                   "presence_ratio_minimum": -np.inf,
                                                   "isi_violations_maximum": np.inf
                                               })
    else:
        session = EcephysSession.from_nwb_path(os.path.join(session_directory, 'session_' + str(session_id) + '.nwb'))

    probe_id = session.units[session.units.probe_description == probe].probe_id.iloc[0]
    session_units = session.units[session.units.probe_id == probe_id]
    if len(area) == 0:
        area = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISam', 'VISpm']
    session_units['unit_id'] = session_units.index
    cortical_units_ids = np.array(
        [idx for idx, ccf in enumerate(session_units.ecephys_structure_acronym.values) if ccf in area])
    session_units_cortex = session_units.iloc[cortical_units_ids]

    #   spike counts
    duration = round(np.mean(trials.duration.values))
    time_bin_edges = np.linspace(0, duration, int(duration * Fs_lfp) + 1)
    spike_counts = session.presentationwise_spike_counts(
        bin_edges=time_bin_edges,
        stimulus_presentation_ids=trials.index.values,
        unit_ids=session_units_cortex.index.values,
        binarize=True
    )

    SUA = np.squeeze(spike_counts.values)
    [n_tr, T, n_units] = SUA.shape
    sua_final = rearrange(SUA, 'b c h -> h b c')

    SUA = rearrange(SUA, 'b c h -> (b c) h')
    z_SUA = stats.zscore(SUA, axis=0)
    z_SUA = z_SUA.reshape([n_tr, T, n_units])

    if responsiveness:
        n_samples = 1000

        def baseline_shift_range(arr, amount):
            return np.apply_along_axis(lambda x: x + (np.random.rand(1)) * amount, 1, arr)

        trials_spontaneous = du.trials_table(session, "spontaneous")

        spontaneous_start = trials_spontaneous.iloc[0].Start
        spontaneous_end = trials_spontaneous.iloc[0].End
        spontaneous_trial_inds = np.ones((n_samples,)) * trials_spontaneous.index.values[0]
        baseline_shift_callback = lambda arr: baseline_shift_range(arr, spontaneous_end - spontaneous_start)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            responses_spont = session.presentationwise_spike_counts(time_bin_edges,
                                                                    stimulus_presentation_ids=spontaneous_trial_inds,
                                                                    unit_ids=session_units_cortex.index.values,
                                                                    time_domain_callback=baseline_shift_callback)

        resp_mean = spike_counts.mean(dim='time_relative_to_stimulus_onset')
        resp_spont_mean = responses_spont.mean(dim='time_relative_to_stimulus_onset')

        def get_sig_fraction(unit_id):
            resp_for_unit = resp_mean.sel(unit_id=unit_id).data
            spont_for_unit = resp_spont_mean.sel(unit_id=unit_id).data

            sig_level_spont = np.quantile(spont_for_unit, 0.95)
            sig_fraction_spont = np.sum(resp_for_unit > sig_level_spont) / len(resp_for_unit)

            return sig_fraction_spont

        responsiveness = [get_sig_fraction(unit_id) for unit_id in session_units_cortex.index.values]
        resp_mean = resp_mean.values.transpose()
        resp_spont_mean = resp_spont_mean.values.transpose()

        return probe, np.nanmean(z_SUA, axis=-1), sua_final, responsiveness
    else:
        return probe, np.nanmean(z_SUA, axis=-1), sua_final


def qc_val(key_name):
    qc = {'rate_threshold': 1,  # greater than
               'presence_ratio': 0.95,  # greater than
               'amplitude_cutoff': 0.1,  # less than
               'isi_violations': 0.5,  # less than
               'RF_max': 8,  # less than
               'RF_min': 0,  # greater than
               'r2_threshold': 0.005,
               'waveform_threshold': 0.4}
    return qc[key_name]


def unit_metrics(session_id, unit_table, probes):
    session = du.get_nwb_session(session_id, with_qc=False)
    rf_mapping = ReceptiveFieldMapping(session)
    unit_data = pd.DataFrame()

    for p_no, probe in enumerate(probes):

        if len(session.units[session.units.probe_description == probe].probe_id) == 0:
            continue
        probe_id = session.units[session.units.probe_description == probe].probe_id.iloc[0]
        session_units = session.units[session.units.probe_id == probe_id]
        area = ['VISp', 'VISl', 'VISli', 'VISrl', 'VISal', 'VISam', 'VISpm']
        session_units = session_units.rename(columns={"channel_local_index": "channel_id",
                                                      "ecephys_structure_acronym": "ccf",
                                                      'probe_vertical_position': "ypos"})

        session_units['unit_id'] = session_units.index
        cortical_units_ids = np.array([idx for idx, ccf in enumerate(session_units.ccf.values) if ccf in area])
        session_units_cortex = session_units.iloc[cortical_units_ids]
        n_units = len(session_units_cortex)

        RF = np.array([[np.argmax(np.max(rf_mapping.get_receptive_field(unit), axis=0)),  # x
                        np.argmax(np.max(rf_mapping.get_receptive_field(unit), axis=1))]  # y
                       for unit in session_units_cortex['unit_id'].values])

        cortical_layers = np.array([unit_table[unit_table.iloc[:, 0] == unit]['cortical_layer'].values
                                    for unit in session_units_cortex['unit_id'].values])

        qc_layer = []
        for layer in cortical_layers:
            if 0 <= layer <= 1:
                l = np.nan
            elif 2 <= layer <= 3:
                l = 'L23'
            elif layer == 4:
                l = 'L4'
            elif 5 <= layer <= 6:
                l = 'L56'
            else:
                l = np.nan
            qc_layer.append(l)

        temp_df = pd.DataFrame()
        temp_df['unit_id'] = session_units_cortex.index
        temp_df['unit'] = np.arange(0, len(session_units_cortex.index))
        temp_df['probe'] = probe
        temp_df['session_id'] = session_id
        temp_df['total_rate'] = session_units_cortex['firing_rate'].values
        temp_df['presence_ratio'] = session_units_cortex['presence_ratio'].values
        temp_df['amplitude_cutoff'] = session_units_cortex['amplitude_cutoff'].values
        temp_df['isi_violations'] = session_units_cortex['isi_violations'].values
        temp_df['rf_x'] = RF[:, 0] if n_units > 0 else -1
        temp_df['rf_y'] = RF[:, 1] if n_units > 0 else -1
        temp_df['cell_type'] = np.array(['RS' if session_units_cortex['waveform_duration'].values[nrn] > qc_val('waveform_threshold') else 'FS'
                                         for nrn in range(len(session_units_cortex['waveform_duration']))])
        temp_df['wf_duration'] = session_units_cortex['waveform_duration'].values
        temp_df['layer'] = np.array(qc_layer)
        temp_df['area'] = session_units_cortex['ccf'].values

        unit_data = unit_data.append(temp_df)

    return unit_data


def analysis_metrics(unit_ids):
    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    analysis_metrics = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity')

    inter_ids = analysis_metrics.index.intersection(unit_ids)
    df = pd.DataFrame()
    df['unit_id'] = inter_ids
    df['unit'] = [np.where(unit_ids == unit_id)[0][0] for unit_id in inter_ids]
    df['probe'] = analysis_metrics.loc[inter_ids].name.values
    df['session_id'] = analysis_metrics.loc[inter_ids].ecephys_session_id.values
    df['rf_area'] = analysis_metrics.loc[inter_ids].area_rf.values
    df['mod_idx_dg'] = analysis_metrics.loc[inter_ids].mod_idx_dg.values
    df['run_mod_dg'] = analysis_metrics.loc[inter_ids].run_mod_dg.values
    df['run_pval_dg'] = analysis_metrics.loc[inter_ids].run_pval_dg.values
    df['pref_ori_dg'] = analysis_metrics.loc[inter_ids].pref_ori_dg.values
    df['pref_tf_dg'] = analysis_metrics.loc[inter_ids].pref_tf_dg.values
    df['c50_dg'] = analysis_metrics.loc[inter_ids].c50_dg.values
    df['lifetime_sparseness_dg'] = analysis_metrics.loc[inter_ids].lifetime_sparseness_dg.values
    return df


def bin_spikes(sua, Fs_original, Fs):
    dims = sua.shape
    bin_w = math.floor(Fs_original / Fs)
    Tm = int(Fs * dims[-1] / Fs_original)
    binned = [np.nansum(sua[:, :, 0 + t * bin_w: bin_w + t * bin_w], axis=-1) for t in range(Tm)]
    return rearrange(np.array(binned), 'l b w -> b w l')


def rate_match_for_CV_FF(sua, min_rate, Fs):
    n_drop_bins = 5
    from_ids = list(zip(*np.where(~np.isnan(sua) & (sua > 0))))
    original_rate = np.nanmean(sua) * Fs
    while original_rate > np.max([min_rate, 1]):
        rand_ids = np.random.choice(len(from_ids), size=n_drop_bins, replace=False)
        for i in rand_ids:
            if sua[from_ids[i]] >= 1:
                sua[from_ids[i]] = sua[from_ids[i]] - 1
        original_rate = np.nanmean(sua) * Fs
    return sua


def sample_match_for_FF(sua, min_samples):
    from_ids = list(zip(*np.where(~np.isnan(sua))))
    rand_ids = np.random.choice(len(from_ids), size=min_samples, replace=False)
    mask = np.zeros_like(sua, dtype=bool)
    for idx in rand_ids:
        row, col = from_ids[idx]
        mask[row, col] = True

    sua[~mask] = np.nan
    return sua


def sample_match_for_SV(spike_count_mat, min_samples):
    return spike_count_mat[np.sort(random.sample(range(0, spike_count_mat.shape[0]), min_samples)), :]


def fano_factor(sua, win=None, min_trials=10):
    # sua - n_tr X T
    N, T = sua.shape
    ff = np.zeros(T)
    for t in range(T):
        if len(np.where(~np.isnan(np.sum(sua[:, t: np.min([t + win, T])], axis=-1)))[0]) > min_trials:
            dat = [np.sum(sua[trial, t: np.min([t + win, T])], axis=-1) for trial in range(N) if
                   ~np.isnan(np.sum(sua[trial, t: np.min([t + win, T])], axis=-1))]
            ff[t] = (np.nanstd(dat) ** 2) / np.nanmean(dat)
        else:
            ff[t] = np.nan
    return ff


def shared_variance(spike_count_mat, n_comp=None, n_folds=3):
    [n_times, n_units] = spike_count_mat.shape
    if not n_comp:
        kf = KFold(n_splits=n_folds)
        comp_range = range(0, int(n_units / 2), 4)
        log_likelihood = np.zeros((len(comp_range), n_folds))
        for f, (train_index, test_index) in enumerate(kf.split(spike_count_mat)):
            for c, comp in enumerate(comp_range):
                transformer = FactorAnalysis(n_components=comp, random_state=0, svd_method='lapack')
                transformer.fit(spike_count_mat[train_index])
                log_likelihood[c, f] = transformer.score(spike_count_mat[test_index])
        n_comp = comp_range[np.argmax(np.mean(log_likelihood, axis=-1))]

    transformer = FactorAnalysis(n_components=n_comp, random_state=0, svd_method='lapack')
    transformer.fit(spike_count_mat)
    shared_cov = np.dot(transformer.components_.T, transformer.components_)
    independent_cov = transformer.noise_variance_
    shared_units = []
    for unit in range(n_units):
        perc = 100 * np.dot(shared_cov[unit], shared_cov[unit]) / (
                np.dot(shared_cov[unit], shared_cov[unit]) + independent_cov[unit])
        shared_units.append(perc)
    return np.array(shared_units)


def coefficient_of_variation(sua, state_ranges=None, Fs_spikes=1250, Fs=30, t_lim=None):
    if not state_ranges:
        isi = np.diff(np.where(sua)[0]) / Fs_spikes
        return np.nanstd(isi) / np.nanmean(isi)
    else:
        isi = [np.diff(
            np.where(sua[int(state_ranges[i][0] * Fs_spikes / Fs):int(state_ranges[i][1] * Fs_spikes / Fs)] > 0)[
                0]) / Fs_spikes for i
               in range(len(state_ranges))]
        isi = np.concatenate(isi)
        isi = isi[isi <= t_lim]
        return np.nanstd(isi) / np.nanmean(isi)



def encoding_patterns():

    try:
        df_fullmodel = pd.read_csv('data/HMM_GLM/fullmodel_r2_qc1.csv')
        df_stim = pd.read_csv('data/HMM_GLM/stim_r2_qc1.csv')
        df_behavior = pd.read_csv('data/HMM_GLM/behavior_r2_qc1.csv')
        df_lfp = pd.read_csv('data/HMM_GLM/lfp_r2_qc1.csv')
        df_pop = pd.read_csv('data/HMM_GLM/pop_r2_qc1.csv')
    except FileNotFoundError:
        df_fullmodel = model_details('full')
        df_stim  = model_details('stim')
        df_behavior = model_details('behavior')
        df_lfp = model_details('lfp')
        df_pop = model_details('pop')

    # Step 1: Select the specific columns from each dataframe
    df_stim_filtered = df_stim[['unit', 'probe', 'session_id', 'SH_r2', 'SI_r2', 'SL_r2', 'final_r2']]
    df_behavior_filtered = df_behavior[['unit', 'probe', 'session_id', 'SH_r2', 'SI_r2', 'SL_r2', 'final_r2']]
    df_lfp_filtered = df_lfp[['unit', 'probe', 'session_id', 'SH_r2', 'SI_r2', 'SL_r2', 'final_r2']]
    df_pop_filtered = df_pop[['unit', 'probe', 'session_id', 'SH_r2', 'SI_r2', 'SL_r2', 'final_r2']]

    # Step 2: Sequentially merge each filtered dataframe with df_fullmodel
    df_summary = pd.merge(df_fullmodel, df_stim_filtered, on=['unit', 'probe', 'session_id'], how='left',
                          suffixes=('', '_stim'))
    df_summary = pd.merge(df_summary, df_behavior_filtered, on=['unit', 'probe', 'session_id'], how='left',
                          suffixes=('', '_behavior'))
    df_summary = pd.merge(df_summary, df_lfp_filtered, on=['unit', 'probe', 'session_id'], how='left', suffixes=('', '_lfp'))
    df_summary = pd.merge(df_summary, df_pop_filtered, on=['unit', 'probe', 'session_id'], how='left', suffixes=('', '_pop'))


    try:
        df_summary['n_sources'] = [len(
            np.where(row[['behavior', 'same\narea LFPs', 'other area\npopulation activity', 'stimulus']].values > 0.1)[
                0]) for pos, row in df_summary.iterrows()]
    except:
        df_summary = df_summary.rename(columns={'final_r2_behavior': 'behavior', 'final_r2_stim': 'stimulus',
                                                'final_r2_lfp': 'same\narea LFPs',
                                                'final_r2_pop': 'other area\npopulation activity'})
        df_summary['n_sources'] = [len(
            np.where(row[['behavior', 'same\narea LFPs', 'other area\npopulation activity', 'stimulus']].values > 0.1)[
                0]) for pos, row in df_summary.iterrows()]
    return df_summary


def model_details(model_name):

    def make_r2_df(r2_state, r2_final, num_units):
        unit_df = pd.DataFrame()
        for p_no, probe in enumerate(probes):
            tdf = pd.DataFrame()
            tdf['SH_r2'] = r2_state[p_no, :num_units[probe], 0]
            tdf['SI_r2'] = r2_state[p_no, :num_units[probe], 1]
            tdf['SL_r2'] = r2_state[p_no, :num_units[probe], 2]
            tdf['final_r2'] = r2_final[p_no, :num_units[probe]]
            tdf['probe'] = probe
            tdf['unit'] = np.arange(num_units[probe])
            unit_df = unit_df.append(tdf)
        return unit_df

    session_ids = du.get_session_ids()
    probes = ['probeC', 'probeD', 'probeF', 'probeE', 'probeB', 'probeA']
    unit_table = pd.read_csv('data/unit_table.csv')
    stim = 'natural_movie_one_more_repeats'
    df_model = pd.DataFrame()
    for session_id in session_ids:
        unit_df = unit_metrics(session_id, unit_table, probes)
        num_units = {probe: len(unit_df[unit_df.probe == probe]) for probe in probes}

        # encoding information
        try:
            if model_name != 'stim':
                r2_state = np.load('data/HMM_predictor/' + model_name + '_model/single_neuron_model/r2_3s_nrn_' + str(session_id) + '_' + stim + '_1.npy')
                r2_final = np.load('data/HMM_predictor/' + model_name + '_model/single_neuron_model/r2_3s_final_nrn_' + str(session_id) + '_' + stim + '_1.npy')
            else:
                r2_state = np.load('data/HMM_predictor/' + model_name + '_model/single_neuron_model/r2_3s_nrn_' + str(session_id) + '_' + stim + '_stim.npy')
                r2_final = np.load('data/HMM_predictor/' + model_name + '_model/single_neuron_model/r2_3s_final_nrn_' + str(session_id) + '_' + stim + '_stim.npy')
            r2_df = make_r2_df(r2_state, r2_final, num_units)
            r2_df['session_id'] = session_id
        except Exception as e:
            print(e)
            continue

        if model_name == 'full':
            unit_ids = unit_df.unit_id.values
            analyses_df = analysis_metrics(unit_ids)
            df_merged = pd.merge(unit_df, analyses_df, on=['unit', 'probe', 'session_id', 'unit_id'], how='left')
            df_merged = pd.merge(df_merged, r2_df, on=['unit', 'probe', 'session_id'], how='left')
        else:
            df_merged = pd.merge(unit_df, r2_df, on=['unit', 'probe', 'session_id'], how='left')

        # QC
        df_merged_qc = df_merged[(df_merged.total_rate > qc_val('rate_threshold'))
                                 & (df_merged.presence_ratio > qc_val('presence_ratio'))
                                 & (df_merged.amplitude_cutoff < qc_val('amplitude_cutoff'))
                                 & (df_merged.isi_violations < qc_val('isi_violations'))
                                 & (df_merged.rf_x < qc_val('RF_max'))
                                 & (df_merged.rf_x > qc_val('RF_min'))
                                 & (df_merged.rf_y < qc_val('RF_max'))
                                 & (df_merged.rf_y > qc_val('RF_min'))
                                 & (df_merged.area != 'VISli')]

        df_model = df_model.append(df_merged_qc)
    return df_model


