import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import data_utils as du
import units_utils as uu
import behavior_utils as bu
import lfp_utils as lu

from sklearn.linear_model import Ridge
from scipy.linalg import hankel
import pandas as pd
from sklearn.model_selection import train_test_split
from Ridge_GLM import Ridge_GLM


data_dir = 'D:/ecephys__project_cache'
Fs_spikes = Fs_lfp = 1250
Fs = 30


class HMM_predictor:
    def __init__(self, session_id=None, stim=None, num_states=3,
                 model_type=None, n_folds=5, n_folds_tuning=3, probes=None, tau=None):
        self.session_id = session_id
        self.model_type = model_type
        self.stim = stim
        self.tau = tau * Fs
        self.n_folds = n_folds
        self.n_folds_tuning = n_folds_tuning
        self.num_states = num_states

        self.probes = ['probeC', 'probeD', 'probeF', 'probeE', 'probeB', 'probeA']
        self.probes_to_run = probes if not probes else self.probes
        self.prbs = None
        self.session = None
        self.trials = None
        self.duration = None
        self.n_trials = None
        self.trial_length = None

        self.r2 = None
        self.r2_final = None
        self.weights = None

    def time_embed(self, x):
        padded_x = np.hstack(
            (np.zeros(self.tau - 1), x.reshape(-1)))  # pad early bins of stimulus with zero
        shifted_x = hankel(padded_x[:-self.tau + 1], x.reshape(-1)[-self.tau:])
        return shifted_x.reshape(1, shifted_x.shape[0], shifted_x.shape[1])

    def neural_activity(self):
        spikes, pop_av = {probe: [] for probe in self.probes_to_run}, {probe: [] for probe in self.probes_to_run}
        spike_rate = {probe: [] for probe in self.probes_to_run}
        w = 0.05 * Fs
        filt = (1 / np.sqrt(2 * np.pi * w ** 2)) * np.exp(
            -((np.arange(-0.5 * Fs, 0.5 * Fs)) ** 2) / (2 * w ** 2))
        for probe in spikes.keys():
            _, units, population = uu.neural_activity(self.session_id, probe, self.stim)
            spikes[probe] = uu.bin_spikes(units, Fs_spikes, Fs)
            spike_rate[probe] = np.array(
                [np.convolve(x.reshape(-1), filt, 'same').reshape(self.n_trials, spike_rate[probe].shape[-1]) for x in
                 spike_rate[probe]])
            pop_av[probe] = du.bin_data(population, Fs_spikes, Fs)

        return {probe: sua_mat for probe, sua_mat in spikes.items() if sua_mat}, \
               {probe: pop_mat for probe, pop_mat in pop_av.items() if pop_mat}, \
               {probe: rate_mat for probe, rate_mat in spike_rate.items() if rate_mat}

    def lfps(self):
        lfps, channel_ids = lu.lfps(self.session_id, self.stim)
        binned_lfps = {probe: [] for probe in lfps.keys() if len(lfps[probe]) > 0 and np.nansum(lfps.keys()) > 0}
        for probe in binned_lfps.keys():
            select_channels, _ = lu.get_layers2(self.session_id, probe, channel_ids[probe])
            if np.sum(select_channels) > 0:
                channels = np.concatenate([np.where(channel_ids[probe] == idx)[0] for idx in select_channels])
                binned_lfps[probe] = du.bin_data(lfps[probe][channels], Fs_lfp, Fs)
        return {probe: lfp_mat for probe, lfp_mat in binned_lfps.items() if lfp_mat}

    def other_area_activity(self, binned_pop_av):
        binned_neighbor_activity = {probe: [] for probe in self.probes_to_run}
        for probe in self.probes_to_run:
            binned_sub_pop_av = []
            for sub_probe in list(set(self.probes_to_run) - {probe}):
                if sub_probe not in self.prbs or np.nansum(binned_pop_av[sub_probe]) != 0:
                    binned_sub_pop_av.append(binned_pop_av[sub_probe])
            binned_neighbor_activity[probe] = np.array(np.squeeze(binned_sub_pop_av))
        return binned_neighbor_activity

    def X_and_y(self):
        # internal brain activity
        binned_spikes, binned_pop_av, binned_spike_rate = self.neural_activity()
        binned_lfps = self.lfps()
        self.prbs = set(binned_lfps.keys()).intersection(binned_spikes.keys())
        binned_neighbor_activity = self.other_area_activity(binned_pop_av)

        # behavior
        pupil_data = bu.pupil_area(self.session, self.stim, self.trials)
        running_data = bu.running(self.session, self.stim, self.trials)
        binned_running_speed = bu.bin_behavior(running_data, Fs, self.duration).reshape(-1)
        binned_pupil_size = bu.bin_behavior(pupil_data, Fs, self.duration).reshape(-1)
        face_motion = bu.face_motion(self.session_id).fit().face_motion
        binned_mvmts = np.nan_to_num(bu.pose_tracking_features(self.session_id, self.stim).values).T.reshape(-1,
                                                                                                             self.n_trials,
                                                                                                             self.trial_length)
        # stimulus
        movie_features = du.get_movie_features()

        design_matrices = {probe: [] for probe in self.probes_to_run}
        for probe in self.probes_to_run:
            if probe not in self.prbs:
                continue

            if np.sum(binned_pupil_size) != 0:
                # orthogonalize behavior features
                features_o1 = np.concatenate((binned_running_speed, binned_pupil_size, face_motion,
                                              binned_mvmts), axis=0)
                # orthogonalize state variables against other variables
                features_o2 = np.concatenate((binned_running_speed, binned_pupil_size, face_motion,
                                              binned_mvmts,
                                              movie_features.reshape(-1, self.n_trials * self.trial_length),
                                              binned_neighbor_activity[probe].reshape(-1,
                                                                                      self.n_trials * self.trial_length),
                                              binned_lfps[probe].reshape(-1, self.n_trials * self.trial_length)),
                                             axis=0)
            else:
                features_o1 = np.concatenate((binned_running_speed, face_motion,
                                              binned_mvmts), axis=0)
                features_o2 = np.concatenate((binned_running_speed, face_motion,
                                              binned_mvmts,
                                              movie_features.reshape(-1, self.n_trials * self.trial_length),
                                              binned_neighbor_activity[probe].reshape(-1,
                                                                                      self.n_trials * self.trial_length),
                                              binned_lfps[probe].reshape(-1, self.n_trials * self.trial_length)),
                                             axis=0)

            if self.model_type == 'single_neuron':
                features_o1 = StandardScaler().fit_transform(features_o1.T).T
                features_o2 = StandardScaler().fit_transform(features_o2.T).T

            a = features_o1.T
            q, _ = np.linalg.qr(a)
            features_o1 = q.T

            a = features_o2.T
            q, _ = np.linalg.qr(a)
            features_o2 = q.T

            features = np.concatenate((features_o1, movie_features.reshape(-1, self.n_trials * self.trial_length),
                                       features_o2[-8:]), axis=0)
            if np.sum(binned_pupil_size) == 0:
                features = np.concatenate((features[0].reshape(1, -1), binned_pupil_size, features[1:]),
                                          axis=0)

            if self.model_type == 'single_neuron':
                features = np.array([features[i] / np.max(features[i]) for i in range(features.shape[0])])

            time_shifted_features = np.zeros((self.n_trials * self.trial_length, features.shape[0] * self.tau))
            for i in range(features.shape[0]):
                time_shifted_features[:, i * self.tau:(i + 1) * self.tau] = self.time_embed(features[i])
            design_matrices[probe] = np.hstack((np.ones((self.n_trials * self.trial_length, 1)), time_shifted_features))

        if self.model_type == 'single_neuron':
            return design_matrices, binned_spikes, binned_spike_rate
        else:
            return design_matrices, binned_pop_av

    def train_test_sets(self, X, y, y_rate, states):
        sc1 = StandardScaler()
        K = self.n_folds
        kf = KFold(n_splits=K, shuffle=True, random_state=0)
        X_train_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}
        y_train_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}

        X_test_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}
        y_test_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}
        yr_test_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}

        for ns in range(self.num_states):
            X_ns = np.nan_to_num(X[np.where(states == ns)[0]])
            y_ns = np.nan_to_num(y[np.where(states == ns)[0]])
            if self.model_type == 'single_neuron':
                yr_ns = np.nan_to_num(y_rate[np.where(states == ns)[0]])
            else:
                yr_ns = y_ns
            for k, (train_index, test_index) in enumerate(kf.split(X_ns)):
                X_train_folds[ns, k], y_train_folds[ns, k] = sc1.fit_transform(X_ns[train_index]), y_ns[train_index]
                X_test_folds[ns, k], y_test_folds[ns, k] = sc1.transform(X_ns[test_index]), y_ns[test_index]
                if self.model_type == 'single_neuron':
                    yr_test_folds[ns, k] = yr_ns[test_index]

        if self.model_type == 'single_neuron':
            return X_train_folds, y_train_folds, X_test_folds, y_test_folds, yr_test_folds
        else:
            return X_train_folds, y_train_folds, X_test_folds, y_test_folds

    def population_model_tuning(self, X, y, states):
        # hyperparameter tuning
        kf_val = KFold(n_splits=self.n_folds_tuning, shuffle=True, random_state=0)
        final_alpha = np.zeros(self.num_states)
        for ns in range(self.num_states):
            X_ns = np.nan_to_num(X[np.where(states == ns)[0]])
            y_ns = np.nan_to_num(y[np.where(states == ns)[0]])
            X_val, _, y_val, _ = train_test_split(X_ns, y_ns, test_size=0.70, random_state=42)
            alphas = 10 ** np.arange(0, 5)
            r2_tuning = np.zeros((5, 3))
            for a, alpha in enumerate(alphas):
                for k, (train_index, test_index) in enumerate(kf_val.split(X_val)):
                    sc1 = StandardScaler()
                    X_val_train, y_val_train = sc1.fit_transform(X_val[train_index]), y_val[train_index]
                    X_val_test, y_val_test = sc1.transform(X_val[test_index]), y_val[test_index]
                    X_val_train[:, 0] = np.ones((X_val_train.shape[0]))
                    X_val_test[:, 0] = np.ones((X_val_test.shape[0]))
                    reg = Ridge(alpha=alpha, fit_intercept=True).fit(X_val_train, y_val_train)
                    r2_tuning[a, k] = reg.score(X_val_test, y_val_test)

            final_alpha[ns] = alphas[np.argmax(np.nanmean(r2_tuning, axis=1))]
        return final_alpha

    def population_model(self, design_matrices, binned_pop_av, states):
        # initialize
        r2 = np.ones((len(self.probes_to_run), self.num_states)) * np.nan
        r2_final = np.ones((len(self.probes_to_run))) * np.nan
        weights = np.ones((len(self.probes_to_run), self.num_states, design_matrices[self.prbs[0]].shape[1])) * np.nan
        sc1 = StandardScaler()

        for p, probe in enumerate(self.probes_to_run):
            if len(design_matrices[probe]) == 0:
                continue
            print(probe + ' :', end='')
            X = design_matrices[probe]
            y = binned_pop_av[probe].reshape(-1, 1)

            # hyperparameter tuning
            final_alpha = self.population_model_tuning(X, y, states)

            # cross validation sets
            K = self.n_folds
            X_train_folds, y_train_folds, X_test_folds, y_test_folds = self.train_test_sets(X, y, states)

            # fold initializations
            r2_final_fold = np.zeros(K)
            r2_state_fold = np.zeros((self.num_states, K))

            y_pred_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}
            for k in range(K):
                y_k, y_pk = [], []
                for ns in range(self.num_states):
                    reg = Ridge(alpha=final_alpha[ns]).fit(X_train_folds[ns, k], y_train_folds[ns, k])
                    r2_state_fold[ns, k] = reg.score(X_test_folds[ns, k], y_test_folds[ns, k])
                    y_pred_folds[ns, k] = reg.predict(X_test_folds[ns, k])
                    y_pk.append(y_pred_folds[ns, k])
                    y_k.append(y_test_folds[ns, k])
                r2_final_fold[k] = r2_score(np.concatenate(y_k), np.concatenate(y_pk))

            r2[p] = np.nanmean(r2_state_fold, axis=1)
            r2_final[p] = np.nanmean(r2_final_fold)

            # weights
            for ns in range(self.num_states):
                X_ns = np.nan_to_num(X[np.where(states == ns)[0]])
                y_ns = np.nan_to_num(y[np.where(states == ns)[0]])
                reg = Ridge(alpha=final_alpha[ns]).fit(sc1.fit_transform(X_ns), y_ns)
                weights[p, ns, :X_ns.shape[1]] = np.fliplr(reg.coef_)

        self.r2 = r2
        self.r2_final = r2_final
        self.weights = weights
        return self

    def single_neuron_model(self, design_matrices, binned_spikes, binned_spike_rate, states):
        r2 = np.ones((len(self.probes_to_run), 250, self.num_states)) * np.nan
        r2_final = np.ones((len(self.probes_to_run), 250)) * np.nan
        weights = np.ones((len(self.probes_to_run), 250, self.num_states, 812)) * np.nan

        for p, probe in enumerate(self.probes_to_run):
            if probe not in self.prbs:
                continue
            X = design_matrices[probe]
            n_units = binned_spikes[probe].shape[0]
            K = self.n_folds

            for unit in range(n_units):
                y = binned_spikes[probe][unit].reshape(-1, 1)
                if len(np.where(np.nanmean(y.reshape(self.n_trials, self.trial_length), axis=1) > 0)[
                           0]) < 0.7 * self.n_trials:
                    continue
                y_rate = binned_spike_rate[probe][unit].reshape(-1, 1)

                r2_final_fold = np.zeros(K)
                r2_state_fold = np.zeros((self.num_states, K))

                try:
                    X_train_folds, y_train_folds, X_test_folds, \
                    y_test_folds, yr_test_folds = self.train_test_sets(X, y, y_rate, states)

                    yr_pred_folds = {(ns, k): [] for ns in range(self.num_states) for k in range(K)}
                    final_alpha = np.zeros((self.num_states, K))
                    for k in range(K):  # cv outer
                        y_k, y_pk = [], []
                        for ns in range(self.num_states):
                            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
                            # define the model
                            model = Ridge_GLM(bin_sz=1 / Fs)
                            # define search space
                            distributions = dict(l=2 ** np.arange(3, 10, 2))
                            # define search
                            search = GridSearchCV(model, distributions, scoring='r2', cv=cv_inner, refit=True,
                                                  n_jobs=1)
                            # execute search
                            result = search.fit(X_train_folds[ns, k], y_train_folds[ns, k].reshape(-1))
                            # get the best performing model fit on the whole training set
                            best_model = result.best_estimator_
                            final_alpha[ns, k] = result.best_params_['l']
                            print(result.best_params_)

                            yr_pred_folds[ns, k] = best_model.predict(X_test_folds[ns, k])
                            r2_state_fold[ns, k] = r2_score(yr_test_folds[ns, k], yr_pred_folds[ns, k])

                            y_pk.append(yr_pred_folds[ns, k])
                            y_k.append(yr_test_folds[ns, k])

                        r2_final_fold[k] = r2_score(np.concatenate(y_k), np.concatenate(y_pk))

                    r2[p, unit, :] = np.nanmean(r2_state_fold, axis=1)
                    r2_final[p, unit] = np.nanmean(r2_final_fold)

                    for ns in range(self.num_states):
                        X_ns = np.nan_to_num(X[np.where(states == ns)[0]])
                        y_ns = np.nan_to_num(y[np.where(states == ns)[0]])
                        reg = Ridge_GLM(l=np.median(final_alpha[ns]), bin_sz=1 / Fs).fit(X_ns, y_ns.reshape(-1))
                        weights[p, unit, ns, :X_ns.shape[1]] = np.fliplr(reg.weights)
                except:
                    print('Error in prediction..')
                    r2[p, unit, :] = np.nan
                    r2_final[p, unit] = np.nan
                    weights[p, unit] = np.ones((self.num_states, 812)) * np.nan

        self.r2 = r2
        self.r2_final = r2_final
        self.weights = weights

        return self

    def fit(self, design_matrices=None, binned_y=None, binned_yrate=None):
        self.session = du.get_nwb_session(self.session_id)
        self.trials = du.trials_table(self.session, self.stim)
        self.duration = int(self.trials.duration.mean())

        # state
        states = np.load('../data/states/states_' + str(self.session_id) + '.npy')
        [self.n_trials, self.trial_length] = states.shape
        states = states.reshape(-1)

        if self.model_type == 'population':
            if not design_matrices:
                design_matrices, binned_y = self.X_and_y()
            self.population_model(design_matrices, binned_y, states)
        else:
            if not design_matrices:
                design_matrices, binned_y, binned_yrate = self.X_and_y()
            self.single_neuron_model(design_matrices, binned_y, binned_yrate, states)

        return self

