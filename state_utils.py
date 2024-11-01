import numpy as np
import data_utils as du
import ssm
from ssm.util import find_permutation
from einops import rearrange
from itertools import groupby, permutations, combinations_with_replacement


class HMM:
    def __init__(self, num_trials = None, trial_length = None, num_states=3, num_iters=100):
        self.num_states = num_states
        self.num_trials = num_trials
        self.trial_length = trial_length
        self.num_iters = num_iters
        self.num_samples = num_trials * trial_length
        self.obs_dim = None
        self.log_likelihoods = []
        self.states = []
        self.hmm = None
        self.state_definition = []
        self.top_eig = None

    def set_data_params(self, input_lfp):
        if not self.num_trials:
            self.num_trials = input_lfp.shape[1]
        if not self.trial_length:
            self.trial_length = input_lfp.shape[3]
        if not self.num_samples:
            self.num_samples = self.num_trials * self.trial_length
        if not self.obs_dim:
            self.obs_dim = input_lfp.shape[0]*input_lfp.shape[2]

    def rearrange_input(self, input_lfp):
        return rearrange(input_lfp, 'c t b m -> (t m) (c b)')

    def reassign_states(self, states, input_lfp):
        states = states.reshape(self.num_trials, self.trial_length)
        theta_to_gamma_ratio = np.zeros(self.num_states)
        theta, gamma = 0, 3
        for c in range(self.num_states):
            theta_mat, gamma_mat = input_lfp[:, :, theta, :], input_lfp[:, :, gamma, :]
            theta_to_gamma_ratio[c] = np.nanmean(theta_mat[:, states[:] == c]) / np.nanmean(gamma_mat[:, states[:] == c])
        cluster_pos = np.argsort(theta_to_gamma_ratio)
        reassigned_states = np.zeros(states.reshape(-1).shape)
        for n, pos in enumerate(cluster_pos):
            reassigned_states[states == pos] = n
        return reassigned_states

    def set_state_definition(self, input_lfp):
        state_def = np.zeros([input_lfp.shape[2], self.num_states])
        for band in range(input_lfp.shape[2]):
            mat = (input_lfp[:, :, band, :] - np.nanmean(input_lfp[:, :, band, :])) / np.nanstd(
                input_lfp[:, :, band, :])
            for num_state in range(self.num_states):
                state_def[band, num_state, 0] = np.nanmean(mat[:, self.states[:] == num_state])
        for band in range(input_lfp.shape[2]):
            state_def[band] = (state_def[band] - np.nanmean(state_def[band])) / np.nanstd(state_def[band])
        self.state_definition = state_def
        self.top_eig = np.linalg.eig(np.cov(np.transpose(self.state_definition)))

    def bic(self):
        p = self.num_states + self.num_states * (self.num_states - 1) + self.num_states * self.obs_dim * (
                    self.num_states + self.obs_dim) / 2
        return -2 * self.log_likelihoods[-1] + p * np.log(self.num_samples)

    def fit(self, input_lfp):
        self.set_data_params(input_lfp)
        input_lfp_2D = self.rearrange_input(input_lfp)
        np.random.seed(seed=42)
        hmm = ssm.HMM(self.num_states, self.obs_dim, observations="gaussian")
        hmm_lls = hmm.fit(input_lfp_2D, method="em", num_em_iters=self.num_iters)

        states_init = hmm.most_likely_states(input_lfp_2D).reshape(self.num_trials, self.trial_length)
        self.states = self.reassign_states(states_init, input_lfp).reshape(self.num_trials, self.trial_length)
        self.log_likelihoods = hmm.log_likelihood(input_lfp_2D)

        hmm.permute(find_permutation(self.states.astype(int), states_init.astype(int)))
        self.states = hmm.most_likely_states(input_lfp_2D).reshape(self.num_trials, self.trial_length)
        self.set_state_definition(input_lfp)

        self.hmm = hmm

        return self


def state_dwell_times(states, Fs):
    num_states = 3
    n_tr = states.shape[0]
    dwell_times = []
    for s_no in range(num_states):
        dwell_times_per_trial = []
        for trial in range(n_tr):
            r = du.find_ranges(np.where(states[trial] == s_no)[0])
            if len(r) > 0:
                intervals = np.concatenate([np.diff(r[i]) for i in range(len(r))])
                intervals[intervals == 0] = 1
                dwell_times_per_trial.append(intervals / Fs)
        dwell_times.append(np.concatenate(dwell_times_per_trial))
    return dwell_times


def sequence_probability(sessions, sequence_length=3):
    all_combs = np.concatenate([list(permutations(x)) for x in list(combinations_with_replacement([0, 1, 2], sequence_length))])
    probability = {tuple(x): [] for x in all_combs if not any(sum(1 for _ in g) > 1 for _, g in groupby(x))}
    for session in sessions:
        # no consecutive repeats
        p = {tuple(x): 0 for x in all_combs if not any(sum(1 for _ in g) > 1 for _, g in groupby(x))}
        states = np.load('../data/states/states_' + str(session) + '.npy').reshape(-1)
        rem_dupes = [v for i, v in enumerate(states) if i == 0 or v != states[i-1]]
        for t in range(len(rem_dupes) - sequence_length):
            seq = np.array(rem_dupes[t:t + sequence_length])
            p[tuple(seq)] += 1
        total = sum(p.values())
        for key in p.keys():
            probability[key].append(p[key]/total)
    return probability
