import os
import numpy as np
import pandas as pd
from einops import rearrange
import numpy.matlib
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import EcephysSyncDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from glob import glob
import math
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import umap
import random
from sklearn.cluster import MeanShift
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import tqdm

import matplotlib.pyplot as plt

data_dir = "D:/ecephys__project_cache/"


def get_probes_session(session_id):
    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    probes = cache.get_probes()
    return probes[probes.ecephys_session_id == session_id].name.values


def get_session_ids():
    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    session_ids = sessions[sessions.session_type == 'functional_connectivity'].index.values
    return session_ids[session_ids!=839557629]


def get_nwb_session(session_id, with_qc = True):
    session_directory = os.path.join(data_dir + 'session_' + str(session_id))
    if with_qc:
        return EcephysSession.from_nwb_path(os.path.join(session_directory, 'session_' + str(session_id) + '.nwb'))
    else:
        return EcephysSession.from_nwb_path(os.path.join(session_directory, 'session_' + str(session_id) + '.nwb'),
                                     api_kwargs={
                                         "amplitude_cutoff_maximum": np.inf,
                                         "presence_ratio_minimum": -np.inf,
                                         "isi_violations_maximum": np.inf
                                     })


def get_frame_times_for_video(session_id):
    sync_file = glob(os.path.join(data_dir + '/session_' + str(session_id), '*.sync'))[0]
    obj = EcephysSyncDataset.factory(sync_file)
    frame_times = obj.get_events_by_line(9, units='seconds')
    return frame_times[::2]


def trials_table(session, stim):
    trials = []
    if stim != 'invalid_presentation':
        trials = session.get_stimulus_table([stim])
        trials = trials.rename(columns={"start_time": "Start", "stop_time": "End"})
        if 'natural_movie' in stim:
            frame_times = trials.End - trials.Start
            # chunk each movie clip
            trials = trials[trials.frame == 0]
            duration = np.mean(np.diff(trials.Start.values)[:5])
            trials['duration'] = duration
            trials['End'] = trials['Start'] + duration
        elif stim == 'spontaneous':
            duration = trials.duration
            index = np.where(duration == max(duration))[0][0]
            if max(duration) > 20:  # only keep the longest spontaneous; has to be longer than 20 sec
                w = 20
                start = np.array(list(trials.iloc[index].Start + np.arange(5, max(duration) - w, w)))
                end = start + w
                trials = pd.DataFrame()
                trials['Start'] = start
                trials['End'] = end
                trials['duration'] = w
                duration = w
        else:
            duration = np.round(np.mean(trials.duration.values), 2)
            trials['duration'] = duration
    return trials


class mutual_information:
    def __init__(self, multiplier_X=None, multiplier_Y=None, alpha = 1.01):
        self.multiplier_X = multiplier_X
        self.multiplier_Y = multiplier_Y
        self.alpha = alpha
        self.mi = None
        self.Hx = None
        self.Hy = None
        self.Hxy = None

    def GaussMat(self, mat, mult=None):
        # Gram matrix

        # mat - intensity functions of size num_trials X Time
        if len(mat.shape) == 1:
            mat = mat.reshape(-1, 1)
        [N, T] = mat.shape

        if mult:
            # sig = (mult) * N ** (-1 / (4 + T))  # scott's rule
            sig = (mult) * (1.06 * np.nanstd(mat[:])) * (N ** (-1 / 5)) # silverman's rule
        else:
            # sig = N**(-1/(4+T)) # scott's rule
            sig = (1.06 * np.nanstd(mat[:])) * (N ** (-1 / 5))
        pairwise_sq_dists = squareform(pdist(mat, 'sqeuclidean'))
        return np.exp(-pairwise_sq_dists / sig ** 2)

    def alpha_entropy(self, mat, mult=None):
        # X - intensity functions of size num_trials X Time
        # alpha - order of entropy
        # Gram matrix - using Gaussian kernel
        # evaluates the distances between input samples after projection to RKHS

        N = len(mat)
        K = self.GaussMat(mat, mult=mult) / N

        # evaluation of the eigen spectrum
        L, _ = np.linalg.eig(K)
        absL = np.abs(L)

        # entropy estimation
        H = (1 / (1 - self.alpha)) * math.log2(np.min([np.sum(absL ** self.alpha), 0.9999]))

        return H

    def joint_alpha_entropy(self, X, Y):
        N = len(X)
        Kx = self.GaussMat(X, mult = self.multiplier_X) / N
        Ky = self.GaussMat(Y, mult = self.multiplier_Y) / N
        Kxy = Kx * Ky * N  # must be element wise multiplication

        Lxy, _ = np.linalg.eig(Kxy)
        absLxy = np.abs(Lxy)
        Hxy = (1 / (1 - self.alpha)) * math.log2(np.min([np.sum(absLxy ** self.alpha), 0.9999]))  # joint entropy

        return Hxy

    def MI(self, X, Y):
        # ref: Giraldo, Luis Gonzalo Sanchez, Murali Rao, and Jose C. Principe.
        # "Measures of entropy from data using infinitely divisible kernels."
        # IEEE Transactions on Information Theory 61.1 (2014): 535-548.
        # assumes IID samples

        try:
            self.Hy = self.alpha_entropy(Y,  mult = self.multiplier_Y)
            self.Hx = self.alpha_entropy(X,  mult = self.multiplier_X)
            self.Hxy = self.joint_alpha_entropy(X, Y)
            self.mi = (self.Hx + self.Hy - self.Hxy) / np.sqrt(self.Hx * self.Hy)
            return self
        except np.linalg.LinAlgError:
            print("need to adjust multiplier!")
            return self


def get_movie_features():
    path = r"C:\Users\shailaja.akella\Dropbox (Personal)\AI_neural_variability\code\python codes"

    # Intensity, contrast, kurtosis, energy
    scaler = StandardScaler()
    image_stats = np.load(path +'/movie_features/natural_movie_1_image_stats.npy')
    image_stats_scaled = np.squeeze(np.array([scaler.fit_transform(x.reshape(-1, 1)) for x in image_stats]))
    image_stats_scaled_trials = np.squeeze(np.array([numpy.matlib.repmat(x, 60, 1) for x in image_stats_scaled]))

    # energy
    energy = np.load(path +'/movie_features/energy.npy')
    energy_scaled = scaler.fit_transform(energy.reshape(-1, 1))
    energy_scaled_trials = np.array(numpy.matlib.repmat(energy_scaled.T, 60, 1)).reshape(1, 60, len(energy.T))

    # edge
    edge = np.load(path + '/movie_features/edginess.npy')
    edge_scaled = scaler.fit_transform(edge.reshape(-1, 1))
    edge_scaled_trials = np.array(numpy.matlib.repmat(edge_scaled.T, 60, 1)).reshape(1, 60, len(edge))

    return np.concatenate((image_stats_scaled_trials, energy_scaled_trials, edge_scaled_trials))


def bin_data(data, Fs_original, Fs):
    dims = data.shape
    bin_w = math.floor(Fs_original/Fs)
    Tm = int(Fs*dims[-1]/Fs_original)
    if len(dims) == 2:
        binned = [np.nanmean(data[:, 0 + t * bin_w: bin_w + t * bin_w], axis=-1) for t in range(Tm)]
        return np.array(binned).T
    if len(dims) == 3:
        binned = [np.nanmean(data[:,:,  0 + t * bin_w: bin_w + t * bin_w], axis=-1) for t in range(Tm)]
        return rearrange(np.array(binned), 'l b w -> b w l')


def consecutive(s):
    maxrun = -1
    maxend = 0
    rl = {}
    for x in s:
        run = rl[x] = rl.get(x-1, 0) + 1
        if run > maxrun:
            maxend, maxrun = x, run
    return np.arange(maxend-maxrun+1, maxend+1)


def plot_density(data, vmin = None, vmax = None, bandwidth = None):
    if not vmin and not vmax:
        x_pos = np.arange(np.min(data), np.max(data), 0.001)
    else:
        x_pos = np.arange(vmin, vmax, 0.001)
    if bandwidth:
        kernel = stats.gaussian_kde(data, bw_method = bandwidth)
        density = kernel(x_pos)
    else:
        kernel = stats.gaussian_kde(data)
        density = kernel(x_pos)
    return x_pos, density


def find_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def consensus_clustering(df_summary, n_boot=200, meanshift_bandwidth=2,
                         n_neighbors_interval=None, n_components_interval=None,
                         cluster_range=np.arange(2, 15),
                         make_plots=False, save_plot_location=False):

    # Consensus clustering - UMAP + MeanShift + random number of neighbors and components

    if n_components_interval is None:
        n_components_interval = [2, 5]
    if n_neighbors_interval is None:
        n_neighbors_interval = [5, 10]
    if not make_plots:
        save_plot_location = None

    np.random.seed(10)
    weights = df_summary['behavior', 'area LFPs', 'other area\npopulation activity', 'stimulus', 'n_sources'].copy()
    weights = StandardScaler().fit_transform(weights)

    n_units = weights.shape[0]
    labels = np.zeros((n_units, n_boot))

    for n in tqdm.tqdm(range(n_boot), desc='running repeats', total=n_boot, leave=True, dynamic_ncols=True):
        W_umap = umap.UMAP(n_neighbors=random.randint(n_neighbors_interval[0], n_neighbors_interval[1]),
                           n_components=np.random.randint(n_components_interval[0], n_components_interval[1]),
                           min_dist=np.random.uniform(0.02, 0.5)).fit_transform(weights)
        clusterer = MeanShift(bandwidth=meanshift_bandwidth).fit(W_umap)
        labels[:, n] = clusterer.labels_

    # Construct probability matrix
    matrix = np.zeros((n_units, n_units))
    dist = []
    for n in range(n_boot):
        tmp = np.atleast_2d(labels[:, n]).T * (1 / (np.atleast_2d(labels[:, n])))
        tmp[tmp != 1] = 0
        matrix += tmp
        matrix_previous = matrix - tmp
        dist.append(np.linalg.norm(matrix / float(n) - matrix_previous / float(n - 1)))
    dist = np.array(dist)
    prob_matrix = matrix / float(n_boot)

    # Linkage
    Lin = sch.linkage(prob_matrix, 'ward')

    # Compute and plot dendrogram.
    Z = sch.dendrogram(Lin, orientation='right')
    index = Z['leaves']
    D = prob_matrix[index, :]
    D = D[:, index]
    if make_plots:
        fig = plt.figure(figsize=(5, 3))
        axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
        axdendro.set_xticks([])
        axdendro.set_yticks([])

        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
        im = axmatrix.matshow(D, origin='lower', cmap='Purples', vmax=1, vmin=0)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # plot colorbar.
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        plt.colorbar(im, cax=axcolor)

        if save_plot_location:
            plt.savefig(save_plot_location + "/dendrogram.png", dpi=300)

    # evaluate silhouette score to select n_clusters.
    score = np.zeros(13)
    for n, nc in enumerate(cluster_range):
        labels = sch.fcluster(Lin, nc, criterion='maxclust')
        score[n] = silhouette_score(prob_matrix, labels)

    kneedle = KneeLocator(np.arange(2, 15), score, S=3, curve="concave", direction="increasing",
                          interp_method="polynomial")
    kneedle.plot_knee_normalized()

    kneedle.plot_knee(figsize=(3, 3), ylabel='silhouette score', xlabel='number of clusters')
    NC = kneedle.knee

    if NC == 0:
        NC = 14

    if save_plot_location:
        plt.savefig(save_plot_location + 'silhouette_scores.png', dpi=300)

    labels = sch.fcluster(Lin, NC, criterion='maxclust')

    return labels, df_summary, score, dist, prob_matrix