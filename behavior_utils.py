import sys
import numpy as np
from sklearn.impute import SimpleImputer
import data_utils as du
import cv2
import pandas as pd
from glob import glob
import os
from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import h5py

data_dir = 'D:/ecephys__project_cache/'
code_dir = 'C:/Users/shailaja.akella/Dropbox Personal)/AI_neural_variability/code/python codes/'


def pupil_area(session, stim, trials=[]):
    Fs_pupil = 30
    if len(trials) == 0:
        trials = du.trials_table(session, stim)
    duration = int(trials.duration.mean())
    presentations = len(trials)
    pupil_data = session.get_screen_gaze_data()
    pupil_size = np.zeros([presentations, int(duration * Fs_pupil)])
    pupil_times = np.zeros([presentations, int(duration * Fs_pupil)])
    scale_pupil = StandardScaler()
    try:
        scale_pupil.fit(pupil_data['raw_pupil_area'].values.reshape(-1, 1))
    except TypeError:
        print('No pupil data')

    for trial, (ind, row) in enumerate(trials.iterrows()):
        start = row['Start']
        end = start + row['duration']
        try:
            mask = (pupil_data.index.values >= start) \
                   & (pupil_data.index.values < end)
            L = np.min([len(np.where(mask)[0]), int(duration * Fs_pupil)])
            pupil_size[trial, :L] = pupil_data[mask].raw_pupil_area.values[:int(duration * Fs_pupil)]
            pupil_times[trial, :L] = pupil_data[mask].index.values[:int(duration * Fs_pupil)]
            pupil_times[trial, :L] = pupil_times[trial, :L] - pupil_times[trial, 0]
        except AttributeError:
            continue

    # impute pupil size to remove nans
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_pupil_size = imputer.fit_transform(scale_pupil.fit_transform(pupil_size))

    return {'values': imputed_pupil_size, 'times': pupil_times}


def running(session, stim, trials=[]):
    Fs_running = 60
    if len(trials) == 0:
        trials = du.trials_table(session, stim)
    duration = int(trials.duration.mean())
    presentations = len(trials)

    speeds = np.zeros([presentations, int(duration * Fs_running)])
    speed_times = np.zeros([presentations, int(duration * Fs_running)])
    scale_running = StandardScaler()
    scale_running.fit(session.running_speed['velocity'].values.reshape(-1, 1))

    for trial, (ind, row) in enumerate(trials.iterrows()):
        start = row['Start']
        end = start + row['duration']

        mask = (session.running_speed['start_time'].values >= start) \
               & (session.running_speed['start_time'].values < end)
        L = len(np.where(mask)[0])
        speeds[trial, :L] = session.running_speed[mask]['velocity']
        speed_times[trial, :L] = session.running_speed[mask]['start_time']
        speed_times[trial, :L] = speed_times[trial, :L] - speed_times[trial, 0]

    # impute running to remove nans
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_speeds = imputer.fit_transform(scale_running.fit_transform(speeds))

    return {'values': imputed_speeds, 'times': speed_times}


def bin_behavior(data, Fs, duration):
    n_tr = data['times'].shape[0]
    Tm = duration * Fs
    win = 1 / Fs

    binned = np.zeros([n_tr, Fs * duration])
    for t in range(Tm):
        for trial in range(n_tr):
            mask = np.where(
                (data['times'][trial, :] >= 0 + t * win) & (data['times'][trial, :] <= win + t * win))
            if data['values'][trial, mask].shape[-1] > 0:
                binned[trial, t] = np.nanmean(data['values'][trial, mask], axis=1)
            else:
                binned[trial, t] = binned[trial, t - 1]
    return binned


class face_motion:
    def __init__(self, session_id, vid_path=None, crop_dims=[], svd=False, n_comp=None,
                 stim='natural_movie_one_more_repeats'):
        self.session_id = session_id
        self.vid_path = vid_path
        self.crop_dims = crop_dims
        self.stim = stim
        self.svd = svd
        self.n_comp = n_comp
        self.start = None
        self.stop = None
        self.face_motion = None

    def vid_to_array(self):
        vid = cv2.VideoCapture(self.vid_path)
        all_frames = []
        check = True
        x1, x2, y1, y2 = self.crop_dims
        for trial_start, trial_end in zip(self.start, self.stop):
            vid.set(cv2.CAP_PROP_POS_FRAMES, trial_start)
            i = trial_start
            frames = []
            while check:
                check, arr = vid.read()
                if arr is not None:
                    frames.append(arr[x1:x2, y1:y2, 0])
                    i += 1
                if i == trial_end:
                    frames = np.array(frames)
                    all_frames.append(frames)
                    break
        all_frames = np.array(all_frames)  # trials X time X pixels1 X pixels2
        return all_frames

    def face_vid_for_stim(self):
        Fs_vid = 30
        session = du.get_nwb_session(self.session_id)
        trials = du.trials_table(session, self.stim)
        frame_times = du.get_frame_times_for_video(self.session_id)

        # stim times
        self.start = [np.argmin(np.abs(x - frame_times)) for x in trials.Start.values]
        duration = np.round(np.mean(trials.duration.values)) * Fs_vid
        self.stop = self.start + duration

        # video
        if not self.vid_path:
            self.vid_path = glob(os.path.join(data_dir + '/session_' + str(self.session_id), '*behavior.avi'))[0]

        if len(self.crop_dims) == 0:
            self.crop_dims = [180, 250, 275, 350]

        return self.vid_to_array()

    def fit(self):

        assert self.session_id is not None, "provide session id"

        face_motion_path = data_dir + '/session_' + str(self.session_id) + '_VarMat/' \
                           + self.stim + '_total_face_motion.npy'
        if os.path.exists(face_motion_path):
            self.face_motion = np.load(face_motion_path)
            return self

        vid_mat = self.face_vid_for_stim()
        n_trials, trial_length, _, _ = vid_mat.shape

        # absolute face motion energy
        vid_rearr = rearrange(vid_mat, 'c p b h -> c p (b h)')
        diff_mat = np.abs(np.diff(vid_rearr, axis=1))
        diff_mat = rearrange(diff_mat, 'c b h -> (c b) h')

        if self.svd:
            # PCA of absolute face motion energy
            self.n_comp = self.n_comp if self.n_comp else 5
            pca = PCA(n_components=self.n_comp)
            x = StandardScaler().fit_transform(diff_mat)
            pca_diff_mat = pca.fit_transform(x)

            pca_diff_mat = np.array([savgol_filter(x, 31, 3) for x in pca_diff_mat.T]).T
            pca_diff_mat = np.append(pca_diff_mat.reshape(n_trials, trial_length - 1, self.n_comp),
                                     np.zeros((n_trials, 1, self.n_comp)), axis=1)
            self.face_motion = pca_diff_mat.reshape(-1, self.n_comp)
            return self

        else:
            # total absolute face motion energy
            std_sum = StandardScaler().fit_transform(np.sum(diff_mat, axis=1).reshape(-1, 1))
            std_sum = np.append(std_sum.reshape(n_trials, trial_length - 1), np.zeros((n_trials, 1)), axis=1)
            self.face_motion = savgol_filter(std_sum.reshape(1, -1), 31, 3).reshape(n_trials, trial_length)
            return self


def pose_tracking_features(session_id, stim):

    nodes = ['body_center', 'forelimb1', 'hindlimb1', 'hindlimb2', 'tail_start', 'tail_end']

    files = glob(data_dir + '/session_' + str(session_id) + '_VarMat/*_diff.npy')
    if len(files) > 0:
        df = pd.DataFrame(columns = nodes)
        for n, node in enumerate(nodes):
            df[node] = np.load(glob(data_dir + '/session_' + str(session_id) +
                               '_VarMat/*' + node + '_diff.npy')[0])
        return df

    with h5py.File(code_dir + '/behavior/sleap_models/labels_' +
                   str(session_id) + '_' + stim + '.v001.analysis.h5', 'r') as f:
        tracks_matrix = np.squeeze(f['tracks'][:])

    df = pd.DataFrame()
    for i, node in enumerate(nodes):
        node_mat = np.sqrt((tracks_matrix[0, i, 1:] - tracks_matrix[0, i, :-1]) ** 2 +
                       (tracks_matrix[1, i, 1:] - tracks_matrix[1, i, :-1]) ** 2)
        df[node] = np.concatenate([node_mat, [0]], axis=0)

    df = df.interpolate('pad')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    Hz = 30
    w = 0.05 * Hz
    filt = (1 / np.sqrt(2 * np.pi * w ** 2)) * np.exp(
        -((np.arange(-0.5 * Hz, 0.5 * Hz)) ** 2) / (2 * w ** 2))
    for i, node in enumerate(nodes):
        x = df[node].values
        x = imp_mean.fit_transform(x.reshape(-1, 1)).reshape(-1)
        x = np.convolve(x, filt, 'same')
        x = x / np.max(x)
        df[node] = x

    return df


def MI_behavior_states(session_id, stim):

    import behavior_utils as bu

    binned_mvmts = pose_tracking_features(session_id, stim)
    nodes = pose_tracking_features(session_id, stim).keys()
    binned_mvmts = np.nan_to_num(binned_mvmts.values)
    states = np.load('../data/states/states_' + str(session_id) + '.npy')
    [n_trials, trial_length] = states.shape

    binned_mvmts = binned_mvmts.reshape(-1, n_trials, trial_length)

    nodes += ['pupil size', 'running', 'face motion']

    # trials
    session = du.get_nwb_session(session_id)
    trials = du.trials_table(session, stim)

    # running, pupil, face motion
    Fs = 30
    duration = int(trials.duration.mean())
    pupil_data = pupil_area(session, stim, trials)
    running_data = running(session, stim, trials)
    binned_running_speed = bin_behavior(running_data, Fs, duration)
    binned_pupil_size = bin_behavior(pupil_data, Fs, duration)
    face_motion = bu.face_motion(session_id=session_id).fit().face_motion.reshape(n_trials, trial_length)

    behavior = np.concatenate((binned_mvmts, face_motion, binned_pupil_size, binned_running_speed))
    states = states.reshape(-1)
    all_trial = {node: [] for node in nodes}

    for trial in range(n_trials):
        for n, node in enumerate(nodes):
            if np.nansum(behavior[n, trial]) > 0:
                all_trial[node].append(du.mutual_information().MI(states[trial], binned_mvmts[n, trial]))

    return np.array([np.nanmean(all_trial[node]) for node in nodes]), nodes
