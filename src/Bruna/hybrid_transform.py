import mne

from time import time
import numpy as np
import pandas as pd

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

from sklearn.base import BaseEstimator, TransformerMixin

from alignment import split_runs_EA


class HybridAggregateTransform(BaseEstimator, TransformerMixin):
    def __init__(self, EA_len_run=None, data_code = None, kw_args=None, shuffle=False, sample=None):
        self.kw_args = kw_args
        self.use_EA = EA_len_run is not None
        self.EA_len_run = EA_len_run
        self.n_trials_used = 0
        self.shuffle = shuffle
        self.data_code = data_code
        self.sample = sample

    def fit(self, X, y=None, subject_groups=None, info=None, labels=None):
        self.labels = labels
        self.groups = subject_groups
        self.info = info
        return self

    def sample_Schirrmeister(self, X):
        """
        Standardizing the number of trials for all subjects.
        Problem: My method supposes qll subjects have the same number of trials, which should also be a multiple of
        the number of trials/run or number of trials for aligment

        Here, I'm fixing this issue for HGD
        """
        X_data = X.get_data()

        # If more than 1 subject
        if len(np.unique(self.groups)) > 1:
            X_aux = [];labels_aux = []; groups_aux = []
            for subj in np.unique(self.groups):
                X_subj = X_data[self.groups == subj]
                X_subj = X_subj[:408, :, :]
                y_subj = self.labels[self.groups == subj]
                y_subj = y_subj[:408]
                groups_subj = self.groups[self.groups == subj]
                groups_subj = groups_subj[:408]
                X_aux.append(X_subj)
                labels_aux.append(y_subj)
                groups_aux.append(groups_subj)
            X_aux = np.concatenate(X_aux)
            labels_aux = np.concatenate(labels_aux)
            groups_aux = np.concatenate(groups_aux)

        else:
            n = X_data.shape[0]
            q = n // 24
            d = int(q * 24)

            X_aux = X_data[:d, :, :]
            labels_aux = self.labels[:d]
            groups_aux = self.groups[:d]

        # Setting new label and subject indexing
        self.labels = labels_aux
        self.groups = groups_aux

        return X_aux

    def transform(self, X, y=None):
        initial_time = time()

        if self.data_code == 'Schirrmeister2017':
            X_aux = self.sample_Schirrmeister(X)
        else:
            X_aux = X.get_data()

        # If EA is required
        if self.use_EA:
            X = split_runs_EA(X_aux, self.EA_len_run)
        else:
            X = X_aux * 1e6

        print(f"(1) EA {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        # Create dict mapping each individual to their labeled trials
        subjects = {i: [] for i in np.unique(self.groups)}
        for index, trial in enumerate(X):
            subjects[self.groups[index]].append((trial, self.labels[index]))

        print(f"(2) Split {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        # Get number of trials per subject
        n_trials_per_subject = min(np.unique([len(subjects[i]) for i in subjects]))
        print('N trials used: ', n_trials_per_subject)
        self.n_trials_used = n_trials_per_subject

        # Get channel names
        ch_names = [self.info["ch_names"][i] + f"_s{k}" for i in range(len(self.info["ch_names"])) for k in
                    subjects.keys()]

        print(f"(3) Setup {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        new_trials = []
        # Reshape data format of each trial: (subjects, data)
        for trial_i in range(n_trials_per_subject):
            trial = []
            target = []

            for subject in subjects:
                trial.append(subjects[subject][trial_i][0])
                target.append(subjects[subject][trial_i][1])

            info = mne.create_info(ch_names=ch_names, sfreq=self.info["sfreq"])

            raw = mne.io.RawArray(np.vstack(trial), info)
            base_dataset = BaseDataset(raw, pd.Series({"target": np.array(target)}), target_name="target")
            new_trials.append(base_dataset)

        print('trial :', len(new_trials))

        print(f"(4) Process {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        dataset = BaseConcatDataset(new_trials)
        windows_dataset = create_fixed_length_windows(
            dataset,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=len(new_trials[0]),
            window_stride_samples=len(new_trials[0]),
            drop_last_window=False
        )

        print(f"Total {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")
        return windows_dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
