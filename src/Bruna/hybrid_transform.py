import mne

from time import time
import numpy as np
import pandas as pd
import torch

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

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
        X_data = X.get_data()
        if len(np.unique(self.groups)) > 1:
            X_aux = []
            labels_aux = []
            groups_aux = []
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
            self.labels = labels_aux
            self.groups = groups_aux
        else:
            n = X_data.shape[0]
            q = n // 24
            d = int(q * 24)

            X_aux = X_data[:d, :, :]
            labels_aux = self.labels[:d]
            groups_aux = self.groups[:d]
            self.labels = labels_aux
            self.groups = groups_aux

        return X_aux

    def sample_data(self, X):

        sample = self.sample
        X_aux = []
        m = self.EA_len_run
        n = X.shape[0]
        n_samples = int(m * sample)

        for k in range(int(n / m)):
            run = X[k * m:(k + 1) * m]
            idx = np.random.randint(0, m, n_samples)
            X_aux.append(run[idx])
        X_EA = np.concatenate(X_aux)
        return X_EA

    def transform(self, X, y=None):
        initial_time = time()

        if self.data_code == 'Schirrmeister2017':
            X_aux = self.sample_Schirrmeister(X)

        else:
            X_aux = X.get_data()

        if self.sample is not None:
            if type(self.sample)==float:
                X_aux, labels_aux, groups_aux = self.sample_data(X_aux)
                self.labels = labels_aux
                self.groups = groups_aux

                # update the size of the run
                self.EA_len_run = int(self.EA_len_run * self.sample)

        # If EA is required
        if self.use_EA:
            X = split_runs_EA(X_aux, self.EA_len_run)
        else:
            X = X_aux * 1e6
        print(f"(1) EA {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")
        #print(X.shape)
        # Create dict mapping each individual to their labeled trials
        subjects = {i: [] for i in np.unique(self.groups)}
        for index, trial in enumerate(X):
            subjects[self.groups[index]].append((trial, self.labels[index]))

        print("Subjects ", subjects.keys())
        print("Number of trials ", len(subjects[list(subjects.keys())[0]]))

        print(f"(2) Split {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        # Get number of trials per subject
        n_trials_per_subject = min(np.unique([len(subjects[i]) for i in subjects]))

        print(f'N trials per subject: {n_trials_per_subject}')
        self.n_trials_used = n_trials_per_subject

        # Get channel names
        ch_names = [self.info["ch_names"][i] + f"_s{k}" for i in range(len(self.info["ch_names"])) for k in
                    subjects.keys()]

        print(f"(3) Setup {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        new_trials = []
        # Reorder trial
        for trial_i in range(n_trials_per_subject):
            trial = []
            target = []

            # TODO: Shuffle data
            for subject in subjects:
                trial.append(subjects[subject][trial_i][0])
                target.append(subjects[subject][trial_i][1])
            info = mne.create_info(ch_names=ch_names, sfreq=self.info["sfreq"])
            raw = mne.io.RawArray(np.vstack(trial), info)
            base_dataset = BaseDataset(raw, pd.Series({"target": np.array(target)}), target_name="target")
            new_trials.append(base_dataset)
        print(f"(4) Process {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        print('Number of trials ',len(new_trials))
        print('Len labels ', len(self.labels))
        print('Len groups ',len(self.groups))
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
