import mne

from time import time
import numpy as np
import pandas as pd

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from dataset import split_runs_EA


class HybridAggregateTransform(BaseEstimator, TransformerMixin):
    def __init__(self, EA_len_run=None, kw_args=None):
        self.kw_args = kw_args
        self.use_EA = EA_len_run != None
        self.EA_len_run = EA_len_run
        self.n_trials_used = 0

    def fit(self, X, y=None, subject_groups=None, info=None, labels=None):
        self.labels = labels
        self.groups = subject_groups
        self.info = info
        return self

    def transform(self, X, y=None):
        initial_time = time()

        # If EA is required
        if self.use_EA:
            X = split_runs_EA(X.get_data(), self.EA_len_run)
        print(f"(1) EA {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        # Create dict mapping each individual to their labeled trials
        subjects = {i: [] for i in np.unique(self.groups)}
        for index, trial in enumerate(X):
            subjects[self.groups[index]].append((trial, self.labels[index]))

        print(f"(2) Split {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        # Get number of trials per subject
        n_trials_per_subject = min(np.unique([len(subjects[i]) for i in subjects]))
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

            # Create new trials such that one trial corresponds of one trial of each subject
            for subject in subjects:
                trial.append(subjects[subject][trial_i][0])
                target.append(subjects[subject][trial_i][1])
            info = mne.create_info(ch_names=ch_names, sfreq=self.info["sfreq"])
            raw = mne.io.RawArray(np.vstack(trial) * 1e6, info)
            base_dataset = BaseDataset(raw, pd.Series({"target": np.array(target)}), target_name="target")
            new_trials.append(base_dataset)

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
