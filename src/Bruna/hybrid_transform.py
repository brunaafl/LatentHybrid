import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from skorch.dataset import ValidSplit, unpack_data
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, GradientNormClipping, Checkpoint, WandbLogger
from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.utils import to_tensor, to_numpy, to_device

import numpy as np
import pandas as pd

from braindecode.models import EEGNetv4, Deep4Net, ShallowFBCSPNet
from braindecode import EEGClassifier
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

from sklearn.model_selection import (
    LeaveOneGroupOut,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time

from copy import deepcopy

from mne.epochs import BaseEpochs
import mne

import pdb

# from torchviz import make_dot

from train import define_clf

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA

from dataset import split_runs_EA

import wandb

from torch.nn import init

from braindecode.augmentation import AugmentedDataLoader, GaussianNoise

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter


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
        if self.use_EA:
            X = split_runs_EA(X.get_data(), self.EA_len_run)
        subjects = {i: [] for i in np.unique(self.groups)}
        print(f"(1) EA {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        for index, trial in enumerate(X):
            subjects[self.groups[index]].append((trial, self.labels[index]))

        print(f"(2) Split {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        sizes = [len(subjects[i]) for i in subjects]

        n_subjects = len(subjects)
        n_trials_per_subject = min(
            np.unique([len(subjects[i]) for i in subjects]))  # len(subjects[list(subjects.keys())[0]])
        self.n_trials_used = n_trials_per_subject
        ch_names = [self.info["ch_names"][i] + f"_s{k}" for i in range(len(self.info["ch_names"])) for k in
                    subjects.keys()]

        print(f"(3) Setup {(time() - initial_time) * 1000}ms | {(time() - initial_time)}s")

        new_trials = []
        for trial_i in range(n_trials_per_subject):
            trial = []
            target = []
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
