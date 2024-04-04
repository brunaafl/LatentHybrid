import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from skorch.dataset import ValidSplit, unpack_data
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, GradientNormClipping, Checkpoint, WandbLogger
from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.utils import to_tensor, to_numpy

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


class LazyLayerNorm(LazyModuleMixin, nn.LayerNorm):
    cls_to_become = nn.LayerNorm

    def __init__(self, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__(0, eps, elementwise_affine)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)


def initmod(module):
    init.xavier_uniform_(module.weight, gain=1)
    return module


def gen_slice_EEGNet_normtest(n_chans, n_classes, input_window_samples, config, start=0, end=19, drop_prob=0.5,
                              norm=nn.BatchNorm2d):
    temp_model = EEGNetv4(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == 0:
        return nn.Identity()

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    net = list(temp_model.children())[start:end]
    if norm != nn.BatchNorm2d:
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = norm()
    return nn.Sequential(*net)


def gen_slice_DeepNet(n_chans, n_classes, input_window_samples, config, start=0, end=29, drop_prob=0.5, remove_bn=False):
    temp_model = Deep4Net(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    if end == 0:
        return nn.Identity()

    if remove_bn:
        for i, module in enumerate(temp_model):
            if isinstance(temp_model[i], nn.BatchNorm2d):
                pass
            # temp_model[i] = nn.Identity()

    net = list(temp_model.children())[start:end]
    if not remove_bn:
        net.append(nn.ELU())

    return nn.Sequential(*net)


def gen_slice_EEGNet(n_chans, n_classes, input_window_samples, config, start=0, end=19, drop_prob=0.5, remove_bn=False,
                     norm=None):
    temp_model = EEGNetv4(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == 0:
        return nn.Identity()

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    net = list(temp_model.children())[start:end]
    if remove_bn:
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = nn.Identity()
    else:
        net.append(nn.LayerNorm(([1, 1002]), elementwise_affine=False))
    return nn.Sequential(*net)


def gen_slice_ShallowNet(n_chans, n_classes, input_window_samples, config, start=0, end=29, drop_prob=0.5,
                         remove_bn=False):
    temp_model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    if end == 0:
        return nn.Identity()

    if remove_bn:
        for i, module in enumerate(temp_model):
            if isinstance(temp_model[i], nn.BatchNorm2d):
                pass
            # temp_model[i] = nn.LayerNorm([977, 1])

    net = list(temp_model.children())[start:end]

    return nn.Sequential(*net)


model_gen = {
    "DeepNet": [gen_slice_DeepNet, 8, 9],
    "EEGNet": [gen_slice_EEGNet, 12, 13],  # 5, 6 / 12, 13
    "ShallowNet": [gen_slice_ShallowNet, 4, 4],
    "ShallowNetShared": [gen_slice_ShallowNet, 0, 0],
    "EEGNetShared": [gen_slice_EEGNet, 0, 0],
    "DeepNetShared": [gen_slice_DeepNet, 0, 0],
    "EEGNetNormTest": [gen_slice_EEGNet_normtest, 5, 6],
}

norms = {
    "Identity": nn.Identity,
    "BatchNorm2d": nn.LazyBatchNorm2d,
    "InstanceNorm2d": nn.LazyInstanceNorm2d,
    "LayerNorm": LazyLayerNorm,
}


class HybridModel(nn.Module):
    def __init__(self, num_models, model_type, n_chans, n_classes, input_window_samples, config=None, freeze=None,
                 args=None):
        super(HybridModel, self).__init__()
        self._args = (n_chans, n_classes, input_window_samples)
        self.config = config
        self.args = args
        self.num_models = num_models
        self.model_type = model_type
        self.shared_modules = model_gen[self.model_type][0](n_chans, n_classes, input_window_samples, config,
                                                            start=model_gen[self.model_type][2],
                                                            norm=norms[self.args.sharednorm])
        self.unique_modules = nn.ModuleList()
        self.freeze = freeze == "freeze"
        self.norm = nn.Identity()
        for model in range(num_models):
            self.unique_modules.append(self.init_unique_modules(*self._args))

    def init_unique_modules(self, n_chans, n_classes, input_window_samples):
        unique_head = model_gen[self.model_type][0](n_chans, n_classes, input_window_samples, self.config,
                                                    end=model_gen[self.model_type][1], norm=norms[self.args.uniquenorm])
        return unique_head

    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    def forward(self, x):
        inputs = self.split_input(x)
        out = []
        for i, model_input in enumerate(inputs):
            temp_unique = self.unique_modules[i](model_input)
            # pdb.set_trace()
            temp_norm = self.norm(temp_unique)
            temp_shared = self.shared_modules(temp_norm)
            out.append(temp_shared)
        result = torch.stack(out)
        if result.requires_grad:
            result.retain_grad()

        return result

    def generate_branch_model(self):
        new_layers = self.init_unique_modules(*self._args)
        cloned_layers = copy.deepcopy(self.shared_modules)
        norm_clone = copy.deepcopy(self.norm)
        if self.freeze:
            cloned_layers.requires_grad_(False)
        return SpecializedModel(new_layers, norm_clone, cloned_layers)


class SpecializedModel(nn.Module):
    def __init__(self, unique_modules, norm_clone, cloned_modules):
        super(SpecializedModel, self).__init__()
        self.shared_modules = cloned_modules
        self.norm = norm_clone
        self.unique_modules = unique_modules

    def forward(self, x):
        x = self.unique_modules(x)
        x = self.norm(x)
        x = self.shared_modules(x)
        return x

    def predict(self, X):
        return self.forward(X).argmax()


# Just testing EEGNetShared performance
class HybridClassifier(EEGClassifier):

    def get_loss(self, y_pred, y_true, *args, **kwargs):

        y_true = to_tensor(y_true, device=self.device)
        losses = []
        for subject in range(y_pred.shape[0]):
            subject_slice = torch.select(y_pred, 0, subject)
            if y_pred.requires_grad:
                subject_slice.retain_grad()
            loss = self.criterion_(subject_slice, y_true[:, subject])
            losses.append(loss)

        loss = sum(losses) / self.module.num_models

        # make_dot(y_pred, show_attrs=True, params=dict(self.module.named_parameters())).render("model", format="svg")
        return loss

    """
    def get_loss(self, y_pred, y_true, *args, **kwargs):

        y_true = to_tensor(y_true, device=self.device)
        y_pred_flat = []
        for subject in range(y_pred.shape[0]):
            subject_slice = torch.select(y_pred, 0, subject)
            if y_pred.requires_grad:
                subject_slice.retain_grad()
            y_pred_flat.append(subject_slice)
        y_pred_flat = torch.cat(y_pred_flat, dim=0)
        y_true_flat = torch.flatten(y_true)
        loss = self.criterion_(y_pred_flat, y_true_flat)

        # make_dot(y_pred, show_attrs=True, params=dict(self.module.named_parameters())).render("model", format="svg")
        return loss
    """


class HybridScoring(EpochScoring):
    def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
        self.y_preds_ = []
        self.y_trues_ = []
        for subject_i in range(net.module.num_models):
            self.y_preds_.append([])
        self.tag = False

    def on_batch_end(
            self, net, batch, y_pred, training, **kwargs):
        if not self.use_caching or training != self.on_train:
            return

        _X, y = unpack_data(batch)
        self.y_trues_.append(y)
        for subject_i in range(net.module.num_models):
            self.y_preds_[subject_i].append(torch.select(y_pred, 0, subject_i))

    def on_epoch_end(
            self,
            net,
            dataset_train,
            dataset_valid,
            **kwargs):
        X_test, y_test, y_pred = self.get_test_data(dataset_train, dataset_valid)

        unwrapped_y_pred = []
        for subject_i in range(net.module.num_models):
            unwrapped_y_pred.append(torch.vstack(y_pred[subject_i]))

        with _cache_net_forward_iter(net, self.use_caching, unwrapped_y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)


def get_subject_acc_scorer(subject):
    def scoring_for_subject_i(model, x, y_true):
        out = model.forward_iter()
        y_preds = [z for z in out]
        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        return accuracy_score(true_slice, predictions)

    return scoring_for_subject_i


def get_subject_loss_scorer(subject, criterion):
    def scoring_for_subject_i(model, x, y_true):
        out = model.forward_iter()
        y_preds = [z for z in out]
        true_slice = to_tensor(y_true[:, subject], device=model.device)
        loss = criterion(y_preds[subject], true_slice)
        return loss

    return scoring_for_subject_i


def average_acc_scoring(model, x, y_true):
    out = model.forward_iter()
    y_preds = [z for z in out]
    accuracies_per_subject = []
    for subject in range(len(y_preds)):
        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        accuracies_per_subject.append(accuracy_score(true_slice, predictions))
    return sum(accuracies_per_subject) / len(accuracies_per_subject)


def active_wandb(args, config, subject, train=True):
    wconfig = {
        "batch_size": config.train.batch_size,
        "exp_name": config.train.experiment_name,
        "model_type": args.model,
        "dataset": args.dataset,
        "freeze": args.freeze,
        "alignment": args.ea,
        "dropout": config.model.drop_prob,
        "lr": config.train.lr,
        "patience": config.train.patience,
        "n_epochs": config.train.n_epochs,
        "weight_decay": config.train.weight_decay,
        "subject": subject,
        "train": train,
        "uniquenorm": args.uniquenorm,
        "sharednorm": args.sharednorm,
    }

    run = wandb.init(
        project="Hybrid EEG",
        group=config.train.experiment_name,
        name=f"{subject}-Shared:{args.sharednorm}-Unique:{args.uniquenorm}",
        config=wconfig
    )
    return run


def define_hybrid_clf(model, config, experiment_name):
    """
    Transform the pytorch model into classifier object to be used in the training
    Parameters
    ----------
    model: pytorch model
    config: dict with the configuration parameters
    Returns
    -------
    clf: skorch classifier
    """
    weight_decay = config.train.weight_decay
    batch_size = config.train.batch_size
    lr = config.train.lr
    patience = config.train.patience
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lrscheduler = LRScheduler(policy='CosineAnnealingLR', T_max=config.train.n_epochs)

    # onecycle = LRScheduler(policy=OneCycleLR, max_lr=lr*25, steps_per_epoch=1, total_steps=config.train.n_epochs)

    scoring_callbacks = [HybridScoring(scoring=get_subject_acc_scorer(i), on_train=False, name=f'{i:02d}_valid_acc',
                                       lower_is_better=False) for i in range(model.num_models)]

    # scoring_callbacks = [HybridScoring(scoring=get_subject_acc_scorer(i), on_train=True, name=f'{i:02d}_train_acc', lower_is_better=False) for i in range(model.num_models)]

    # scoring_callbacks = [HybridScoring(scoring=get_subject_loss_scorer(i, torch.nn.NLLLoss()), on_train=False, name=f'{i:02d}_valid_loss', lower_is_better=True) for i in range(model.num_models)]

    clf = HybridClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=config.train.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   GradientNormClipping(gradient_clip_value=1),
                   WandbLogger(wandb.run),
                   Checkpoint(monitor="valid_loss_best", load_best=True,
                              dirname=f"/workspace/params/temptrain-{experiment_name}", f_params="params.pt"),
                   lrscheduler,
                   HybridScoring(scoring=average_acc_scoring, on_train=True, name='avg_train_acc',
                                 lower_is_better=False),
                   HybridScoring(scoring=average_acc_scoring, on_train=False, name='avg_valid_acc',
                                 lower_is_better=False)] + scoring_callbacks,
        device=device,
        verbose=1,
    )
    return clf


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


class HybridEvaluation(BaseEvaluation):
    def __init__(self, *args, eval_config=None, EA_in_eval=False, len_run=None, wandb_params=None, **kwargs):
        super(HybridEvaluation, self).__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def evaluate(self, dataset, pipelines, grid_search):

        """Evaluate results on a single dataset.

        This method return a generator. each results item is a dict with
        the following convension::

            res = {'time': Duration of the training ,
                   'dataset': dataset id,
                   'subject': subject id,
                   'session': session id,
                   'score': score,
                   'n_samples': number of training examples,
                   'n_channels': number of channel,
                   'pipeline': pipeline name}
        """

        init_time = time()
        X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)
        print(f"(1) Data got {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        print(f"(2) Encoded {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)

        scorer = get_scorer(self.paradigm.scoring)

        print(f"(3) Setup done {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        cv = LeaveOneGroupOut()
        # Progressbar at subject level
        subject_num = 0
        for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject", ):
            subject = groups[test[0]]
            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)

            # iterate over pipelines
            for name, clf in run_pipes.items():
                t_start = time()
                copyclf = deepcopy(clf)
                subject_num += 1
                train_run = active_wandb(*self.wandb_params, subject_num, train=True)
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)
                wandb.finish()

                duration = time() - t_start

                ix = test < (self.len_run * 2 + test[0])
                # ix = sessions[test] == 'session_T'

                eval_model = model["Net"].module.generate_branch_model()
                eval_classifier = define_clf(eval_model, self.eval_config)
                if self.EA_in_eval:
                    create_dataset = TransformaParaWindowsDatasetEA(self.len_run)
                else:
                    create_dataset = TransformaParaWindowsDataset()
                eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + model["Hybrid_adapter"].n_trials_used))

                eval_run = active_wandb(self.wandb_params[0], self.eval_config, subject_num, train=False)
                for callback in eval_classifier.callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                if type(eval_model.unique_modules) != type(nn.Identity()) or \
                        list(eval_model.shared_modules.parameters())[0].requires_grad:
                    eval_clf = eval_pipe.fit(X[test[ix]], y[test[ix]])
                    create_dataset.y = y[test[ix_eval]]
                    score = _score(eval_clf, X[test[ix_eval]], y[test[ix_eval]], scorer)
                else:
                    eval_classifier.initialize()
                    eval_classifier.classes_inferred_ = np.unique(to_numpy(y))
                    create_dataset.y = y[test[ix_eval]]
                    Xproc = create_dataset.transform(X[test[ix_eval]], y[test[ix_eval]])
                    score = _score(eval_classifier, Xproc, y[test[ix_eval]], scorer)

                wandb.run.summary['eval_score'] = score
                wandb.finish()

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "dataset": dataset,
                    "subject": subject,
                    "session": 'session_E',
                    "score": score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                }

                print(res)

                yield res
