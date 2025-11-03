import warnings
from pathlib import Path

import moabb
import torch

from skorch.callbacks import WandbLogger
from skorch.utils import to_numpy

import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
)
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer, accuracy_score

from sklearn.pipeline import Pipeline
from torchinfo import torchinfo

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time

from copy import deepcopy

from mne.epochs import BaseEpochs

from hybrid_transform import HybridAggregateTransform
from hybrid_classifier import define_hybrid_clf

import wandb

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

class HybridEvaluation(BaseEvaluation):
    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None, mode='Fit', wandb_params=None,remove_bn='False', seed=0, criterion_type=None,
                 **kwargs):
        add_cols = ["head"]
        super(HybridEvaluation, self).__init__(additional_columns=add_cols, *args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params
        self.run_dir = run_dir
        self.mode = mode
        self.remove_bn = remove_bn
        self.criterion_type = criterion_type
        self.seed = seed

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

        # Get number of channels
        nchan = (
            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
        )

        # Progressbar at subject level
        subject_num = 0
        for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject", ):
            subject = groups[test[0]]

            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)

            # iterate over pipelines
            for name, clf in run_pipes.items():

                # Start wandb monitoring
                t_start = time()
                copyclf = deepcopy(clf)
                subject_num += 1
                train_run = active_wandb(*self.wandb_params, subject_num, train=True)
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                # Fit
                #pdb.set_trace()
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)

                if self.remove_bn=='True':
                    bn = 'nobn'
                else:
                    bn='bn'

                if self.EA_in_eval:
                    ea = 'ea'
                else:
                    ea = 'noea'

                model_dir = Path(f'/workspace/models/{self.wandb_params[1].train.experiment_name}_{bn}_{ea}')
                model_path = model_dir / f'best_model_{subject_num}-shared.pth'
                model_dir.mkdir(parents=True, exist_ok=True)

                print(f"(1) Model saved at {model_dir}")
                torch.save(model['Net'].module.state_dict(), model_path)

                artifact = wandb.Artifact(f'best_model_{subject_num}-shared.pth', type="model")
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)

                wandb.finish()

                duration = time() - t_start
                # Test set
                ix = test < (self.len_run * 2 + test[0])

                # Iterate over all source heads
                for subj in range(model['Net'].module.num_models):

                    copy_model = deepcopy(model)
                    eval_model = copy_model["Net"].module.generate_branch_model(subj)
                    eval_model.num_models = 1

                    eval_classifier = define_hybrid_clf(deepcopy(eval_model), self.eval_config,
                                                        experiment_name='Evaluation', criterion_type=self.criterion_type,)
                    if self.EA_in_eval:
                        create_dataset = HybridAggregateTransform(EA_len_run=self.len_run, data_code=dataset.code)
                    else:
                        create_dataset = HybridAggregateTransform(data_code=dataset.code)
                    eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                    # Evaluation set
                    ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                             test < (test[0] + copy_model["Hybrid_adapter"].n_trials_used))

                    eval_run = active_wandb_eval(self.wandb_params[0], self.eval_config, subject_num, subj, train=False)

                    for callback in eval_classifier.callbacks:
                        if isinstance(callback, WandbLogger):
                            callback.wandb_run = wandb.run

                    # Inference part

                    # If not fine-tuning
                    if self.mode == 'Inference':
                        eval_pipe['Net'].initialize()
                        eval_pipe['Net'].module.shared_modules = deepcopy(eval_model.shared_modules)
                        eval_pipe['Net'].module_.shared_modules = deepcopy(eval_model.shared_modules)
                        eval_pipe['Net'].module.unique_modules = deepcopy(eval_model.unique_modules)
                        eval_pipe['Net'].module_.unique_modules = deepcopy(eval_model.unique_modules)

                        eval_pipe["Braindecode_dataset"].labels = y[test[ix_eval]]
                        eval_pipe["Braindecode_dataset"].groups = groups[test[ix_eval]]
                        eval_pipe["Braindecode_dataset"].info = X[test[ix_eval]].info
                        X_trn = eval_pipe['Braindecode_dataset'].transform(X[test[ix_eval]])

                        # Fix dimension and predict
                        eval_pipe['Net'].module.eval()
                        pred,_ = eval_pipe['Net'].forward(X_trn)
                        y_pred = pred.flatten(0, 1).argmax(dim=1)
                        # Compute accuracy
                        score = accuracy_score(y[test[ix_eval]], y_pred)
                        print(score)

                    # If fine-tuning
                    else:
                        t_start = time()
                        eval_clf = eval_pipe.fit(X[test[ix]], None,
                                                           Braindecode_dataset__labels=y[test[ix]],
                                                           Braindecode_dataset__subject_groups=groups[test[ix]],
                                                           Braindecode_dataset__info=X[test[ix]].info)
                        duration = duration + time() - t_start

                        torch.save(eval_clf['Net'].module.state_dict(), model_dir / f'best_model_{subject_num}-head-{subj}_ft.pth')
                        artifact = wandb.Artifact(f'best_model_{subject_num}-head-{subj}_ft.pth', type="model")
                        artifact.add_file(str(model_dir / f'best_model_{subject_num}-head-{subj}_ft.pth'))
                        wandb.log_artifact(artifact)

                        eval_clf["Braindecode_dataset"].labels = y[test[ix_eval]]
                        eval_clf["Braindecode_dataset"].groups = groups[test[ix_eval]]
                        eval_clf["Braindecode_dataset"].info = X[test[ix_eval]].info
                        X_trn = eval_clf['Braindecode_dataset'].transform(X[test[ix_eval]])

                        # Predict
                        pred, _ = eval_pipe['Net'].forward(X_trn)
                        y_pred = pred.flatten(0, 1).argmax(dim=1)
                        score = accuracy_score(y[test[ix_eval]], y_pred)

                        wandb.run.summary['eval_score'] = score
                        wandb.finish()

                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "head": subj,
                        "subject": subject,
                        "session": 'session_E',
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    print(res)
                    yield res
            #break

class HybridChooseHead(BaseEvaluation):
    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None, mode='Fit',
                 wandb_params=None,
                 **kwargs):
        add_cols = ["head"]
        super(HybridChooseHead, self).__init__(additional_columns=add_cols, *args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params
        self.run_dir = run_dir
        self.mode = mode

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

        # Get number of channels
        nchan = (
            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
        )

        # Progressbar at subject level
        subject_num = 0
        for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject", ):
            subject = groups[test[0]]

            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)

            # iterate over pipelines
            for name, clf in run_pipes.items():

                # Start wandb monitoring
                t_start = time()
                copyclf = deepcopy(clf)
                subject_num += 1
                train_run = active_wandb(*self.wandb_params, subject_num, train=True)
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                # Fit
                #pdb.set_trace()
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)
                wandb.finish()

                duration = time() - t_start

                # Test set
                ix = test < (self.len_run * 2 + test[0])

                # Evaluation set
                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + model["Hybrid_adapter"].n_trials_used))

                copy_model = deepcopy(model)

                best_subject = 0
                best_score = 0

                # Choose best head based on the inference
                for subj in range(copy_model['Net'].module.num_models):

                    eval_model = copy_model["Net"].module.generate_branch_model(subj)
                    eval_model.num_models = 1

                    copy_eva_model = deepcopy(eval_model)
                    eval_classifier = define_hybrid_clf(copy_eva_model, self.eval_config,
                                                        experiment_name='Evaluation')
                    if self.EA_in_eval:
                        create_dataset = HybridAggregateTransform(EA_len_run=self.len_run)
                    else:
                        create_dataset = HybridAggregateTransform()
                    eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                    # Inference on the calibration set
                    eval_pipe['Net'].initialize()
                    eval_pipe['Net'].module.shared_modules = deepcopy(eval_model.shared_modules)
                    eval_pipe['Net'].module_.shared_modules = deepcopy(eval_model.shared_modules)
                    eval_pipe['Net'].module.unique_modules = deepcopy(eval_model.unique_modules)
                    eval_pipe['Net'].module_.unique_modules = deepcopy(eval_model.unique_modules)

                    eval_pipe["Braindecode_dataset"].labels = y[test[ix]]
                    eval_pipe["Braindecode_dataset"].groups = groups[test[ix]]
                    eval_pipe["Braindecode_dataset"].info = X[test[ix]].info
                    X_trn = eval_pipe['Braindecode_dataset'].transform(X[test[ix]])

                    # Fix dimension and predict
                    y_pred = eval_pipe['Net'].forward(X_trn).flatten(0, 1).argmax(dim=1)
                    # Compute accuracy
                    score = accuracy_score(y[test[ix]], y_pred)
                    print(score)

                    if score > best_score:
                        best_score = score
                        best_subject = subj
                        print(best_subject)

                res = {
                    "time": duration,
                    "dataset": dataset,
                    "head": best_subject,
                    "subject": subject,
                    "session": 'session_E',
                    "score": best_score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                }

                print(res)
                yield res



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
    }

    run = wandb.init(
        project=f"{args.model}",
        group=config.train.experiment_name,
        name=f"{subject}-Shared:{args.ea}",
        config=wconfig
    )
    return run


def active_wandb_eval(args, config, subject, subj, train=True):
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
        "head": subj,
        "subject": subject,
        "train": train,
    }

    run = wandb.init(
        project=f"{args.model}",
        group=config.train.experiment_name,
        name=f"{subject}-Head-{subj}:{args.ea}",
        config=wconfig
    )
    return run
