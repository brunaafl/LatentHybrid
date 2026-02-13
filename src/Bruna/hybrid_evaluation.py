import warnings
from pathlib import Path

import moabb
import torch
from pyriemann.utils.covariance import covariances

from skorch.callbacks import WandbLogger
from skorch.utils import to_numpy

import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
)
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer, accuracy_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from torchinfo import torchinfo

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time

from copy import deepcopy

from mne.epochs import BaseEpochs
from analysis import class_distinctiveness
from hybrid_transform import HybridAggregateTransform
from hybrid_classifier import define_hybrid_clf
import wandb

from alignment import euclidean_alignment

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

##### Base class #####

class _BaseHybridEvaluation(BaseEvaluation):
    """
    Base class containing shared logic for Hybrid evaluations.
    Handles initialization, data loading, and the main cross-validation loops.
    """

    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None,
                 mode='Fit', wandb_params=None, remove_bn='False', seed=0, criterion_type=None,
                 online="off", cross_session = False, **kwargs):
        add_cols = ["head"]
        super().__init__(additional_columns=add_cols, *args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params
        self.run_dir = Path(run_dir) if run_dir else None
        self.mode = mode
        self.remove_bn = remove_bn
        self.criterion_type = criterion_type
        self.seed = seed
        self.online = online
        self.cross_session = cross_session

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def _safe_copy_skorch_model(self, model):
        """
        Creates a deep copy of a Skorch estimator.
        """
        new_model = clone(model)

        if model['Net'].initialized_:
            if hasattr(model['Net'], 'module_params_'):
                new_model['Net'].module_params_ = deepcopy(model['Net'].module_params_)

            # Initialize the new model to create the underlying PyTorch module
            new_model['Net'].initialize()

            new_model['Net'].module.load_state_dict(model['Net'].module.state_dict())
            new_model['Net'].module_.load_state_dict(model['Net'].module_.state_dict())

        return new_model

    def _get_model_dir_name(self):
        """Helper to generate model directory string."""
        bn = 'nobn' if self.remove_bn == 'True' else 'bn'
        ea = 'ea' if self.EA_in_eval else 'noea'
        experiment_name = self.wandb_params[1].train.experiment_name
        return Path(f'/workspace/models/{experiment_name}_{bn}_{ea}')

    def _create_eval_pipeline(self, model, subj_idx, dataset_code):
        """Creates the evaluation pipeline for a specific head."""
        # Isolate the specific head (branch)
        copy_model = deepcopy(model)
        eval_model = copy_model["Net"].module.generate_branch_model(subj_idx)
        eval_model.num_models = 1

        # Define classifier
        eval_classifier = define_hybrid_clf(
            deepcopy(eval_model),
            self.eval_config,
            experiment_name='Evaluation',
            criterion_type=self.criterion_type
        )

        # Define Dataset Transform
        if self.EA_in_eval:
            create_dataset = HybridAggregateTransform(EA_len_run=self.len_run, data_code=dataset_code)
        else:
            create_dataset = HybridAggregateTransform(data_code=dataset_code)

        return Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)]), eval_model

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):
        """
        Main evaluation loop - initial pretrain that is common to both methods
        Delegates specific processing to `_process_results`.
        """
        # --- 1. Data Loading ---
        init_time = time()
        X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)
        print(f"(1) Data loaded: {(time() - init_time):.2f}s")

        # Encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)
        print(f"(2) Encoded: {(time() - init_time):.2f}s")

        # Metadata
        groups = metadata.subject.values
        subjects=None
        if self.cross_session:
            subjects = groups
            groups = metadata.session.values

        n = len(np.unique(groups))
        nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]

        cv = LeaveOneGroupOut()
        subject_num = 0

        print(f"(3) Setup done: {(time() - init_time):.2f}s")

        # Cross-Subject Loop
        for train, test in tqdm(cv.split(X, y, groups), total=n, desc=f"{dataset.code}-CrossSubject"):
            subject = groups[test[0]]

            # Check computation cache
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

            for name, clf in run_pipes.items():
                t_start = time()
                subject_num += 1

                copyclf = deepcopy(clf)
                _ = active_wandb(*self.wandb_params, subject_num, train=True)

                # Attach WandB logger
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                # Fit the main model
                copyclf['Net'].classes = [0,1]

                model = copyclf.fit(
                    X[train], None,
                    Hybrid_adapter__labels=y[train],
                    Hybrid_adapter__subject_groups=groups[train],
                    Hybrid_adapter__info=X[train].info
                )

                # Delegate to subclass for saving/evaluation/inference
                yield from self._process_results(
                            model=model,
                            dataset=dataset,
                            X=X, y=y, groups=groups,
                            train_indices=train,
                            test_indices=test,
                            subject_num=subject_num,
                            subject_id=subject,
                            subjects = subjects,
                            pipeline_name=name,
                            nchan=nchan,
                            t_start=t_start
                        )

                # Clean up WandB run after processing pipeline
                if wandb.run is not None:
                    wandb.finish()

    def _process_results(self, **kwargs):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError



########################


class HybridEvaluation(BaseEvaluation):
    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None,
                 mode='Fit', wandb_params=None,remove_bn='False', seed=0, criterion_type=None,
                 online=False, **kwargs):
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
        self.online = online

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def safe_copy_skorch_model(self,model):
        """
        Creates a deep copy of a Skorch estimator.
        """
        new_model = clone(model)

        # 2. If the original model was already trained/initialized, we must transfer the state
        if model['Net'].initialized_:
            if hasattr(model['Net'], 'module_params_'):
                new_model['Net'].module_params_ = deepcopy(model['Net'].module_params_)

            # Initialize the new model to create the underlying PyTorch module
            new_model['Net'].initialize()

            new_model['Net'].module.load_state_dict(model['Net'].module.state_dict())
            new_model['Net'].module_.load_state_dict(model['Net'].module_.state_dict())

        return new_model


    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):

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
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

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
                copyclf['Net'].classes = [0,1]

                print(X[train].get_data().shape)

                # Fit
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)

                # Save Model Logic
                bn = 'nobn' if self.remove_bn == 'True' else 'bn'
                ea = 'ea' if self.EA_in_eval else 'noea'

                model_dir = Path(f'/workspace/models/{self.wandb_params[1].train.experiment_name}_{bn}_{ea}')
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f'best_model_{subject_num}-shared.pth'

                print(f"(1) Model saved at {model_dir}")
                state_dict = model['Net'].module.state_dict()
                torch.save(model['Net'].module.state_dict(), model_path)

                artifact = wandb.Artifact(f'best_model_{subject_num}-shared.pth', type="model")
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)

                wandb.finish()

                duration = time() - t_start
                # Test set
                ix = test < (self.len_run * 2 + test[0])

                print(sum(ix))
                # Evaluation set
                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + model["Hybrid_adapter"].n_trials_used))
                print('n trials: ', model["Hybrid_adapter"].n_trials_used)
                print(sum(ix_eval))

                # Evaluation
                # Iterate over all source heads
                for subj in range(model['Net'].module.num_models):

                    #trained_pytorch_module = deepcopy(model["Net"].module_)
                    #eval_model = trained_pytorch_module.generate_branch_model(subj)
                    copy_model = self.safe_copy_skorch_model(model)
                    #self.verification(model, copy_model)
                    eval_model = copy_model["Net"].module.generate_branch_model(subj)
                    eval_model.num_models = 1

                    eval_classifier = define_hybrid_clf(deepcopy(eval_model), self.eval_config,
                                                        experiment_name='Evaluation', criterion_type=self.criterion_type,)
                    if self.EA_in_eval:
                        create_dataset = HybridAggregateTransform(EA_len_run=self.len_run, data_code=dataset.code)
                    else:
                        create_dataset = HybridAggregateTransform(data_code=dataset.code)
                    eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                    ### Inference part

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
                        print(X[test[ix_eval]].get_data().shape)
                        X_trn = eval_pipe['Braindecode_dataset'].transform(X[test[ix_eval]])

                        # For online exp
                        if self.online:
                            # if EA
                            if eval_pipe["Braindecode_dataset"].EA_len_run is not None:
                                # Remove online EA from transformation
                                eval_pipe["Braindecode_dataset"].EA_len_run = None

                                _, r = euclidean_alignment(X[test[ix]][:self.len_run])
                                X_eval = np.matmul(r, X[test[ix_eval]])
                            else:
                                X_eval = X[test[ix_eval]]
                        else:
                            X_eval = X[test[ix_eval]]

                        # Fix dimension and predict
                        eval_pipe['Net'].module.eval()
                        pred,_ = eval_pipe['Net'].forward(X_eval)
                        y_pred = pred.flatten(0, 1).argmax(dim=1)
                        # Compute accuracy
                        score = accuracy_score(y[test[ix_eval]], y_pred)
                        print(score)

                    # If fine-tuning
                    else:
                        eval_run = active_wandb_eval(self.wandb_params[0], self.eval_config, subject_num, subj,
                                                     train=False)

                        for callback in eval_classifier.callbacks:
                            if isinstance(callback, WandbLogger):
                                callback.wandb_run = wandb.run

                        print(X[test[ix]].get_data().shape)
                        t_start = time()
                        eval_clf = eval_pipe.fit(X[test[ix]], None,
                                                           Braindecode_dataset__labels=y[test[ix]],
                                                           Braindecode_dataset__subject_groups=groups[test[ix]],
                                                           Braindecode_dataset__info=X[test[ix]].info)
                        duration = duration + time() - t_start

                        # Save Fine-tuned model
                        ft_path = model_dir / f'best_model_{subject_num}-head-{subj}_ft.pth'
                        torch.save(eval_clf['Net'].module.state_dict(), ft_path)

                        artifact = wandb.Artifact(f'best_model_{subject_num}-head-{subj}_ft.pth', type="model")
                        artifact.add_file(str(ft_path))
                        wandb.log_artifact(artifact)

                        eval_clf["Braindecode_dataset"].labels = y[test[ix_eval]]
                        eval_clf["Braindecode_dataset"].groups = groups[test[ix_eval]]
                        eval_clf["Braindecode_dataset"].info = X[test[ix_eval]].info

                        X_trn = eval_clf['Braindecode_dataset'].transform(X[test[ix_eval]])

                        # For online exp
                        if self.online:
                            # if EA
                            if eval_clf["Braindecode_dataset"].EA_len_run is not None:
                                # Remove online EA from transformation
                                eval_clf["Braindecode_dataset"].EA_len_run = None

                                _, r = euclidean_alignment(X[test[ix]][:self.len_run])
                                X_eval = np.matmul(r, X[test[ix_eval]])
                            else: X_eval = X[test[ix_eval]]
                        else: X_eval = X[test[ix_eval]]

                        # Predict
                        eval_clf['Net'].module.eval()
                        print(X[test[ix_eval]].get_data().shape)
                        pred, feat = eval_clf['Net'].forward(X_trn)
                        feat = feat.flatten(0, 1).squeeze(2).to('cpu')
                        y_pred = pred.flatten(0, 1).argmax(dim=1)

                        #eval_pipe['Net'].module.eval()
                        #y_pred, feat = eval_pipe['Net'].specialized_predict(X_trn)

                        # Compute accuracy
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
    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None, mode='Fit', wandb_params=None,remove_bn='False', criterion_type=None,
                 **kwargs):
        add_cols = ["head", 'class_distinctivness']
        super(HybridChooseHead, self).__init__(additional_columns=add_cols, *args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params
        self.run_dir = run_dir
        self.mode = mode
        self.criterion_type = criterion_type
        self.remove_bn = remove_bn

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def safe_copy_skorch_model(self,model):
        """
        Creates a deep copy of a Skorch estimator.
        """
        new_model = clone(model)

        # 2. If the original model was already trained/initialized, we must transfer the state
        if model['Net'].initialized_:
            if hasattr(model['Net'], 'module_params_'):
                new_model['Net'].module_params_ = deepcopy(model['Net'].module_params_)

            # Initialize the new model to create the underlying PyTorch module
            new_model['Net'].initialize()

            new_model['Net'].module.load_state_dict(model['Net'].module.state_dict())
            new_model['Net'].module_.load_state_dict(model['Net'].module_.state_dict())

        return new_model

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):

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
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

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
                copyclf['Net'].classes = [0,1]

                print(X[train].get_data().shape)

                # Fit
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)

                # Save Model Logic
                bn = 'nobn' if self.remove_bn == 'True' else 'bn'
                ea = 'ea' if self.EA_in_eval else 'noea'

                model_dir = Path(f'/workspace/models/{self.wandb_params[1].train.experiment_name}_{bn}_{ea}')
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f'best_model_{subject_num}-shared.pth'

                print(f"(1) Model saved at {model_dir}")
                state_dict = model['Net'].module.state_dict()
                torch.save(model['Net'].module.state_dict(), model_path)

                artifact = wandb.Artifact(f'best_model_{subject_num}-shared.pth', type="model")
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)

                wandb.finish()

                duration = time() - t_start

                # Test set
                ix = test < (self.len_run * 2 + test[0])

                # Choose best head based on the inference
                for subj in range(model['Net'].module.num_models):

                    copy_model = self.safe_copy_skorch_model(model)
                    eval_model = copy_model["Net"].module.generate_branch_model(subj)
                    eval_model.num_models = 1

                    eval_classifier = define_hybrid_clf(deepcopy(eval_model), self.eval_config,
                                                        experiment_name='Evaluation', criterion_type=self.criterion_type,)
                    if self.EA_in_eval:
                        create_dataset = HybridAggregateTransform(EA_len_run=self.len_run, data_code=dataset.code)
                    else:
                        create_dataset = HybridAggregateTransform(data_code=dataset.code)
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
                    eval_pipe['Net'].module.eval()
                    pred, feat = eval_pipe['Net'].forward(X_trn)
                    feat = feat.flatten(0, 1).squeeze(2).to('cpu')

                    y_pred = pred.flatten(0, 1).argmax(dim=1)
                    # Compute accuracy
                    score = accuracy_score(y[test[ix]], y_pred)

                    cov = covariances(feat, estimator='lwf')
                    cls_distinc = class_distinctiveness(cov, y[test[ix]])

                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "head": subj,
                        "subject": subject,
                        "session": 'session_E',
                        "score": score,
                        "class_distinctivness":cls_distinc,
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
