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
                 online="off", cross_session = False, choose = False, dataset_code="BNCI2014001",**kwargs):

        add_cols = ["head"]
        if choose:
            add_cols.append("class_distinctivness")

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
        self.dataset_code = dataset_code

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def _handle_online_ea(self, X, test_indices, ix_calib, ix_eval, dataset_obj):
        """Helper to handle the specific matrix multiplication for Online EA."""
        if self.online=='on' and dataset_obj.EA_len_run is not None:
            # Remove automatic alignment in transformation
            dataset_obj.EA_len_run = None
            _, r = euclidean_alignment(X[test_indices[ix_calib]].get_data()[:self.len_run])
            return np.matmul(r, X[test_indices[ix_eval]].get_data())
        return X[test_indices[ix_eval]]

    def _save_weights(self, model, wandb, model_dir, subject_num, head = None):

        if head is None:
            model_path = model_dir / f'best_model_{subject_num}-shared.pth'
            pth_artifact = f'best_model_{subject_num}-shared.pth'

        else:
            model_path = model_dir / f'best_model_{subject_num}-head-{head}_ft.pth'
            pth_artifact = f'best_model_{subject_num}-head-{head}_ft.pth'

        torch.save(model['Net'].module.state_dict(), model_path)
        artifact = wandb.Artifact(pth_artifact, type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)


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

            new_model['Net'].optimizer_.load_state_dict(model['Net'].optimizer_.state_dict())

        return new_model

    def _get_model_dir_name(self):
        """Helper to generate model directory string."""
        bn = 'nobn' if self.remove_bn == 'True' else 'bn'
        ea = 'ea' if self.EA_in_eval else 'noea'
        experiment_name = self.wandb_params[1].train.experiment_name
        return Path(f'/workspace/models/{experiment_name}_{bn}_{ea}')

    def _create_eval_pipeline(self, model, subj_idx, dataset_code):
        """Creates the evaluation pipeline for a specific head."""

        # COpy model weights
        copy_model = self._safe_copy_skorch_model(model)

        eval_model = copy_model["Net"].module.generate_branch_model(subj_idx)
        eval_model.num_models = 1

        # Define classifier
        eval_classifier = define_hybrid_clf(eval_model, self.eval_config, experiment_name='Evaluation', criterion_type=self.criterion_type, dataset=self.dataset_code )
        print('EA in eval? ', self.EA_in_eval)

        # Create dataset transforms
        if self.EA_in_eval:
            create_dataset = HybridAggregateTransform(EA_len_run=self.len_run, data_code=dataset_code)
        else:
            create_dataset = HybridAggregateTransform(data_code=dataset_code)

        eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

        return eval_pipe, eval_model

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):
        """
        Main evaluation loop - initial pretrain that is common to both methods
        Delegates specific processing to `_process_results` of each child class
        """

        # Loqd data
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

                _ = active_wandb(*self.wandb_params, subject_num, train=True)

                # Start wandb monitoring
                t_start = time()
                copyclf = deepcopy(clf)
                train_run = active_wandb(*self.wandb_params, subject_num, train=True)
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run
                copyclf['Net'].classes = [0,1]  # Fixing error when not explicitly passing y

                # Fit
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)

                # Save Model Logic
                bn = 'nobn' if self.remove_bn == 'True' else 'bn'
                ea = 'ea' if self.EA_in_eval else 'noea'

                model_dir = Path(f'/workspace/models/{self.wandb_params[1].train.experiment_name}_{bn}_{ea}')
                model_dir.mkdir(parents=True, exist_ok=True)

                self._save_weights(model, wandb, model_dir, subject_num)

                wandb.finish()

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
                            t_start=t_start,
                            model_dir=model_dir
                        )

                # Clean up WandB run after processing pipeline
                if wandb.run is not None:
                    wandb.finish()

    def _process_results(self, **kwargs):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError



########################


class HybridEvaluation(_BaseHybridEvaluation):
    """
    Standard Hybrid Evaluation:
    - Iterates all heads
    - Creates new specific pipeline for each head
    - Performs Fine-tuning or Inference on the Test set
    """

    def _process_results(self, model, dataset, X, y, groups, train_indices, test_indices,
                         subject_num, subject_id, pipeline_name, nchan, t_start, model_dir, subjects=None):

        duration = time() - t_start

        # Define calibration set
        ix_calib = test_indices < (self.len_run*2 + test_indices[0])

        # Define evaluation set
        ix_eval = np.logical_and(
            test_indices >= (self.len_run * 2 + test_indices[0]),
            test_indices < (test_indices[0] + model["Hybrid_adapter"].n_trials_used)
        )

        # Iterate over heads
        num_heads = model['Net'].module.num_models
        print('Num heads: ', num_heads)
        for subj_idx in range(num_heads):

            # Create Pipeline for this specific head
            eval_pipe, eval_model = self._create_eval_pipeline(model, subj_idx, dataset.code)

            # Transform evaluation data
            # Online Logic
            X_eval_input = self._handle_online_ea(X, test_indices, ix_calib, ix_eval,
                                                  eval_pipe["Braindecode_dataset"])

            # If not fine-tuning
            if self.mode == 'Inference':
                eval_pipe['Net'].initialize()
                eval_pipe['Net'].module.load_state_dict(eval_model.state_dict())
                eval_pipe['Net'].module_.load_state_dict(eval_model.state_dict())

                # Hard parameterization
                # TODO: put on fit as optional
                eval_pipe["Braindecode_dataset"].labels = y[test_indices[ix_eval]]
                eval_pipe["Braindecode_dataset"].groups = groups[test_indices[ix_eval]]
                eval_pipe["Braindecode_dataset"].info = X[test_indices[ix_eval]].info

                # Transform predict
                X_trn = eval_pipe['Braindecode_dataset'].transform(X_eval_input)

                # Fix dimension and predict
                eval_pipe['Net'].module.eval()
                pred, _ = eval_pipe['Net'].forward(X_trn)
                y_pred = pred.flatten(0, 1).argmax(dim=1)

                # Compute accuracy
                score = accuracy_score(y[test_indices[ix_eval]], y_pred)
                print(score)

            # If fine-tuning
            else:

                # Setup WandB for Eval
                _ = active_wandb_eval(self.wandb_params[0], self.eval_config, subject_num, subj_idx, train=False)
                for callback in eval_pipe['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                t_start = time()

                # Fit on Calibration Set
                eval_clf = eval_pipe.fit(
                    X[test_indices[ix_calib]], None,
                    Braindecode_dataset__labels=y[test_indices[ix_calib]],
                    Braindecode_dataset__subject_groups=groups[test_indices[ix_calib]],
                    Braindecode_dataset__info=X[test_indices[ix_calib]].info
                )
                duration = duration + time() - t_start

                # Save Fine-tuned model
                self._save_weights(eval_clf, wandb, model_dir, subject_num, head=subj_idx)

                eval_clf["Braindecode_dataset"].labels = y[test_indices[ix_eval]]
                eval_clf["Braindecode_dataset"].groups = groups[test_indices[ix_eval]]
                eval_clf["Braindecode_dataset"].info = X[test_indices[ix_eval]].info

                # Predict
                eval_clf['Net'].module.eval()
                X_trn = eval_clf['Braindecode_dataset'].transform(X_eval_input)

                pred, _ = eval_clf['Net'].forward(X_trn)
                y_pred = pred.flatten(0, 1).argmax(dim=1)

                # Compute accuracy
                score = accuracy_score(y[test_indices[ix_eval]], y_pred)

                wandb.run.summary['eval_score'] = score
                wandb.finish()

            # Yield Result
            res = {
                "time": duration,
                "dataset": dataset,
                "head": subj_idx,
                "subject": subject_id,
                "session": 'both',
                "score": score,
                "n_samples": len(train_indices),
                "n_channels": nchan,
                "pipeline": pipeline_name,
            }
            print(res)
            yield res


class HybridChooseHead(_BaseHybridEvaluation):
    """
    Hybrid Head Selection:
    - Iterates all heads
    - Instantiates pipeline for each head
    - Performs JUST inference on the Calibration set (used to select the best head)
    """

    def _process_results(self, model, dataset, X, y, groups, train_indices, test_indices,
                         subject_num, subject_id, pipeline_name, nchan, t_start, model_dir, subjects=None):
        duration = time() - t_start

        # Define Split Indices
        # Calibration set: used for inference here
        ix_calib = test_indices < (self.len_run*2 + test_indices[0])

        # Iterate over heads
        num_heads = model['Net'].module.num_models

        for subj_idx in range(num_heads):
            # Create Pipeline
            eval_pipe, eval_model = self._create_eval_pipeline(model, subj_idx, dataset.code)

            # Setup Inference - i need to load ever
            eval_pipe['Net'].initialize()
            eval_pipe['Net'].module.load_state_dict(eval_model.state_dict())
            eval_pipe['Net'].module_.load_state_dict(eval_model.state_dict())

            # Prepare Dataset (On Calibration indices)
            eval_pipe["Braindecode_dataset"].labels = y[test_indices[ix_calib]]
            eval_pipe["Braindecode_dataset"].groups = groups[test_indices[ix_calib]]
            eval_pipe["Braindecode_dataset"].info = X[test_indices[ix_calib]].info

            # Transform and Predict
            X_trn = eval_pipe["Braindecode_dataset"].transform(X[test_indices[ix_calib]])
            eval_pipe['Net'].module.eval()
            pred, feat = eval_pipe['Net'].forward(X_trn)
            feat = feat.transpose(0, 1).flatten(0, 1).squeeze(2).to('cpu').numpy()
            y_pred = pred.transpose(0, 1).flatten(0, 1).argmax(dim=1)

            score = accuracy_score(y[test_indices[ix_calib]], y_pred)

            # Compute class distinctiveness
            cov = covariances(feat, estimator='lwf')
            class_d = class_distinctiveness(cov, y[test_indices[ix_calib]])

            res = {
                "time": duration,
                "dataset": dataset,
                "head": subj_idx,
                "subject": subject_id,
                "session": 'both',
                "score": score,
                "n_samples": len(train_indices),
                "n_channels": nchan,
                "pipeline": pipeline_name,
                "class_distinctivness": class_d,
            }

            print(res)
            yield res

class HybridSessionEvaluation(_BaseHybridEvaluation):
    """
    Session Hybrid Evaluation:
    - Shared model using all subjects
    - Saves the shared model.
    - Iterates all heads.
    - Performs Inference on the other session
    """

    def _process_results(self, model, dataset, X, y, groups, train_indices, test_indices,
                         subject_num, subject_id, subjects, pipeline_name, nchan, t_start):

        # Save Shared Model
        model_dir = self._get_model_dir_name()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'best_model_sess_{subject_num}-shared.pth'

        print(f"(1) Model saved at {model_dir}")
        torch.save(model['Net'].module.state_dict(), model_path)

        artifact = wandb.Artifact(f'best_model_sess_{subject_num}-shared.pth', type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        wandb.finish()  # Finish training run

        duration_train = time() - t_start

        # Iterate over all source heads
        num_heads = model['Net'].module.num_models

        for subj_head in range(num_heads):

            # Get indexes where test subject
            subj_indices = subjects == subjects[subj_head]

            # Dont need a calibration set for this case - whithin subject
            ix_eval = subj_indices[test_indices]

            # Create Pipeline for this specific head
            eval_pipe, eval_model = self._create_eval_pipeline(model, subj_head, dataset.code)

            # Inference
            eval_pipe['Net'].initialize()
            # Copy weights
            eval_pipe['Net'].module.shared_modules = deepcopy(eval_model.shared_modules)
            eval_pipe['Net'].module_.shared_modules = deepcopy(eval_model.shared_modules)
            eval_pipe['Net'].module.unique_modules = deepcopy(eval_model.unique_modules)
            eval_pipe['Net'].module_.unique_modules = deepcopy(eval_model.unique_modules)

            # Prepare Dataset
            eval_pipe["Braindecode_dataset"].labels = y[test_indices[ix_eval]]
            eval_pipe["Braindecode_dataset"].groups = groups[test_indices[ix_eval]]
            eval_pipe["Braindecode_dataset"].info = X[test_indices[ix_eval]].info

            # Transform & Predict
            X_trn = eval_pipe["Braindecode_dataset"].transform(X[ix_eval])
            eval_pipe['Net'].module.eval()
            pred, _ = eval_pipe['Net'].forward(X_trn)
            y_pred = pred.flatten(0, 1).argmax(dim=1)

            score = accuracy_score(y[test_indices[ix_eval]], y_pred)
            final_duration = duration_train

            # Yield Result
            res = {
                "time": final_duration,
                "dataset": dataset,
                "head": subj_head,
                "subject": subject_id,
                "session": 'session_E',
                "score": score,
                "n_samples": len(train_indices),
                "n_channels": nchan,
                "pipeline": pipeline_name,
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
