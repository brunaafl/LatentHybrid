from skorch.utils import to_tensor, to_numpy, to_device

import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer, accuracy_score
from sklearn.pipeline import Pipeline

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time

from copy import deepcopy

from mne.epochs import BaseEpochs

from hybrid_transform import HybridAggregateTransform
from train import define_clf, define_clf_hybrid

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
import wandb


class SharedEvaluation(BaseEvaluation):
    def __init__(self, *args, eval_config=None, EA_in_eval=False, len_run=None, wandb_params=None, **kwargs):
        super(SharedEvaluation, self).__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.len_run = len_run
        self.EA_in_eval = EA_in_eval

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def evaluate(self, dataset, pipelines, grid_search):

        # Get data
        init_time = time()
        X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)
        print(f"(1) Data got {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)
        print(f"(2) Encoded {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Extract metadata
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

                # Fit and update progress
                t_start = time()
                copyclf = deepcopy(clf)
                # TODO : Uncommenting this line after testing HybridClassifier
                # model = copyclf.fit(X[train], y[train])
                model = copyclf.fit(X[train], None, Braindecode_dataset__labels=y[train],
                                    Braindecode_dataset__subject_groups=groups[train],
                                    Braindecode_dataset__info=X[train].info)
                duration = time() - t_start

                # Separate len_run*2 trials for test
                ix = test < (self.len_run * 2 + test[0])

                for p in list(model["Net"].module.shared_modules.parameters()):
                    if p.requires_grad:
                        p.requires_grad = False

                # Let's try another approach
                # TODO: Remove after test
                model["Net"].module.num_models = 1
                eval_classifier = define_clf_hybrid(deepcopy(model['Net'].module), self.eval_config,
                                                    experiment_name='Evaluation')
                if self.EA_in_eval:
                    create_dataset = HybridAggregateTransform(EA_len_run=self.len_run)
                else:
                    create_dataset = HybridAggregateTransform()
                eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + len(test)))

                eval_pipe['Net'].initialize()
                eval_pipe['Net'].module = deepcopy(model["Net"].module.shared_modules)
                eval_pipe['Net'].module_ = deepcopy(model["Net"].module.shared_modules)
                eval_pipe["Braindecode_dataset"].labels = y[test[ix_eval]]
                eval_pipe["Braindecode_dataset"].groups = groups[test[ix_eval]]
                eval_pipe["Braindecode_dataset"].info = X[test[ix_eval]].info
                X_trn = eval_pipe['Braindecode_dataset'].transform(X[test[ix_eval]])

                # Fix dimension and predict
                y_pred = eval_pipe['Net'].forward(X_trn).flatten(0, 1).argmax(dim=1)
                # Compute accuracy
                score = accuracy_score(y[test[ix_eval]], y_pred)
                print(score)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "dataset": dataset,
                    "subject": subject,
                    "session": 'session_E',
                    "fine-tuning": 'False',
                    "score": score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                }

                print(res)

                yield res
            #break


class EEGSharedEvaluation(BaseEvaluation):
    def __init__(self, *args, eval_config=None, EA_in_eval=False, len_run=None, wandb_params=None, **kwargs):
        super(EEGSharedEvaluation, self).__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.len_run = len_run
        self.EA_in_eval = EA_in_eval

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def evaluate(self, dataset, pipelines, grid_search):

        # Get data
        init_time = time()
        X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)
        print(f"(1) Data got {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)
        print(f"(2) Encoded {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Extract metadata
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

                # Fit and update progress
                t_start = time()
                copyclf = deepcopy(clf)
                model = copyclf.fit(X[train], y[train])
                duration = time() - t_start

                # Separate len_run*2 trials for test
                ix = test < (self.len_run * 2 + test[0])

                for p in list(model["Net"].module.parameters()):
                    if p.requires_grad:
                        p.requires_grad = False

                eval_classifier = define_clf(deepcopy(model['Net'].module), self.eval_config,
                                             warm_start=True, experiment_name='EEGClassifier')

                print(model['Net'].module)
                if self.EA_in_eval:
                    create_dataset = TransformaParaWindowsDatasetEA(self.len_run)
                else:
                    create_dataset = TransformaParaWindowsDataset()
                eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + len(test)))

                for p in list(eval_pipe["Net"].module.parameters()):
                    if p.requires_grad:
                        p.requires_grad = False

                model["Net"].module.requires_grad_(False)

                eval_pipe['Net'].initialize()
                eval_pipe['Net'].module = deepcopy(model["Net"].module)
                eval_pipe['Net'].module_ = deepcopy(model["Net"].module)
                eval_pipe["Braindecode_dataset"].classes_inferred_ = np.unique(to_numpy(y))
                eval_pipe["Braindecode_dataset"].y = y[test[ix]]

                score = _score(eval_pipe, X[test[ix]], y[test[ix]], scorer)

                print(score)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "dataset": dataset,
                    "subject": subject,
                    "session": 'session_E',
                    "fine-tuning": 'False',
                    "score": score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                }

                print(res)

                yield res
            #break
