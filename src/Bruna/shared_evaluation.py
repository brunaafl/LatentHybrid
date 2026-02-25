from skorch.utils import to_tensor, to_numpy, to_device
import warnings
from pathlib import Path

import moabb
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
from alignment import euclidean_alignment
from train import define_clf, define_clf_hybrid

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
import wandb

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


class EEGSharedEvaluation(BaseEvaluation):
    def __init__(self, *args, eval_config=None, EA_in_eval=False, len_run=None, online='off', **kwargs):
        super(EEGSharedEvaluation, self).__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.len_run = len_run
        self.EA_in_eval = EA_in_eval
        self.online = online

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):

        # Get data
        init_time = time()
        X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)
        print(f"(1) Data got {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"(2) Encoded {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        # Extract metadata
        groups = metadata.subject.values
        n_subjects = len(dataset.subject_list)
        scorer = get_scorer(self.paradigm.scoring)

        print(f"(3) Setup done {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

        cv = LeaveOneGroupOut()

        # Progressbar at subject level
        subject_num = 0

        for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject"):
            subject = groups[test[0]]

            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

            # iterate over pipelines
            for name, clf in run_pipes.items():

                # Fit and update progress
                t_start = time()
                copyclf = deepcopy(clf)
                copyclf['Net'].classes = [0, 1]

                model = copyclf.fit(X[train], y[train])
                duration = time() - t_start

                # Get sampling freq from the trained model
                sfreq = model['Braindecode_dataset'].sfreq

                # Define indices for Calibration and Evaluation
                # Separate len_run*2 trials for calibration
                ix_calib = test < (self.len_run * 2 + test[0])
                # Evaluation indices
                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + len(test)))

                # Handle Online Adaptation
                if self.online == 'on' and self.EA_in_eval:
                    # FIX: Use set_params to replace the pipeline step instead of direct assignment
                    model.set_params(Braindecode_dataset=TransformaParaWindowsDataset(sfreq))

                    # Perform Euclidean Alignment
                    _, r = euclidean_alignment(X[test[ix_calib]].get_data()[:self.len_run])
                    X_eval = np.matmul(r, X[test[ix_eval]].get_data())
                else:
                    X_eval = X[test[ix_eval]]

                # Prepare model for inference
                model['Net'].module.eval()

                # Update state of the transformer (whether original or replaced)
                model["Braindecode_dataset"].classes_inferred_ = np.unique(to_numpy(y))
                model["Braindecode_dataset"].y = y[test[ix_eval]]

                score = _score(model, X_eval, y[test[ix_eval]], scorer, score_params=None)

                print(score)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "dataset": dataset,
                    "subject": subject,
                    "session": 'both',
                    "fine-tuning": 'False',
                    "score": score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                }

                print(res)

                yield res
