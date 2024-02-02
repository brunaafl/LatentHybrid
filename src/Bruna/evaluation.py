import logging
from typing import Union

from moabb.evaluations.base import BaseEvaluation

import mne
import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from time import time

from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit

from pipeline import TransformaParaWindowsDatasetEA
from dataset import split_runs_EA
from alignment import euclidean_alignment

mne.set_log_level(False)
log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class CrossCrossSubjectEvaluation(BaseEvaluation):
    """
    Temporary name!
    I want to create the evaluation that I did in the experiment 4. For this, I created one
    classifier per subject, and evaluated using the other ones.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    """

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        # this is a bit akward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(self.results.not_yet_computed(pipelines, dataset, subject))
        if len(run_pipes) != 0:

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset,
                                                    return_epochs=self.return_epochs)

            # encode labels
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)

            # extract metadata
            groups = metadata.subject.values
            sessions = metadata.session.values
            n_subjects = len(dataset.subject_list)

            scorer = get_scorer(self.paradigm.scoring)

            # perform leave one subject out CV
            cv = LeaveOneGroupOut()
            # Progressbar at subject level
            model_list = []
            for test, train in tqdm(
                    cv.split(X, y, groups),
                    total=n_subjects,
                    desc=f"{dataset.code}-CrossCrossSubject"):

                # aux = np.unique(groups[test])
                # subj_0 = aux[0]
                run_pipes = self.results.not_yet_computed(pipelines, dataset, groups[train[0]])

                for name, clf in run_pipes.items():
                    t_start = time()
                    model = deepcopy(clf).fit(X[train], y[train])
                    duration = time() - t_start
                    model_list.append(model)

                    # for each test subject
                    for subject in np.unique(groups[test]):
                        # Now evaluate
                        ix = groups[test] == subject
                        score = _score(model, X[test[ix]], y[test[ix]], scorer)
                        session = 'both'
                        nchan = (
                            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                        )
                        res = {
                            "time": duration,
                            "dataset": dataset,
                            "subject": groups[train[0]],
                            "test": subject,
                            "session": session,
                            "score": score,
                            "n_samples": len(train),
                            "n_channels": nchan,
                            "pipeline": name,
                        }

                        yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1


def add_test_column(dataset, results):
    subj_list = dataset.subject_list
    list_ = []
    for i in subj_list:
        subj_copy = deepcopy(subj_list)
        subj_copy.remove(i)
        list_.append(subj_copy)

    array = np.array(list_)
    test = array.flatten()
    results.insert(4, 'test', test, True)
    return results


def eval_exp2(dataset, paradigm, pipes):
    """

    Cross subject evaluation with different sizes of training set.
    For each test subject, at each assay, we have chosen a percentage a=k/n for each test subject j,
    where n is the total number of runs per subject of the dataset and k=1,…, n, from the total number
    of train data using all N-1 subjects.

    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df

    """

    X, y, metadata = paradigm.get_data(dataset, return_epochs=True)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    results = list()
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CSTrainSize"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            train_idx = runs[train] == 'run_0'
            runs_list = np.unique(runs[train])
            runs_idx = list(range(len(runs_list)))

            # MAYBE it could be interesting to sort the runs instead of use the order
            for r in runs_idx:
                tr = runs[train] == f"run_{r}"
                train_idx = np.logical_or(train_idx, tr)

                t_start = time()
                model = deepcopy(clf).fit(X[train[train_idx]], y[train[train_idx]])
                duration = time() - t_start

                session = 'both'

                # I don't think we need to divide in sessions
                # ix = sessions[test] == session
                score = _score(model, X[test], y[test], scorer)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "subject": subject,
                    "n_train_runs": r + 1,
                    "session": session,
                    "score": score,
                    "time": duration,
                    "n_samples": len(train[train_idx]),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }

                results.append(res)

    results = pd.DataFrame(results)

    return results


def eval_exp4(dataset, paradigm, pipes, run_dir):
    """

    Create one model per subject and the with the others

    :param run_dir:
    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df
             model_list: list of clf

    """
    X, y, metadata = paradigm.get_data(dataset, return_epochs=True)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    results = []
    model_list = []
    # for each test subject
    for test, train in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-IndividualModels"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            cvclf = deepcopy(clf)
            t_start = time()
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start
            model_list.append(model)

            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp4.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp4.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp4.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp4.pkl"),
            )

            # for each test subject
            for subj in np.unique(groups[test]):
                # Now evaluate
                ix = groups[test] == subj
                score = _score(model, X[test[ix]], y[test[ix]], scorer)
                session = 'both'
                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "subject": groups[train[0]],
                    "test": subj,
                    "session": session,
                    "score": score,
                    "time": duration,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }

                results.append(res)

    results = pd.DataFrame(results)

    return results, model_list


def eval_exp3(dataset, paradigm, pipes, run_dir, nn_model, use_ses='both', online=False):
    X, y, metadata = paradigm.get_data(dataset, return_epochs=True)

    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    results = []
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject"):

        subject = int(groups[test[0]])

        for name, clf in pipes.items():

            if use_ses in np.unique(sessions):
                ses = sessions[train] == use_ses
                t_idx = train[ses]
                X_t = X[t_idx]
                y_t = y[t_idx]
            else:
                t_idx = train
                X_t = X[t_idx]
                y_t = y[t_idx]

            len_train = len(y_t)
            # Create the new model and initialize it
            cvclf = deepcopy(clf)
            # fit
            t_start = time()
            model = cvclf.fit(X_t, y_t)
            duration = time() - t_start

            # Save params
            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp3.pkl"),
            )

            # Define test set
            ix = sessions[test] == 'session_E'
            X_test = X[test[ix]]
            y_test = y[test[ix]]

            # Predict on the test data if offline
            # If online and EA, you have no data to align the test here
            if not online:
                # First, using 0 runs
                score = _score(model, X_test, y_test, scorer)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )

                res = {
                    "subject": groups[test[0]],
                    "n_test_runs": 0,
                    "test_session": 'session_E',
                    "score": score,
                    "time": duration,
                    "n_samples": len_train,
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }
                results.append(res)

            # Prepare for fine tuning
            tftr0 = runs[test] == 'run_0'
            tfts = sessions[test] == 'session_T'
            test_ft_idx = np.logical_and(tftr0, tfts)

            len_run = sum(test_ft_idx * 1)

            X_test = X_test.get_data()

            # now, add the runs in the train set
            for k in range(len(np.unique(runs))):

                X_test_ = X_test

                # runs to put in the training
                tftr = runs[test] == f"run_{k}"
                # find the session_T part of the run
                inter = np.logical_and(tfts, tftr)
                # find the union between the previous fine-tuning test
                test_ft_idx = np.logical_or(test_ft_idx, inter)

                # Compute train data
                train_idx = np.concatenate((t_idx, test[test_ft_idx]))
                X_train = X[train_idx].get_data()
                y_train = y[train_idx]

                # Create a new model initialized with the saved params
                ftclf = create_clf_ft(nn_model, 100)

                ftclf.initialize()

                # Initialize with the saved parameters
                ftclf.load_params(
                    f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                    f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                    f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                    f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp3.pkl"), )

                # Freeze some layers
                ftclf.module_.conv_temporal.weight.requires_grad = False
                ftclf.module_.bnorm_temporal.weight.requires_grad = False
                ftclf.module_.conv_spatial.weight.requires_grad = False
                ftclf.module_.bnorm_1.weight.requires_grad = False
                ftclf.module_.conv_separable_depth.weight.requires_grad = False
                ftclf.module_.conv_separable_point.weight.requires_grad = False
                ftclf.module_.bnorm_2.weight.requires_grad = False

                # Euclidean Alignment if needed
                if type(pipes[name][0]) == type(TransformaParaWindowsDatasetEA(len_run=len_run)):
                    if online:
                        # Align train subjects
                        X_train[:len_train] = split_runs_EA(X_train[:len_train], len_run)
                        # Align all aux data (fine-tuning data)
                        X_train[len_train:], r_op = euclidean_alignment(X_train[len_train:])
                        # Use ref matrix to align test data
                        X_test_ = np.matmul(r_op, X_test)

                    else:
                        # Align normally train and test
                        X_train = split_runs_EA(X_train, len_run)
                        X_test_ = split_runs_EA(X_test, len_run)

                # Fit
                t_start = time()
                ftmodel = ftclf.fit(X_train, y_train)
                duration = time() - t_start

                # Predict on the test set
                score = _score(ftmodel, X_test_, y_test, scorer)
                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "subject": groups[test[0]],
                    "n_test_runs": k + 1,
                    "test_session": 'session_E',
                    "score": score,
                    "time": duration,
                    "n_samples": len(y_train),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }
                results.append(res)

    results = pd.DataFrame(results)
    return results


def create_clf_ft(model, max_epochs):
    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True

    ftclf = EEGClassifier(
        deepcopy(model),
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(0.20, random_state=42),  # using valid_set for validation
        optimizer__lr=0.0125 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=50),
                   EpochScoring(scoring='roc_auc', on_train=True, name='train_acc', lower_is_better=False),
                   EpochScoring(scoring='roc_auc', on_train=False, name='valid_acc',
                                lower_is_better=False)],
        device=device,
        verbose=1,
    )

    ftclf.initialize()

    return ftclf
