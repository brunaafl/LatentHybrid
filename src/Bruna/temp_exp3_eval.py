"""
Just a temporary file to present all the versions for experiment 3's evaluation.
When I decide which one is the optimal one, I'll delete ths file and add the func to the evaluation's file.

"""

import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from time import time

from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from tqdm import tqdm

from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit

from pipeline import TransformaParaWindowsDatasetEA
from dataset import split_runs_EA


def eval_exp3_v1(dataset, paradigm, pipes, run_dir):
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
    # for each test subject (Leave One Out)
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject"):

        # Test subject
        subject = groups[test[0]]

        for name, clf in pipes.items():

            cvclf = deepcopy(clf)
            # fit
            t_start = time()
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start

            # Save params
            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp3.pkl"),
            )

            # Predict on the test data
            # First, using 0 runs (zero-shot)
            ix = sessions[test] == 'session_E'
            X_test = X[test[ix]]
            y_test = y[test[ix]]
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
                "n_samples": len(train),
                "n_channels": nchan,
                "dataset": dataset.code,
                "pipeline": name,
            }
            results.append(res)

            # Save the scorer
            tftr0 = runs[test] == 'run_0'
            tfts = sessions[test] == 'session_T'
            test_ft_idx = np.logical_and(tftr0, tfts)

            # now, add the runs in the train set
            # for run in np.unique(runs):
            for k in range(len(np.unique(runs))):
                # runs to put in the training
                tftr = runs[test] == f"run_{k}"
                # find the session_T part of the run
                inter = np.logical_and(tfts, tftr)
                # find the union between the previous fine-tuning test
                test_ft_idx = np.logical_or(test_ft_idx, inter)

                # Compute train data
                train_idx = np.concatenate((train, test[test_ft_idx]))
                X_train = X[train_idx]
                y_train = y[train_idx]

                # Create a new model initialized with the saved params
                ftclf = deepcopy(cvclf)

                ftclf.load_params(
                    f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                    f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                    f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                    f_optimizer=str(run_dir / f"final_model_optimizer_{subject}._exp3.pkl"), )

                # Freeze some layers
                ftclf['Net'].module_.conv_temporal.weight.requires_grad = False
                ftclf['Net'].module_.bnorm_temporal.weight.requires_grad = False
                ftclf['Net'].module_.conv_spatial.weight.requires_grad = False
                ftclf['Net'].module_.bnorm_1.weight.requires_grad = False
                ftclf['Net'].module_.conv_separable_depth.weight.requires_grad = False
                ftclf['Net'].module_.conv_separable_point.weight.requires_grad = False
                ftclf['Net'].module_.bnorm_2.weight.requires_grad = False

                # Fit with new train data
                t_start = time()
                ftmodel = ftclf.fit(X_train, y_train)
                duration = time() - t_start

                # Predict on the test set
                score = _score(ftmodel, X_test, y_test, scorer)

                res = {
                    "subject": groups[test[0]],
                    "n_test_runs": k + 1,
                    "test_session": 'session_E',
                    "score": score,
                    "time": duration,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }
                results.append(res)

    results = pd.DataFrame(results)
    return results


def eval_exp3_v2(dataset, paradigm, pipes, run_dir, nn_model, online=False):
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

        subject = groups[test[0]]

        for name, clf in pipes.items():

            # Create the new model and initialize it
            cvclf = deepcopy(clf)
            # fit
            t_start = time()
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start

            # Save params
            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp3.pkl"),
            )
            # Predict on the test data
            # First, using 0 runs
            ix = sessions[test] == 'session_E'
            X_test = X[test[ix]]
            y_test = y[test[ix]]
            score = _score(model, X_test, y_test, scorer)
            print(score)

            nchan = (
                X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
            )

            res = {
                "subject": groups[test[0]],
                "n_test_runs": 0,
                "test_session": 'session_E',
                "score": score,
                "time": duration,
                "n_samples": len(train),
                "n_channels": nchan,
                "dataset": dataset.code,
                "pipeline": name,
            }
            results.append(res)

            # Save the scorer
            tftr0 = runs[test] == 'run_0'
            tfts = sessions[test] == 'session_T'
            test_ft_idx = np.logical_and(tftr0, tfts)

            len_run = sum(test_ft_idx * 1)

            X_test = X_test.get_data()

            # now, add the runs in the train set
            for k in range(len(np.unique(runs))):

                # runs to put in the training
                tftr = runs[test] == f"run_{k}"
                # find the session_T part of the run
                inter = np.logical_and(tfts, tftr)
                # find the union between the previous fine-tuning test
                test_ft_idx = np.logical_or(test_ft_idx, inter)

                # Compute train data
                train_idx = np.concatenate((train, test[test_ft_idx]))
                X_train = X[train_idx].get_data()
                y_train = y[train_idx]

                # Create a new model initialized with the saved params
                cuda = (
                    torch.cuda.is_available()
                )  # check if GPU is available, if True chooses to use it
                device = "cuda" if cuda else "cpu"
                if cuda:
                    torch.backends.cudnn.benchmark = True

                ftclf = EEGClassifier(
                    deepcopy(nn_model),
                    criterion=torch.nn.NLLLoss,
                    optimizer=torch.optim.AdamW,
                    train_split=ValidSplit(0.20, random_state=42),  # using valid_set for validation
                    optimizer__lr=0.0125 * 0.01,
                    optimizer__weight_decay=0,
                    batch_size=64,
                    max_epochs=50,
                    callbacks=[EarlyStopping(monitor='valid_loss', patience=50),
                               EpochScoring(scoring='roc_auc', on_train=True, name='train_acc', lower_is_better=False),
                               EpochScoring(scoring='roc_auc', on_train=False, name='valid_acc',
                                            lower_is_better=False)],
                    device=device,
                    verbose=1,
                )

                ftclf.initialize()

                # Initialize with the saved parameters
                ftclf.load_params(
                    f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                    f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                    f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                    f_optimizer=str(run_dir / f"final_model_optimizer_{subject}._exp3.pkl"), )

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
                    X_train = split_runs_EA(X_train, len_run)

                    if not online:
                        X_test = split_runs_EA(X_test, len_run)

                # Fit
                t_start = time()
                ftmodel = ftclf.fit(X_train, y_train)
                duration = time() - t_start

                # Predict on the test set
                score = _score(ftmodel, X_test, y_test, scorer)
                print(score)

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


def eval_exp3_v3(dataset, paradigm, pipes, run_dir, nn_model):
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

        subject = groups[test[0]]

        for name, clf in pipes.items():

            # Create the new model and initialize it
            cvclf = deepcopy(clf)
            # fit
            t_start = time()
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start

            # Save params
            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp3.pkl"),
            )

            # Predict on the test data
            # First, using 0 runs
            ix = sessions[test] == 'session_E'
            X_test = X[test[ix]]
            y_test = y[test[ix]]
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
                "n_samples": len(train),
                "n_channels": nchan,
                "dataset": dataset.code,
                "pipeline": name,
            }
            results.append(res)

            # Save the scorer
            tftr0 = runs[test] == 'run_0'
            tfts = sessions[test] == 'session_T'
            test_ft_idx = np.logical_and(tftr0, tfts)

            # now, add the runs in the train set

            for k in range(len(np.unique(runs))):
                # runs to put in the training
                tftr = runs[test] == f"run_{k}"
                # find the session_T part of the run
                inter = np.logical_and(tfts, tftr)
                # find the union between the previous fine-tuning test
                test_ft_idx = np.logical_or(test_ft_idx, inter)

                # Compute train data
                train_idx = np.concatenate((train, test[test_ft_idx]))
                X_train = X[train_idx]
                y_train = y[train_idx]

                # Create a new model initialized with the saved params
                cuda = (
                    torch.cuda.is_available()
                )  # check if GPU is available, if True chooses to use it
                device = "cuda" if cuda else "cpu"
                if cuda:
                    torch.backends.cudnn.benchmark = True

                ftclf = EEGClassifier(
                    deepcopy(nn_model),
                    criterion=torch.nn.NLLLoss,
                    optimizer=torch.optim.AdamW,
                    train_split=ValidSplit(0.20, random_state=42),  # using valid_set for validation
                    optimizer__lr=0.0125 * 0.01,
                    optimizer__weight_decay=0,
                    batch_size=64,
                    max_epochs=100,
                    callbacks=[EarlyStopping(monitor='valid_loss', patience=50),
                               EpochScoring(scoring='roc_auc', on_train=True, name='train_acc', lower_is_better=False),
                               EpochScoring(scoring='roc_auc', on_train=False, name='valid_acc',
                                            lower_is_better=False)],
                    device=device,
                    verbose=1,
                )

                ftclf.initialize()

                # Initialize with the saved parameters
                ftclf.load_params(
                    f_params=str(run_dir / f"final_model_params_{subject}_exp3.pkl"),
                    f_history=str(run_dir / f"final_model_history_{subject}_exp3.json"),
                    f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp3.pkl"),
                    f_optimizer=str(run_dir / f"final_model_optimizer_{subject}._exp3.pkl"), )

                create_dataset = pipes[name][0]
                ftpipe = Pipeline([("Braindecode_dataset", create_dataset),
                                   ("Net", clone(ftclf))])

                # Freeze some layers
                ftpipe['Net'].module.conv_temporal.weight.requires_grad = False
                ftpipe['Net'].module.bnorm_temporal.weight.requires_grad = False
                ftpipe['Net'].module.conv_spatial.weight.requires_grad = False
                ftpipe['Net'].module.bnorm_1.weight.requires_grad = False
                ftpipe['Net'].module.conv_separable_depth.weight.requires_grad = False
                ftpipe['Net'].module.conv_separable_point.weight.requires_grad = False
                ftpipe['Net'].module.bnorm_2.weight.requires_grad = False

                # Fit
                t_start = time()
                ftmodel = ftpipe.fit(X_train, y_train)
                duration = time() - t_start
                print(duration)

                # Predict on the test set
                score = _score(ftmodel, X_test, y_test, scorer)
                print(score)

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

        break

    results = pd.DataFrame(results)
    return results

