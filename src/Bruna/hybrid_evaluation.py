from torch import nn

from skorch.callbacks import WandbLogger
from skorch.utils import to_numpy

import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
)
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer

from sklearn.pipeline import Pipeline

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time

from copy import deepcopy

from mne.epochs import BaseEpochs

from train import define_clf

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA

import wandb


class HybridEvaluation(BaseEvaluation):
    def __init__(self, *args, run_dir=None, eval_config=None, EA_in_eval=False, len_run=None, wandb_params=None,
                 **kwargs):
        super(HybridEvaluation, self).__init__(*args, **kwargs)
        self.eval_config = eval_config
        self.EA_in_eval = EA_in_eval
        self.len_run = len_run
        self.wandb_params = wandb_params
        self.run_dir = run_dir

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

                # Start wandb monitoring
                t_start = time()
                copyclf = deepcopy(clf)
                subject_num += 1
                train_run = active_wandb(*self.wandb_params, subject_num, train=True)
                for callback in copyclf['Net'].callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                # Fit
                model = copyclf.fit(X[train], None, Hybrid_adapter__labels=y[train],
                                    Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)
                wandb.finish()

                duration = time() - t_start

                ix = test < (self.len_run * 2 + test[0])

                eval_model = model["Net"].module.generate_branch_model()
                eval_classifier = define_clf(eval_model, self.eval_config, warm_start=True)
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
                    eval_clf = eval_pipe.fit(X[test[ix]], y[test[ix]])  # X[test[ix]], y[test[ix]]
                    create_dataset.y = y[test[ix]]
                    score = _score(eval_clf, X[test[ix]], y[test[ix]], scorer)
                else:
                    """copyclf['Net'].save_params(
                        f_params=str(run_dir / f"final_model_params_{subject}.pkl"),
                        f_history=str(run_dir / f"final_model_history_{subject}.json"),
                        f_criterion=str(run_dir / f"final_model_criterion_{subject}.pkl"),
                        f_optimizer=str(run_dir / f"final_model_optimizer_{subject}.pkl"),
                    )"""
                    eval_classifier.classes_inferred_ = np.unique(to_numpy(y))
                    create_dataset.y = y[train]
                    Xproc = create_dataset.transform(X[train], y[train])
                    score = _score(eval_classifier, Xproc, y[train], scorer)

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
            #break


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
        project=f"{args.type} EEG",
        group=config.train.experiment_name,
        name=f"{subject}-Shared:{args.sharednorm}-Unique:{args.uniquenorm}",
        config=wconfig
    )
    return run
