from torch import nn
import pdb

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
from train import define_clf, define_clf_hybrid

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

                X_aux = X.get_data()

                """torchinfo.summary(model["Net"].module.unique_modules[0],
                                  input_size=(self.eval_config.train.batch_size, X_aux[0].shape[0], X_aux[0].shape[1]))
                torchinfo.summary(model["Net"].module.shared_modules, input_size=(self.eval_config.train.batch_size,  16, 1, 251))"""

                # Test set
                ix = test < (self.len_run * 2 + test[0])

                eval_model = model["Net"].module.generate_branch_model()
                eval_model.num_models = 1

                """torchinfo.summary(eval_model.unique_modules,
                                  input_size=(self.eval_config.train.batch_size, X_aux[0].shape[0], X_aux[0].shape[1]))
                torchinfo.summary(eval_model.shared_modules, input_size=(self.eval_config.train.batch_size,  16, 1, 251))"""

                eval_classifier = define_hybrid_clf(deepcopy(eval_model), self.eval_config, experiment_name='Evaluation')

                #eval_classifier = define_clf(deepcopy(eval_model), self.eval_config, experiment_name='Evaluation')
                if self.EA_in_eval:
                    create_dataset = HybridAggregateTransform(EA_len_run=self.len_run)
                    #create_dataset = TransformaParaWindowsDatasetEA(self.len_run)
                else:
                    create_dataset = HybridAggregateTransform()
                    #create_dataset = TransformaParaWindowsDataset()
                eval_pipe = Pipeline([("Braindecode_dataset", create_dataset), ("Net", eval_classifier)])

                # Evaluation set
                ix_eval = np.logical_and(test >= (self.len_run * 2 + test[0]),
                                         test < (test[0] + model["Hybrid_adapter"].n_trials_used))

                eval_run = active_wandb(self.wandb_params[0], self.eval_config, subject_num, train=False)

                for callback in eval_classifier.callbacks:
                    if isinstance(callback, WandbLogger):
                        callback.wandb_run = wandb.run

                if type(eval_model.unique_modules) != type(nn.Identity()) or \
                        list(eval_model.shared_modules.parameters())[0].requires_grad:

                    """eval_pipe['Net'].initialize()
                    eval_pipe['Net'].module.shared_modules = deepcopy(model["Net"].module.shared_modules)
                    eval_pipe['Net'].module_.shared_modules = deepcopy(model["Net"].module.shared_modules)"""

                    t_start = time()
                    """eval_clf = deepcopy(eval_pipe).fit(X[test[ix_eval]], None, Braindecode_dataset__labels=y[test[ix_eval]],
                                                       Braindecode_dataset__subject_groups=groups[test[ix_eval]],
                                                       Braindecode_dataset__info=X[test[ix_eval]].info)"""
                    pdb.set_trace()
                    eval_clf = deepcopy(eval_pipe).fit(X[test[ix_eval]], y[test[ix_eval]])
                    duration = duration + time() - t_start

                    print(X[test[ix_eval]].get_data().shape)
                    print(X[test[ix]].get_data().shape)

                    #eval_clf['Braindecode_dataset'].labels = y[test[ix_eval]]
                    #eval_clf['Braindecode_dataset'].groups = groups[test[ix_eval]]
                    #eval_clf['Braindecode_dataset'].info = X[test[ix_eval]].info

                    hsua = eval_clf['Braindecode_dataset'].transform(X[test[ix_eval]])
                    print('forward')
                    print(eval_clf['Net'].forward(hsua).shape)
                    #print(X[test[ix_eval]].get_data().shape)
                    #print(X[test[ix_eval]].get_data().shape)
                    #print(eval_clf.predict(X[test[ix_eval]]))
                    #print(eval_clf.predict(X[test[ix_eval]]))

                    #score = _score(eval_clf, X[test[ix_eval]], y[test[ix_eval]], scorer)

                else:

                    """eval_pipe["Braindecode_dataset"].labels = y[test[ix]]
                    eval_pipe["Braindecode_dataset"].groups = groups[test[ix]]
                    eval_pipe["Braindecode_dataset"].info = X[test[ix]].info"""

                    print(model["Net"].module.shared_modules)

                    print(eval_pipe["Net"].module.unique_modules)

                    torchinfo.summary(eval_pipe['Net'].module,
                                      input_size=(
                                      self.eval_config.train.batch_size, X_aux[0].shape[0], X_aux[0].shape[1]))

                    eval_pipe["Braindecode_dataset"].labels = y[train]
                    eval_pipe["Braindecode_dataset"].groups = groups[train]
                    eval_pipe["Braindecode_dataset"].info = X[train].info

                    for p in list(model["Net"].module.shared_modules.parameters()):
                        if p.requires_grad:
                            p.requires_grad = False

                    eval_pipe['Net'].initialize()
                    eval_pipe['Net'].module = deepcopy(model["Net"].module.shared_modules)
                    eval_pipe['Net'].module_ = deepcopy(model["Net"].module.shared_modules)

                    #eval_pipe['Braindecode_dataset'].y = y[train]
                    """score = _score(eval_pipe, X[test[ix]], y[test[ix]], scorer)"""
                    #pdb.set_trace()

                    #eval_classifier.classes_inferred_ = np.unique(to_numpy(y))
                    eval_pipe['Braindecode_dataset'].y = y[train]
                    Xproc = create_dataset.transform(X[train], y[train])
                    score = _score(eval_classifier, Xproc, y[train], scorer)

                    print(eval_pipe.predict_proba(X[train]))
                    print(X[train].get_data().shape)
                    print(eval_pipe)
                    score = _score(eval_pipe, X[train], y[train], scorer)
                    print(score)

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
            break


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
        project=f"{args.model}",
        group=config.train.experiment_name,
        name=f"{subject}-Shared:{args.sharednorm}-Unique:{args.uniquenorm}",
        config=wconfig
    )
    return run
