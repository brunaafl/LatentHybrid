import copy

import torch

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset
from braindecode.models import EEGNetv4
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, Checkpoint, WandbLogger
from skorch.dataset import ValidSplit
from skorch.helper import predefined_split, SliceDataset

import random

def train(model, train_set, device, lr=0.0625 * 0.01, split=False, val_set=None):
    weight_decay = 0
    batch_size = 64
    n_epochs = 100

    if not split:
        train_split_par = None
    else:
        train_split_par = predefined_split(val_set)

    # Model (Classifier)
    clf = EEGClassifier(
        copy.deepcopy(model),
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=train_split_par,  # predefined_split(val_set) using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler",
                         LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            WandbLogger(None),
        ],
        device=device,
    )

    clf.initialize()

    clf = clone(clf)

    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the data.
    clf.fit(train_set, y=None, epochs=n_epochs)
    return clf


def define_clf(model, config):
    """
    Transform the pytorch model into classifier object to be used in the training
    Parameters
    ----------
    model: pytorch model
    device: cuda or cpu
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

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=config.train.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   EpochScoring(scoring='accuracy', on_train=True,
                                name='train_acc', lower_is_better=False),
                   EpochScoring(scoring='accuracy', on_train=False,
                                name='valid_acc', lower_is_better=False),
                   Checkpoint(monitor="valid_loss_best", load_best=True, dirname=f"/workspace/params/tempeval{random.randint(1,100000)}", f_params="params_{last_epoch[epoch]}.pt")],
        device=device,
        verbose=1,
    )
    return clf


def train_all_loo(model, Data_subjects, device, subject_ids, val_subj=None):
    loo = LeaveOneOut()
    list_s = list(range(len(subject_ids)))

    models_list = []
    predicts_list = []
    # Using Leave-One-Out validation
    for train_idx, test_idx in loo.split(list_s):
        train_idx = train_idx + 1
        test_idx = test_idx + 1
        print("Test subject:", test_idx[0])
        test_subj = test_idx[0]
        # Split in Train and Test
        Test = Data_subjects[f'{test_subj}']
        # Test_1,Test_2=Split_Train_Val(Test, val_subj=None)
        Train = BaseConcatDataset([Data_subjects[f'{i}'] for i in train_idx])

        # Split in Train and Validation IF WE WANT
        # Train, Val = split_train_val(Train_Aux, val_subj=val_subj)

        clf = train(copy.deepcopy(model), Train, device)

        y_pred = clf.predict(Test)
        y_true = list(SliceDataset(Test, 1))

        predicts_list.append((y_pred, y_true))
        models_list.append(clf)

        clf.save_params(
            f_params=f"final_model_params_{test_idx}.pkl",
            f_history=f"final_model_history_{test_idx}.json",
            f_criterion=f"final_model_criterion_{test_idx}.pkl",
            f_optimizer=f"final_model_optimizer_{test_idx}.pkl",
        )

    return models_list, predicts_list


def train_func(model, Train_data, Test_data, Val_data, device):
    clf = train(copy.deepcopy(model), Train_data, Val_data, device)
    y_pred_t = clf.predict(Test_data)
    y_true_t = list(SliceDataset(Test_data, 1))
    bac = roc_auc_score(y_true=y_true_t, y_pred=y_pred_t)

    return clf, bac


def init_model(n_chans, n_classes, input_window_samples, config):
    model = EEGNetv4(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )
    return model


def fine_tuning(model, device, subj, Test, Train_test):
    bac_runs = []
    for i in range(len(Train_test)):
        # model, device = init_model()

        # Then, we initialize a new model
        # val_set = Val
        n_epochs = 100
        weight_decay = 0
        lr = 0.0625 * 0.01

        clf_fine_tune = EEGClassifier(
            module=copy.deepcopy(model),
            train_split=None,  # predefined_split(val_set)
            callbacks=[
                "accuracy", ("lr_scheduler",
                             LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ],
            device=device,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay
        )

        clf_fine_tune.initialize()  # This is important!

        clf_fine_tune.load_params(
            f_params=f'final_model_params_{subj}.pkl',
            f_optimizer=f'final_model_optimizer_{subj}.pkl',
            f_history=f'final_model_history_{subj}.json',
            f_criterion=f"final_model_criterion_{subj}.pkl")

        # Fit the new model
        clf_fine_tune.fit(Train_test[i], y=None, epochs=60)

        y_pred2 = clf_fine_tune.predict(Test)
        y_true2 = list(SliceDataset(Test, 1))
        bac2 = roc_auc_score(y_true=y_true2, y_pred=y_pred2)
        bac_runs.append(bac2)
        print(f"With {i + 1} test runs in the train : ")
        print(f"  bac = {bac2}")

        return bac_runs