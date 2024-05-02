import torch

import numpy as np
from sklearn.metrics import accuracy_score

from skorch.dataset import ValidSplit, unpack_data
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, GradientNormClipping, Checkpoint, WandbLogger
from skorch.utils import to_tensor

from braindecode import EEGClassifier

import wandb

from src.Bruna.hybrid_scoring import HybridScoring


class HybridClassifier(EEGClassifier):

    def get_loss(self, y_pred, y_true, *args, **kwargs):

        y_true = to_tensor(y_true, device=self.device)
        losses = []
        for subject in range(y_pred.shape[0]):
            subject_slice = torch.select(y_pred, 0, subject)
            if y_pred.requires_grad:
                subject_slice.retain_grad()
            loss = self.criterion_(subject_slice, y_true[:, subject])
            losses.append(loss)

        loss = sum(losses) / self.module.num_models

        # make_dot(y_pred, show_attrs=True, params=dict(self.module.named_parameters())).render("model", format="svg")
        return loss

    """
    def get_loss(self, y_pred, y_true, *args, **kwargs):

        y_true = to_tensor(y_true, device=self.device)
        y_pred_flat = []
        for subject in range(y_pred.shape[0]):
            subject_slice = torch.select(y_pred, 0, subject)
            if y_pred.requires_grad:
                subject_slice.retain_grad()
            y_pred_flat.append(subject_slice)
        y_pred_flat = torch.cat(y_pred_flat, dim=0)
        y_true_flat = torch.flatten(y_true)
        loss = self.criterion_(y_pred_flat, y_true_flat)

        # make_dot(y_pred, show_attrs=True, params=dict(self.module.named_parameters())).render("model", format="svg")
        return loss
    """


def get_subject_acc_scorer(subject):
    def scoring_for_subject_i(model, x, y_true):
        out = model.forward_iter()
        y_preds = [z for z in out]
        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        return accuracy_score(true_slice, predictions)

    return scoring_for_subject_i


def get_subject_loss_scorer(subject, criterion):
    def scoring_for_subject_i(model, x, y_true):
        out = model.forward_iter()
        y_preds = [z for z in out]
        true_slice = to_tensor(y_true[:, subject], device=model.device)
        loss = criterion(y_preds[subject], true_slice)
        return loss

    return scoring_for_subject_i


def average_acc_scoring(model, x, y_true):
    out = model.forward_iter()
    y_preds = [z for z in out]
    accuracies_per_subject = []
    for subject in range(len(y_preds)):
        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        accuracies_per_subject.append(accuracy_score(true_slice, predictions))
    return sum(accuracies_per_subject) / len(accuracies_per_subject)


def define_hybrid_clf(model, config, experiment_name):
    """
    Transform the pytorch model into classifier object to be used in the training
    Parameters
    ----------
    model: pytorch model
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

    lrscheduler = LRScheduler(policy='CosineAnnealingLR', T_max=config.train.n_epochs)

    scoring_callbacks = [HybridScoring(scoring=get_subject_acc_scorer(i), on_train=False, name=f'{i:02d}_valid_acc',
                                       lower_is_better=False) for i in range(model.num_models)]

    clf = HybridClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=config.train.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   GradientNormClipping(gradient_clip_value=1),
                   WandbLogger(wandb.run),
                   Checkpoint(monitor="valid_loss_best", load_best=True,
                              dirname=f"/workspace/params/temptrain-{experiment_name}", f_params="params.pt"),
                   lrscheduler,
                   HybridScoring(scoring=average_acc_scoring, on_train=True, name='avg_train_acc',
                                 lower_is_better=False),
                   HybridScoring(scoring=average_acc_scoring, on_train=False, name='avg_valid_acc',
                                 lower_is_better=False)] + scoring_callbacks,
        device=device,
        verbose=1,
    )
    return clf
