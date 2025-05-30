import torch
import wandb

import numpy as np

from braindecode import EEGClassifier
from prompt_toolkit.contrib.regular_languages.regex_parser import Lookahead

from sklearn.metrics import accuracy_score
from skorch.dataset import ValidSplit, unpack_data
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, GradientNormClipping, Checkpoint, WandbLogger
from skorch.utils import to_tensor
from torch import nn

from hybrid_scoring import HybridScoring
from alignment_loss import JointAlignmentLoss


# Class adapted to the normal Shared model for testing purposes
class HybridClassifier(EEGClassifier):

    def __init__(self, lambd=0.001, *args, **kwargs):
        super(HybridClassifier, self).__init__(*args, **kwargs)
        self.lambd = lambd

    def get_loss(self, y_pred, y_true, *args, **kwargs):

        # y_pred is a tuple with (out, feat)
        y_pred, feature = y_pred
        y_true = to_tensor(y_true, device=self.device)
        losses = []

        for subject in range(y_pred.shape[0]):

            # I actually want the features before fc, or just after unique
            y_slice = torch.select(y_pred, 0, subject)
            feat_slice = torch.select(feature, 0, subject)

            if y_pred.requires_grad:
                y_slice.retain_grad()
            if feature.requires_grad:
                feat_slice.retain_grad()

            # FOr JointAlignmentLoss
            loss = self.criterion_(feat_slice, y_slice, y_true[:, subject])
            # For normal NLLLoss
            #loss = self.criterion_(y_slice, y_true[:, subject])

            losses.append(loss)

        loss = sum(losses) / self.module.num_models

        return loss

def get_subject_acc_scorer(subject):
    def scoring_for_subject_i(model, x, y_true):
        # Adapt here to deal with (out,feat) tuple
        out = list(model.forward_iter())
        # out, _ = zip(*results)  # Unpack the results
        # out = torch.cat(out, dim=0)  # Concatenate each output type
        y_preds = [z for z in out]

        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        return accuracy_score(true_slice, predictions)

    return scoring_for_subject_i

def get_subject_loss_scorer(subject, criterion):
    def scoring_for_subject_i(model, x, y_true):
        # Adapt here to deal with (out,feat) tuple
        out = list(model.forward_iter())
        # out, _ = zip(*results)  # Unpack the results
        # out = torch.cat(out, dim=0)  # Concatenate each output type
        y_preds = [z for z in out]

        true_slice = to_tensor(y_true[:, subject], device=model.device)
        loss = criterion(y_preds[subject], true_slice)
        return loss

    return scoring_for_subject_i

def average_acc_scoring(model, x, y_true):
    # Adapt here to deal with (out,feat) tuple
    out = list(model.forward_iter())
    # print(len(results))
    # out, _ = results[0] # Unpack the results
    # print(len(out))
    # out = torch.cat(out, dim=0) # Concatenate each output type
    # print(out.shape)
    y_preds = [z for z in out]
    # print(len(y_preds))

    accuracies_per_subject = []
    for subject in range(len(y_preds)):
        subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
        true_slice = y_true[:, subject]
        predictions = np.argmax(subject_slice, axis=1)
        accuracies_per_subject.append(accuracy_score(true_slice, predictions))
    return sum(accuracies_per_subject) / len(accuracies_per_subject)


def define_hybrid_clf(model, config, experiment_name, feat_dim=(16,1,251), n_centers=2):
    """
    Transform the pytorch model into classifier object to be used in the training
    Parameters
    ----------
    experiment_name
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
        module=model,
        criterion=JointAlignmentLoss,
        #criterion=nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        batch_size=batch_size,
        max_epochs=config.train.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   GradientNormClipping(gradient_clip_value=1),
                   WandbLogger(wandb.run),
                   Checkpoint(monitor="valid_loss_best", load_best=True,
                              dirname=f"/root/params/temptrain-{experiment_name}", f_params="params.pt"),
                   HybridScoring(scoring=average_acc_scoring, on_train=True, name='avg_train_acc',
                                 lower_is_better=False),
                   HybridScoring(scoring=average_acc_scoring, on_train=False, name='avg_valid_acc',
                                 lower_is_better=False),
                   lrscheduler,] + scoring_callbacks,
        device=device,
        verbose=1,
        warm_start=True,
    )
    return clf