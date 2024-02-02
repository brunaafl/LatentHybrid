"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)


def true_positive(y_true, y_pred):
    """
    Returning true positive.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    return confusion_matrix(y_true, y_pred)[0, 0]


def true_negative(y_true, y_pred):
    """
    Returning true negative.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    return confusion_matrix(y_true, y_pred)[1, 1]


def false_positive(y_true, y_pred):
    """
    Returning false positive.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    return confusion_matrix(y_true, y_pred)[1, 0]


def false_negative(y_true, y_pred):
    """
    Returning false negative.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    return confusion_matrix(y_true, y_pred)[0, 1]


def tp_n(y_true, y_pred):
    """
    True Positive Normalized
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    conf_ma = confusion_matrix(y_true, y_pred)
    conf_ma = conf_ma.astype("float") / conf_ma.sum(axis=1)[:, np.newaxis]
    return conf_ma[0, 0]


def tn_n(y_true, y_pred):
    """
    True Negative Normalized
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    conf_ma = confusion_matrix(y_true, y_pred)
    conf_ma = conf_ma.astype("float") / conf_ma.sum(axis=1)[:, np.newaxis]
    return conf_ma[1, 1]


def fp_n(y_true, y_pred):
    """
    False Positive Normalized
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    conf_ma = confusion_matrix(y_true, y_pred)
    conf_ma = conf_ma.astype("float") / conf_ma.sum(axis=1)[:, np.newaxis]
    return conf_ma[1, 0]


def fn_n(y_true, y_pred):
    """
    False Negative Normalized
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    """
    conf_ma = confusion_matrix(y_true, y_pred)
    conf_ma = conf_ma.astype("float") / conf_ma.sum(axis=1)[:, np.newaxis]
    return conf_ma[0, 1]


scoring = {
    "acc": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}
