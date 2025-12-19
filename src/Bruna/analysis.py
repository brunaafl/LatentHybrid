import numpy as np
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance

def check_metric(metric, expected_keys=["mean", "distance"]):
    if isinstance(metric, str):
        return [metric] * len(expected_keys)

    elif isinstance(metric, dict):
        if not all(k in metric.keys() for k in expected_keys):
            raise KeyError(
                f"metric must contain {expected_keys}, but got {metric.keys()}"
            )

        return [metric[k] for k in expected_keys]

    else:
        raise TypeError(f"metric must be str or dict, but got {type(metric)}")


def class_distinctiveness(X, y, exponent=1, metric="riemann",
                          return_num_denom=False):
    """ FROM PYRIEMANN """

    metric_mean, metric_dist = check_metric(metric)
    classes = np.unique(y)
    if len(classes) <= 1:
        raise ValueError("y must contain at least two classes")

    means = np.array([
        mean_covariance(X[y == c], metric=metric_mean) for c in classes
    ])

    if len(classes) == 2:
        num = distance(means[0], means[1], metric=metric_dist) ** exponent
        denom = 0.5 * _get_within(X, y, means, classes, exponent, metric_dist)

    else:
        mean_all = mean_covariance(means, metric=metric_mean)
        dists_between = [
            distance(m, mean_all, metric=metric_dist) ** exponent
            for m in means
        ]
        num = np.sum(dists_between)
        denom = _get_within(X, y, means, classes, exponent, metric_dist)

    class_dis = num / denom

    if return_num_denom:
        return class_dis, num, denom
    else:
        return class_dis


def _get_within(X, y, means, classes, exponent, metric):
    """Private function to compute within dispersion."""
    sigmas = []
    for ic, c in enumerate(classes):
        dists_within = [
            distance(x, means[ic], metric=metric) ** exponent
            for x in X[y == c]
        ]
        sigmas.append(np.mean(dists_within))
    sum_sigmas = np.sum(sigmas)
    return sum_sigmas