from braindecode.datasets import create_from_X_y

from numpy import unique

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from alignment import euclidean_alignment
from dataset import split_runs_EA


class TransformaParaWindowsDataset(BaseEstimator, TransformerMixin):
    def __init__(self, kw_args=None):
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.y = y
        # self.X = euclidean_alignment(X.get_data())
        return self

    def transform(self, X, y=None):

        dataset = create_from_X_y(
            X=X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        # dataset_EA = preprocess(dataset,[Preprocessor(euclidean_alignment,apply_on_array=True)])

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class TransformaParaWindowsDatasetEA(BaseEstimator, TransformerMixin):
    def __init__(self, len_run, kw_args=None):
        self.kw_args = kw_args
        self.len_run = len_run

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        X_EA = split_runs_EA(X.get_data(), self.len_run)

        dataset = create_from_X_y(
            X=X_EA,
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class ClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, kw_args=None):
        self.clf = clf
        self.classes_ = None
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.clf.fit(X, y=y, **self.kw_args)
        self.classes_ = unique(y)

        return self.clf

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)