import torch

from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.callbacks import EpochScoring
from skorch.dataset import unpack_data


class HybridScoring(EpochScoring):
    def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
        self.y_preds_ = []
        self.y_trues_ = []
        for subject_i in range(net.module.num_models):
            self.y_preds_.append([])
        self.tag = False

    def on_batch_end(
            self, net, batch, y_pred, training, **kwargs):
        if not self.use_caching or training != self.on_train:
            return

        _X, y = unpack_data(batch)
        self.y_trues_.append(y)
        for subject_i in range(net.module.num_models):
            self.y_preds_[subject_i].append(torch.select(y_pred, 0, subject_i))

    def on_epoch_end(
            self,
            net,
            dataset_train,
            dataset_valid,
            **kwargs):
        X_test, y_test, y_pred = self.get_test_data(dataset_train, dataset_valid)

        unwrapped_y_pred = []
        for subject_i in range(net.module.num_models):
            unwrapped_y_pred.append(torch.vstack(y_pred[subject_i]))

        with _cache_net_forward_iter(net, self.use_caching, unwrapped_y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)
