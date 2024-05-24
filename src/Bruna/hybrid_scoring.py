import torch

from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.callbacks import EpochScoring
from skorch.dataset import unpack_data


class HybridScoring(EpochScoring):
    def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
        self.y_preds_ = []
        self.y_trues_ = []
        print('epoch begin')
        print(net.module.num_models)
        for subject_i in range(net.module.num_models):
            self.y_preds_.append([])
        self.tag = False

    def on_batch_end(
            self, net, batch, y_pred, training, **kwargs):
        if not self.use_caching or training != self.on_train:
            return
        print('batch end')
        _X, y = unpack_data(batch)
        print(y_pred)
        print(y)
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
        #print(dataset_train)
        #print(dataset_valid)
        print('epoch end')
        print(len(y_pred))
        print(len(y_test))
        unwrapped_y_pred = []
        for subject_i in range(net.module.num_models):
            unwrapped_y_pred.append(torch.vstack(y_pred[subject_i]))

        with _cache_net_forward_iter(net, self.use_caching, unwrapped_y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)
