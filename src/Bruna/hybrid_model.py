import copy
import math

import braindecode
import torch
from torch import nn

from braindecode.models import ShallowFBCSPNet, EEGNetv4, CTNet, AttentionBaseNet

from torch.nn import init

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

last_layer = {'EEGNet':19, 'CTNet':7, 'AttentionBaseNet':4}

def gen_slice_model(model,n_chans, n_classes, input_window_samples, config, start=0, end=19, remove_bn=False):

    last = last_layer[model]
    if start == 0 and end < last:
        drop_prob = config.model.drop_prob * 0.9
    else:
        drop_prob = config.model.drop_prob

    model_class = getattr(braindecode.models, model)

    temp_model = model_class(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
    )

    if end == last and start == 0:

        if remove_bn == 'LEA':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = LatentEuclideanAlignment()
        elif remove_bn == 'True':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = nn.Identity()

        return temp_model

    if start==0 and end == 0:
        return nn.Identity()

    net = list(temp_model.children())[start:end]
    if remove_bn == 'True':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = nn.Identity()
    elif remove_bn == 'LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = LatentEuclideanAlignment()
    elif remove_bn == 'One-bn':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = net[i]
                else:
                    net[i] = nn.Identity()
    elif remove_bn == 'One-LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = LatentEuclideanAlignment()
                else:
                    net[i] = nn.Identity()
    return nn.Sequential(*net)

def gen_slice_ABN(n_chans, n_classes, input_window_samples, config, start=0, end=4, remove_bn=False):

    # TODO: add attention next
    temp_model = AttentionBaseNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
    )

    if end == 6 and start == 0:

        if remove_bn == 'LEA':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = LatentEuclideanAlignment()
        elif remove_bn == 'True':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = nn.Identity()

        return temp_model

    if start==0 and end == 0:
        return nn.Identity()

    net = list(temp_model.children())[start:end]
    if remove_bn == 'True':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = nn.Identity()
    elif remove_bn == 'LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = LatentEuclideanAlignment()
    elif remove_bn == 'One-bn':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = net[i]
                else:
                    net[i] = nn.Identity()
    elif remove_bn == 'One-LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = LatentEuclideanAlignment()
                else:
                    net[i] = nn.Identity()
    return nn.Sequential(*net)


class CTNetEncoder(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.ensuredim = original_model.ensuredim
        self.cnn = original_model.cnn
        self.position = original_model.position
        self.trans = original_model.trans
        self.emb_size = original_model.emb_size

    def forward(self, x):
        x = self.ensuredim(x)
        cnn_out = self.cnn(x)
        cnn_out_scaled = cnn_out * math.sqrt(self.emb_size)
        pos_out = self.position(cnn_out_scaled)
        trans_out = self.trans(pos_out)
        features = cnn_out_scaled + trans_out
        return features

class CTNetClassif(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.flatten = original_model.flatten
        self.final_layer = original_model.final_layer

    def forward(self, x):
        x = self.flatten(x)
        out = self.final_layer(x)
        return out

def gen_slice_CTNet(n_chans, n_classes, input_window_samples, config, start=0, end=7, remove_bn=False):

    temp_model = CTNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
        heads=1,
        #drop_prob_posi=0.2
    )

    if end == 7 and start == 0:
        if remove_bn == 'LEA':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = LatentEuclideanAlignment()
        elif remove_bn == 'True':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = nn.Identity()
        return temp_model

    if start==0 and end == 0:
        return nn.Identity()

    if end<7:
        encoder = CTNetEncoder(temp_model)
        return encoder
    if start>0:
        clf = CTNetClassif(temp_model)
        return clf
    return None

def gen_slice_EEGNet(n_chans, n_classes, input_window_samples, config, start=0, end=19, remove_bn='False', ):
    # Maybe? Does it make any sense?
    # Justification: if we are putting the lr of the eval lower, maybe it would make sense if the drop was lower to help fitting

    # TODO: Test removing lowering dropout
    if start == 0 and end < 19:
        drop_prob = config.model.drop_prob * 0.9
    else:
        drop_prob = config.model.drop_prob

    temp_model = EEGNetv4(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=drop_prob
    )

    if end == 19 and start == 0:

        if remove_bn == 'LEA':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = LatentEuclideanAlignment()
        elif remove_bn == 'True':
            for i, module in enumerate(temp_model):
                if isinstance(temp_model[i], nn.BatchNorm2d):
                    temp_model[i] = nn.Identity()

        return temp_model

    if start==0 and end == 0:
        return nn.Identity()

    net = list(temp_model.children())[start:end]
    if remove_bn == 'True':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = nn.Identity()
    elif remove_bn == 'LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = LatentEuclideanAlignment()
    elif remove_bn == 'One-bn':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = net[i]
                else:
                    net[i] = nn.Identity()
    elif remove_bn == 'One-LEA':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = LatentEuclideanAlignment()
                else:
                    net[i] = nn.Identity()
    return nn.Sequential(*net)



model_gen = {
    "EEGNet": [gen_slice_EEGNet, 12, 12],
    "CTNet": [gen_slice_CTNet, 5, 5],
    "AttentionBaseNet": [gen_slice_ABN, 3, 3],
    "EEGNetShared": [gen_slice_EEGNet, 0, 0],
}

norms = {
    "Identity": nn.Identity,
    "BatchNorm2d": nn.LazyBatchNorm2d,
    "InstanceNorm2d": nn.LazyInstanceNorm2d,
}

class LatentEuclideanAlignment(nn.Module):

    # How does it backpropagate?????
    def inv_sqrtm(self, matrix, eps=1e-6):
        # Assume matrix is symmetric and positive semidefinite
        # Based on python project pytorch-sqrtm
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=eps)
        D_inv_sqrt = torch.diag(eigvals.rsqrt())
        return eigvecs @ D_inv_sqrt @ eigvecs.T

    def forward(self, model_input):
        if model_input.dim() != 3:
            model_input = model_input.squeeze()
        r = 0
        for i in range(len(model_input)):
            trial = model_input[i]
            cov = torch.cov(trial)
            r += cov
        r /= len(model_input)
        r_op = self.inv_sqrtm(r)

        return torch.matmul(r_op, model_input).unsqueeze(2)


class HybridModel(nn.Module):
    def __init__(self, num_models, model_type, n_chans, n_classes, input_window_samples, config=None, freeze='freeze',
                 args=None):
        super(HybridModel, self).__init__()
        self._args = (n_chans, n_classes, input_window_samples)
        self.config = config
        self.args = args
        self.num_models = num_models
        self.model_type = model_type
        self.shared_modules = model_gen[self.model_type][0](n_chans, n_classes, input_window_samples, config,
                                                            start=model_gen[self.model_type][2],
                                                            remove_bn=self.args.remove_bn)
        self.unique_modules = nn.ModuleList()
        self.freeze = freeze == "freeze"
        self.norm = nn.Identity()
        self.lea = self.args.remove_bn == 'LEA'
        if self.lea:
            self.aligner = LatentEuclideanAlignment()
        for model in range(num_models):
            self.unique_modules.append(self.init_unique_modules(*self._args))

    def init_unique_modules(self, n_chans, n_classes, input_window_samples):
        unique_head = model_gen[self.model_type][0](n_chans,
                                                    n_classes,
                                                    input_window_samples,
                                                    self.config,
                                                    end=model_gen[self.model_type][1],
                                                    remove_bn=self.args.remove_bn)
        return unique_head

    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    def forward(self, x):
        inputs = self.split_input(x)
        out, feat = [], []
        for i, model_input in enumerate(inputs):
            temp_unique = self.unique_modules[i](model_input)
            temp_norm = self.norm(temp_unique)
            temp_shared = self.shared_modules(temp_norm)
            out.append(temp_shared)
            feat.append(temp_unique)

        result = torch.stack(out)
        feat = torch.stack(feat)

        if result.requires_grad:
            result.retain_grad()
            feat.retain_grad()

        # Return features also
        return result.transpose(0, 1), feat.transpose(0, 1)

    def generate_branch_model(self, subj=None):
        if subj is None:
            new_layers = self.init_unique_modules(*self._args)
        else:
            new_layers = copy.deepcopy(self.unique_modules[subj])
        cloned_layers = copy.deepcopy(self.shared_modules)
        norm_clone = copy.deepcopy(self.norm)
        if self.freeze:
            cloned_layers.requires_grad_(False)
        return SpecializedModel(new_layers, norm_clone, cloned_layers, self.lea)


class SpecializedModel(nn.Module):
    def __init__(self, unique_modules, norm_clone, cloned_modules, lea, num_models=1):
        super(SpecializedModel, self).__init__()
        self.shared_modules = cloned_modules
        self.norm = norm_clone
        self.unique_modules = unique_modules
        self.num_models = num_models
        self.lea = lea
        if self.lea:
            self.aligner = LatentEuclideanAlignment()


    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    def forward(self, x):

        inputs = self.split_input(x)
        out, feat = [], []
        for i, model_input in enumerate(inputs):
            temp_unique = self.unique_modules(model_input)
            feat.append(temp_unique)
            temp_shared = self.shared_modules(temp_unique)
            out.append(temp_shared)

        result = torch.stack(out)

        feat = torch.stack(feat)
        if result.requires_grad:
            result.retain_grad()
            feat.retain_grad()

        #return result, feat
        return result.transpose(0, 1), feat.transpose(0, 1)

