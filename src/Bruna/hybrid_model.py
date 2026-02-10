import copy

import torch
from torch import nn

from braindecode.models import  Deep4Net, ShallowFBCSPNet, EEGNetv4

from torch.nn import init

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from sqrtm import sqrtm
#from eegnet import EEGNetv4


class LazyLayerNorm(LazyModuleMixin, nn.LayerNorm):
    cls_to_become = nn.LayerNorm

    def __init__(self, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__(0, eps, elementwise_affine)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)


def initmod(module):
    init.xavier_uniform_(module.weight, gain=1)
    return module


def gen_slice_EEGNet_normtest(n_chans, n_classes, input_window_samples, config, start=0, end=19, norm=nn.BatchNorm2d):
    temp_model = EEGNetv4(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == 0:
        return nn.Identity()

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    net = list(temp_model.children())[start:end]
    if norm != nn.BatchNorm2d:
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = norm()
    return nn.Sequential(*net)


def gen_slice_DeepNet(n_chans, n_classes, input_window_samples, config, start=0, end=29, remove_bn=False):
    temp_model = Deep4Net(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    if end == 0:
        return nn.Identity()

    if remove_bn:
        for i, module in enumerate(temp_model):
            if isinstance(temp_model[i], nn.BatchNorm2d):
                pass
            # temp_model[i] = nn.Identity()

    net = list(temp_model.children())[start:end]
    if not remove_bn:
        net.append(nn.ELU())

    return nn.Sequential(*net)


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


def gen_slice_ShallowNet(n_chans, n_classes, input_window_samples, config, start=0, end=29, drop_prob=0.5,
                         remove_bn=True, bn_end=True):
    temp_model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=config.model.drop_prob
    )

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    if end == 0:
        return nn.Identity()

    if remove_bn:
        for i, module in enumerate(temp_model):
            if isinstance(temp_model[i], nn.BatchNorm2d):
                pass

    net = list(temp_model.children())[start:end]

    return nn.Sequential(*net)


model_gen = {
    "DeepNet": [gen_slice_DeepNet, 8, 9],
    "EEGNet": [gen_slice_EEGNet, 12, 12],
    "ShallowNet": [gen_slice_ShallowNet, 4, 4],
    "ShallowNetShared": [gen_slice_ShallowNet, 0, 0],
    "EEGNetShared": [gen_slice_EEGNet, 0, 0],
    "DeepNetShared": [gen_slice_DeepNet, 0, 0],
    "EEGNetNormTest": [gen_slice_EEGNet_normtest, 5, 6],
}

norms = {
    "Identity": nn.Identity,
    "BatchNorm2d": nn.LazyBatchNorm2d,
    "InstanceNorm2d": nn.LazyInstanceNorm2d,
    "LayerNorm": LazyLayerNorm,
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
        return result, feat

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

        return result, feat
        #return result.transpose(0, 1), feat.transpose(0, 1)

    def forward_and_predict(self, x):

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

        return result.transpose(0, 1), feat.transpose(0, 1)

    def specialized_predict(self, X):
        pred, feat = self.forward_and_predict(X)
        print(pred.shape)
        print(feat.shape)

        y_pred = pred.reshape(-1, pred.size(-1)).argmax(dim=1)
        feat_flat = feat.reshape(-1, feat.size(-1)).to('cpu')

        return y_pred, feat_flat