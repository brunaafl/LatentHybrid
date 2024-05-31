import copy

import torch
from torch import nn

from braindecode.models import EEGNetv4, Deep4Net, ShallowFBCSPNet

from torch.nn import init

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


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
    if start == 0 and end < 19:
        drop_prob = config.model.drop_prob * 0.9
    else:
        drop_prob = config.model.drop_prob

    temp_model = EEGNetv4(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=config.model.final_conv_length,
        drop_prob=drop_prob
    )

    if end == 0:
        return nn.Identity()

    if end == len(list(temp_model.children())) and start == 0:
        return temp_model

    net = list(temp_model.children())[start:end]
    if remove_bn == 'True':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                net[i] = nn.Identity()
    elif remove_bn == 'One-bn':
        for i, module in enumerate(net):
            if isinstance(net[i], nn.BatchNorm2d):
                if i == len(net) - 1 and start == 0:
                    net[i] = net[i]
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
        for model in range(num_models):
            self.unique_modules.append(self.init_unique_modules(*self._args))

    def init_unique_modules(self, n_chans, n_classes, input_window_samples):
        unique_head = model_gen[self.model_type][0](n_chans, n_classes, input_window_samples, self.config,
                                                    end=model_gen[self.model_type][1],
                                                    remove_bn=self.args.remove_bn)
        return unique_head

    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    def forward(self, x):
        inputs = self.split_input(x)
        out = []
        for i, model_input in enumerate(inputs):
            temp_unique = self.unique_modules[i](model_input)
            temp_norm = self.norm(temp_unique)
            temp_shared = self.shared_modules(temp_norm)
            out.append(temp_shared)
        result = torch.stack(out)
        if result.requires_grad:
            result.retain_grad()

        return result

    def generate_branch_model(self):
        new_layers = self.init_unique_modules(*self._args)
        cloned_layers = copy.deepcopy(self.shared_modules)
        norm_clone = copy.deepcopy(self.norm)
        if self.freeze:
            cloned_layers.requires_grad_(False)
        return SpecializedModel(new_layers, norm_clone, cloned_layers)


class SharedModel(nn.Module):
    def __init__(self, n_chans, n_classes, input_window_samples, config=None, freeze='freeze',
                 args=None):
        super(SharedModel, self).__init__()
        self._args = (n_chans, n_classes, input_window_samples)
        self.config = config
        self.args = args
        self.num_models = 8

        temp_model = EEGNetv4(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=config.model.final_conv_length,
            drop_prob=config.model.drop_prob
        )

        self.shared_modules = temp_model

    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    def forward(self, x):
        inputs = self.split_input(x)
        out = []
        for i, model_input in enumerate(inputs):
            temp_shared = self.shared_modules(model_input)
            out.append(temp_shared)
        result = torch.stack(out)
        if result.requires_grad:
            result.retain_grad()

        return result


class SpecializedModel(nn.Module):
    def __init__(self, unique_modules, norm_clone, cloned_modules, num_models=1):
        super(SpecializedModel, self).__init__()
        self.shared_modules = cloned_modules
        self.norm = norm_clone
        self.unique_modules = unique_modules
        self.num_models = num_models

    def split_input(self, X):
        return torch.split(X, int(X.shape[1] / self.num_models), dim=1)

    """def forward(self, x):
        inputs = self.split_input(x)
        out = []
        for i, model_input in enumerate(inputs):
            temp_unique = self.unique_modules(model_input)
            temp_shared = self.shared_modules(temp_unique)
            out.append(temp_shared)
        result = torch.stack(out)
        if result.requires_grad:
            result.retain_grad()

        print(result.shape)

        return result"""

    def forward(self, x):
        print(x.shape)
        x = self.unique_modules(x)
        print(x.shape)
        x = self.shared_modules(x)
        print(x)
        return x

    def predict(self, X):
        return self.forward(X).argmax()
