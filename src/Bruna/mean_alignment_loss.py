import torch

import numpy as np
import torch.nn as nn
from torch.autograd.function import Function
from numpy import random

class MeanAlignmentLoss(nn.Module):
    def __init__(self, num_classes=1, size_average=True):
        super(MeanAlignmentLoss, self).__init__()
        random.seed(42)

        # If num_classes=1, align every subject to the same center, regardless of class

        # TODO: Ask: what is better, define a common center and learn it, or just try to approximate all subjects?
        # Another thing to ask, is it necessary to learn the centers or you can set them and try to make everyone close?

        self.alignmentlossfunc = AlignmentlossFunc.apply
        self.size_average = size_average

    def forward(self, feat1, feat2, label1, label2):
        batch_size = feat1.size(0)
        feat1 = feat1.view(batch_size, -1)
        feat2 = feat2.view(batch_size, -1)

        # To check the dim of the features
        if feat1.size(1) != feat2.size(1):
            raise ValueError("Feature's dimensions should be equal, but "
                             "{0} != {1}".format(feat2.size(1),feat2.size(1)))

        # To check if number of classes is the same
        if feat1.size(1) != feat2.size(1):
            raise ValueError("Number of classes should be the same, but {0} != {1}".format(
                len(torch.unique(label1)),len(torch.unique(label2))))

        batch_size_tensor = feat1.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.alignmentlossfunc(feat1, feat2, label1, label2, batch_size_tensor)
        return loss


class AlignmentlossFunc(Function):
    @staticmethod
    def forward(ctx, feat1, feat2, label1, label2, batch_size):
        ctx.save_for_backward(feat1, feat2, label1, label2, batch_size)

        # I want to: do a subtraction per class
        unique_classes = torch.unique(label1)
        # Compute the mean per class
        mean1 = torch.stack([feat1[label1 == cls].mean(dim=0) for cls in unique_classes])
        mean2 = torch.stack([feat2[label2 == cls].mean(dim=0) for cls in unique_classes])

        return (mean1 - mean2).pow(2).sum() / 2.0 / batch_size

    # I'm not sure if i need to keep this backward method since i'm not optimizing the center
    @staticmethod
    def backward(ctx, grad_output):
        feat1, feat2, label1, label2, batch_size = ctx.saved_tensors

        if centroids.size(0)==1:
            centers_batch=centroids.clone()
        else:
            centers_batch = centroids.index_select(0, label.long())

        diff = centers_batch - feature

        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centers = centroids.new_zeros(centroids.size())

        # In the case where you have one center per class
        if centroids.size(0) > 1:
            # Regular case: accumulate counts and gradients for multiple centers
            counts = counts.scatter_add_(0, label.long(), ones)
            grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
            grad_centers = grad_centers / counts.view(-1, 1)
        else:
            # Degenerate case: gradient is the average difference for all points
            grad_centers = diff.sum(dim=0, keepdim=True) / feature.size(0)

        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None
