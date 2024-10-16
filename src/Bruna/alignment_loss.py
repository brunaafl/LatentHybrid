import torch
import torch.nn as nn
from torch.autograd.function import Function
from numpy import random

class AlignmentLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=1, size_average=True):
        super(AlignmentLoss, self).__init__()
        random.seed(42)

        # If num_classes=1, align every subject to the same center, regardless of class
        # Change here to feat dimensions
        centroids = random.randn(num_classes, feat_dim)
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.alignmentlossfunc = AlignmentlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)

        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))

        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.alignmentlossfunc(feat, label, self.centroids, batch_size_tensor)
        return loss


class AlignmentlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)

        # Creates a vector with dim of long and respective centers for each label
        if centroids.size(0)==1:
            centers_batch=centroids
        else:
            centers_batch = centroids.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors

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
