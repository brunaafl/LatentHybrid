import torch

import torch.nn as nn

from numpy import random


class JointAlignmentLoss(nn.Module):
    def __init__(self, feat_dim=(16, 1, 251), num_classes=1, centroids=None, lambd = 0.0001):
        super(JointAlignmentLoss, self).__init__()

        self.feat_dim=feat_dim

        # If num_classes=1, align every subject to the same center, regardless of class
        if centroids is None:
            if isinstance(feat_dim, tuple):
                centroids = random.randn(num_classes, *feat_dim)
            else:
                centroids = random.randn(num_classes, feat_dim)

        self.centroids = nn.Parameter(torch.from_numpy(centroids).float())

        # Initialize MSELoss (for centers) and NLL (for the predictions)
        self.mse_loss = nn.MSELoss()
        self.nll = nn.NLLLoss()

        self.lambd = lambd

    def forward(self, feat, y_pred, y_true):

        # Check that feature dimension matches centroid dimension
        if tuple(feat[0].shape) != self.feat_dim:
            raise ValueError(f"Center's dimension: {self.feat_dim} should be equal to input "
                             f"feature's: {tuple(feat[0].shape)}")

        # Creates a vector with dim of long and respective centers for each label
        if self.centroids.size(0) == 1:
            centers_batch = self.centroids.expand_as(feat)
        else:
            centers_batch = self.centroids.index_select(0, y_true.long())

        # Distance of features to center
        alignment_loss = self.mse_loss(feat, centers_batch)
        # Prediction loss
        pred_loss = self.nll(y_pred, y_true)

        loss = pred_loss + self.lambd * alignment_loss

        return loss

