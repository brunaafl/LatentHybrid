import torch

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from numpy import random


class AlignmentLoss(nn.Module):
    def __init__(self, feat_dim=(16, 1, 251), num_classes=1, centroids=None):
        super(AlignmentLoss, self).__init__()

        # If num_classes=1, align every subject to the same center, regardless of class
        if centroids is None:
            print(type(feat_dim))
            if isinstance(feat_dim, tuple):
                centroids = random.randn(num_classes, *feat_dim)
            else:
                centroids = random.randn(num_classes, feat_dim)

        self.centroids = nn.Parameter(torch.from_numpy(centroids).float())

        # Initialize MSELoss
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, feat, label):

        # Check that feature dimension matches centroid dimension
        if feat.size != self.feat_dim:
            raise ValueError(f"Center's dimension: {self.feat_dim} should be equal to input feature's: {feat.size(1)}")

        # Creates a vector with dim of long and respective centers for each label
        if self.centroids.size(0) == 1:
            centers_batch = self.centroids.expand_as(feat)
        else:
            centers_batch = self.centroids.index_select(0, label.long())

        # Compute the mean squared error loss between features and corresponding centroids
        loss = self.mse_loss(feat, centers_batch)

        return loss

