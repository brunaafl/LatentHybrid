import torch
import wandb

import torch.nn as nn

from numpy import random

class JointAlignmentLoss(nn.Module):
    def __init__(self, feat_dim=(16, 1, 251), num_classes=2, centroids=None, lambd = 10):
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

        # Maybe mean squared loss is not the best
        self.mse_loss = nn.MSELoss()
        #self.kl = nn.KLDivLoss()
        self.nll = nn.NLLLoss()

        self.lambd = lambd


    def forward(self, feat, y_pred, y_true):

        # Check that feature dimension matches centroid dimension
        if tuple(feat[0].shape) != self.feat_dim:
            raise ValueError(f"Center's dimension: {self.feat_dim} should be equal to input "
                             f"feature's: {tuple(feat[0].shape)}")

        # Distance of features to center
        mean_feat = feat.mean(axis=0)
        alignment_loss = self.mse_loss(mean_feat, self.centroids)

        # Lets do smth different: compute the distance in the spd space
        # In this case, we define this as a layer in the model not a loss

        # Prediction loss
        pred_loss = self.nll(y_pred, y_true)

        loss = pred_loss + self.lambd * alignment_loss

        return loss
