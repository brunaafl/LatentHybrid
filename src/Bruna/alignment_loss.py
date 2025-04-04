import torch
import wandb

import torch.nn as nn

from numpy import random

wandb.init(project="centroid_tracking")
class JointAlignmentLoss(nn.Module):
    def __init__(self, feat_dim=(16, 1, 251), num_classes=1, centroids=None, lambd = 1):
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

    def compute_covariance(self, feat):
        B, C, _ = feat.shape
        feat = feat.view(B, C, -1)

        covariances = []
        for i in range(B):  # Compute covariance for each sample
            cov_matrix = torch.cov(feat[i])
            covariances.append(cov_matrix)

        return torch.stack(covariances)

    def compute_alignment_loss(self,feat, y_true):

        cov_feat = self.compute_covariance(feat)
        cov_centroids = self.compute_centroids(self.centroids )

        # Compute class-wise mean covariance
        class_covariances = []
        for cls in range(self.num_classes):
            class_mask = (y_true == cls)
            if class_mask.sum() > 0:
                class_cov = cov_feat[class_mask].mean(dim=0)  # Average covariance for class
                class_covariances.append(class_cov)

        # Compute the distance to the corresponding centroid covariance
        alignment_loss = 0
        for cls, class_cov in enumerate(class_covariances):
            centroid_cov = cov_centroids[cls]
            alignment_loss += self.log_euclidean_distance(class_cov, centroid_cov)

        alignment_loss /= self.num_classes  # Normalize


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
        mean_feat = feat.mean(axis=0)
        alignment_loss = self.mse_loss(mean_feat, self.centroids)

        # Lets do smth different: compute the distance in the spd space


        # Prediction loss
        pred_loss = self.nll(y_pred, y_true)

        loss = pred_loss + self.lambd * alignment_loss
        wandb.log({"alignment loss": alignment_loss.detach().cpu().numpy()})
        wandb.log({"classification loss": pred_loss.detach().cpu().numpy()})

        return loss
