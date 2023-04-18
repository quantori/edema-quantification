import torch.nn as nn

from ..base.modules import Activation
from . import base
from . import functional as F


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(base.Loss, nn.L1Loss):
    pass


class MSELoss(base.Loss, nn.MSELoss):
    pass


class CrossEntropyLoss(base.Loss, nn.CrossEntropyLoss):
    pass


class NLLLoss(base.Loss, nn.NLLLoss):
    pass


class BCELoss(base.Loss, nn.BCELoss):
    pass


class BCEWithLogitsLoss(base.Loss, nn.BCEWithLogitsLoss):
    pass
