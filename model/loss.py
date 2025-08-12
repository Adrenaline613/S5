import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. https://arxiv.org/pdf/2002.05709.pdf
        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class DiceLoss(nn.Module):
    def __init__(self, softmax: bool = True, use_weight: bool = True, smooth: float = 1e-6, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()

        self.softmax = softmax
        self.use_weight = use_weight
        self.smooth = smooth
        assert reduction in ['mean', 'sum', 'none'], 'Reduction must be one of [mean, sum, none]'
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # Apply softmax activation to the input if required
        if self.softmax:
            input = F.softmax(input, dim=1)

        # one-hot encoding
        if input.dim() - 1 == target.dim():
            with torch.no_grad():
                y_onehot = F.one_hot(target, num_classes=2)  # one hot encoding in format [N,K,C]
                y_onehot = y_onehot.permute(0, 2, 1)  # transform to format [N,C,K]

        intersection = (input * y_onehot).sum(dim=2)
        union = (input + y_onehot).sum(dim=2)

        if self.use_weight:
            w = 1 / y_onehot.sum(dim=2) ** 2
            w[torch.isinf(w)] = 1.0

            intersection = (w * intersection).sum(dim=1)
            union = (w * union).sum(dim=1)

        else:
            intersection = intersection.sum(dim=1)
            union = union.sum(dim=1)

        gdl = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'mean':
            gdl = gdl.mean()  # average over the batch and channel; scalar
        elif self.reduction == 'sum':
            gdl = gdl.sum()  # sum over the batch and channel; scalar
        elif self.reduction == 'none':
            pass  # unmodified losses per batch; shape [N]

        return gdl


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.9, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([1 - alpha, alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

        self.gamma = gamma

        assert reduction in ['mean', 'sum', 'none'], 'Reduction must be one of [mean, sum, none]'
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Compute the focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets.data.view(-1)].view_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()  # average over the batch and channel; scalar
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()  # sum over the batch and channel; scalar
        elif self.reduction == 'none':
            pass  # unmodified losses per batch; shape [N]

        return focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(
            self,
            dice_weight: float = 0.7,
            dice_class_weight: bool = True,
            alpha: float = 0.9,
            gamma: float = 2.5,
            smooth: float = 1e-6,
            reduction: str = 'mean'
    ):
        super(DiceFocalLoss, self).__init__()

        assert 0 <= dice_weight <= 1, "Dice weight must be between 0 and 1"
        self.dice_weight = dice_weight

        self.dice_loss = DiceLoss(
            softmax=True,
            use_weight=dice_class_weight,
            smooth=smooth,
            reduction=reduction
        )

        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction
        )

        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        # 加权混合
        total_loss = self.dice_weight * dice + (1 - self.dice_weight) * focal
        return total_loss
