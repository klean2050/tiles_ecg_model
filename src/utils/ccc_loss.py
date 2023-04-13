"""adapted from https://github.com/klean2050/MuSe2022/blob/master/loss.py"""

import torch, numpy as np
import torch.nn as nn


def calc_ccc(preds, labels):
    """
    Concordance Correlation Coefficient
    :param preds: 1D np array
    :param labels: 1D np array
    :return:
    """

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc


def mean_ccc(preds, labels):
    """

    :param preds: list of list of lists (num batches, batch_size, num_classes)
    :param labels: same
    :return: scalar
    """
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_cccs = np.array(
        [calc_ccc(preds[:, i], labels[:, i]) for i in range(num_classes)]
    )
    mean_ccc = np.mean(class_wise_cccs)
    return mean_ccc


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None):
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)

        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(
            mask, dim=1, keepdim=True
        )
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(
            mask, dim=1, keepdim=True
        )

        y_true_var = torch.sum(
            mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True
        ) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_var = torch.sum(
            mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True
        ) / torch.sum(mask, dim=1, keepdim=True)

        cov = torch.sum(
            mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True
        ) / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(
            2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2),
            dim=0,
        )
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss


def get_segment_wise_labels(labels):
    # collapse labels to one label per segment (originally for MuSe-Sent)
    segment_labels = [labels[i, 0, :] for i in range(labels.size(0))]
    return torch.stack(segment_labels).long()


def get_segment_wise_logits(logits, feature_lens):
    # determines exactly one output for each segment (last timestamp)
    segment_logits = []
    for i in range(logits.size(0)):
        segment_logits.append(
            logits[i, feature_lens[i] - 1, :]
        )  # (batch-size, frames, classes)
    return torch.stack(segment_logits, dim=0)
