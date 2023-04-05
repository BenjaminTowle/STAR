import torch
import logging

logger = logging.getLogger(__name__)


def asymmetric_loss(X, Y):
    """
    Learn one-directional p(y|x)
    """
    s_x_y = torch.sum(X * Y, dim=1)
    s_x = torch.matmul(X, Y.T)
    lse_s_x = torch.logsumexp(s_x, 1, keepdim=False)
    loss = lse_s_x.add(-s_x_y)
    return loss.mean()


def asymmetric_loss_with_hard_negatives(X, Y):
    """
    Learn one-directional p(y|x) and include x as a hard negative to prevent echoing.
    """
    s_x_y = torch.sum(X * Y, dim=1)
    s_x_x = torch.sum(X * X, dim=1)
    s_x = torch.matmul(X, Y.T)
    s_x = torch.cat([s_x, s_x_x], dim=1)
    lse_s_x = torch.logsumexp(s_x, 1, keepdim=False)
    loss = lse_s_x.add(-s_x_y)
    return loss.mean()

from torch.nn import BatchNorm1d

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twins_loss(X, Y):
    """
    Barlow Twins loss
    """
    bn = BatchNorm1d(X.shape[1], affine=False, track_running_stats=False)
    # empirical cross-correlation matrix
    c = bn(X).T @ bn(Y)
    c.div_(X.shape[0])
    
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    lambd = 0.0051
    loss = on_diag + lambd * off_diag
    return loss


def symmetric_loss(X, Y):
    """
    Learns bi-directional p(y|x) and p(x|y)
    """
    s_x_y = torch.sum(X * Y, dim=1)
    s_x = torch.matmul(X, Y.T)
    s_y = torch.matmul(Y, X.T)
    s_x_s_y = torch.cat((s_x, s_y), dim=1)
    lse_s_x_s_y = torch.logsumexp(s_x_s_y, 1, keepdim=False)
    loss = lse_s_x_s_y.add(-s_x_y)
    return loss.mean()

def get_loss_fn(use_symmetric_loss):
    return symmetric_loss if use_symmetric_loss else asymmetric_loss
