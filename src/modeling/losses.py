import torch


def asymmetric_loss(X: torch.Tensor, Y: torch.Tensor):
    """
    Learn one-directional p(y|x)
    """
    s_x_y = torch.sum(X * Y, dim=1)
    s_x = torch.matmul(X, Y.T)
    lse_s_x = torch.logsumexp(s_x, 1, keepdim=False)
    loss = lse_s_x.add(-s_x_y)
    return loss.mean()


def symmetric_loss(X: torch.Tensor, Y: torch.Tensor):
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
