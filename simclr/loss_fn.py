import torch.nn.functional as F
import torch


def nt_xent(x, t=0.5):
    device = x.device
    num_samples = x.size(0) // 2

    x = F.normalize(x, dim=1)
    x_scores = (x @ x.t()).clamp(min=1e-7)
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values,
    # which will be ~0 after softmax
    x_scale = x_scale - torch.eye(x_scale.size(0), device=device) * 1e5

    # targets 2N elements.
    targets = torch.zeros(x.size()[0], dtype=torch.long, device=device)
    indices = torch.arange(num_samples, device=device)
    targets[:num_samples] = num_samples + indices
    targets[num_samples:] = indices
    return F.cross_entropy(x_scale, targets)
