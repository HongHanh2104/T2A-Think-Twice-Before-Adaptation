import torch
import torch.nn as nn
from torch.nn import functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class NormalizedLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 2, gamma: float = 0.0):
        super(NormalizedLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        norm = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss / norm

        return loss.mean()


class NormalizedNegativeLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 2, gamma: float = 0.0, p0: float = 1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.p0 = p0
        self.logmp = torch.tensor(self.p0).log()
        self.p0 = -((1 - self.p0) ** self.gamma) * self.logmp

    def forward(self, input, target):
        logmp = self.logmp.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        norm = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = 1 - (self.p0 - loss) / (self.num_classes * self.p0 - norm)
        return loss.mean()


@torch.jit.script
def compute_noise_tolerant_negative_loss(
    x, noise_type: str = "uniform", gamma: float = 2.0, alpha: float = 1.0, beta: float = 1.0
):  # -> torch.Tensor:
    num_classes = x.shape[1]
    probs = x.softmax(-1)  # shape: (batch_size, num_classes)
    pseudo_labels = x.argmax(dim=-1)  # shape: (batch_size,)
    if noise_type == "bernoulli":
        flip_probs = 1 - probs
        flipped_labels = torch.bernoulli(flip_probs).long()
    elif noise_type == "uniform":
        flipped_labels = (pseudo_labels == 0).long()  # shape: (batch_size,)
    else:
        flipped_labels = pseudo_labels
    # Eq. 10 in the paper
    Lnorm = NormalizedLoss(num_classes=num_classes, gamma=gamma)(
        probs, flipped_labels
    )
    # Eq. 11 in the paper
    Lnn = NormalizedNegativeLoss(
        num_classes=num_classes, gamma=gamma, p0=probs.min()
    )(probs, flipped_labels)
    return alpha * Lnorm + beta * Lnn  # Eq. 12 in the paper
