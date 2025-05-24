from copy import deepcopy

import torch.nn.functional as F


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def compute_cosine_similarity(a, b, strategy: str):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    if a.shape[1] > b.shape[1]:
        b = F.pad(b, (0, a.shape[1] - b.shape[1]))
    elif b.shape[1] > a.shape[1]:
        a = F.pad(a, (0, b.shape[1] - a.shape[1]))
    return F.cosine_similarity(a, b, dim=1).item()
