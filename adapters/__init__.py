from .base_adapter import BaseAdapter, NoAdapter
from .T2A import T2AAdapter


def create_adapter(model, adaptation_method: str, device=None, **kwargs):
    if adaptation_method == "source":
        return NoAdapter(model, device, **kwargs)
    elif adaptation_method == "T2A":
        return T2AAdapter(model, device, **kwargs)
    else:
        raise NotImplementedError(
            f"Adaptation method {adaptation_method} not implemented"
        )
