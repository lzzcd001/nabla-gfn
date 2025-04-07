import torch

def image_postprocess(x, no_clamp=False):
    # [-1, 1] -> [0, 1]
    if no_clamp:
        return (x + 1) / 2
    return torch.clamp((x + 1) / 2, 0, 1)  # x / 2 + 0.5

from contextlib import contextmanager
@contextmanager
def freeze(module: torch.nn.Module):
    """
    Disable gradient for all module parameters. However, if input requires grad
    the graph will still be constructed.
    """
    try:
        prev_states = [p.requires_grad for p in module.parameters()]
        for p in module.parameters():
            p.requires_grad_(False)
        yield

    finally:
        for p, state in zip(module.parameters(), prev_states):
            p.requires_grad_(state)

