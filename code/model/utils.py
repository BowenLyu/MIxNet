import torch
from torch.autograd import grad

def gradient_(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad_ = grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad_



