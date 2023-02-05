import os
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj