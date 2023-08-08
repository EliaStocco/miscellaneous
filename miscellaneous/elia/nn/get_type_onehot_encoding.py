import torch
import numpy as np

def get_type_onehot_encoding(species)->(torch.tensor,dict):
    type_encoding = {}
    for n,s in enumerate(species):
        type_encoding[s] = n
    type_onehot = torch.eye(len(type_encoding))
    return type_onehot, type_encoding