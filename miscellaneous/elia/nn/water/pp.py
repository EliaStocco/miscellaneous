import torch
from torch.autograd.functional import jacobian
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
from miscellaneous.elia.classes import MicroState
#from miscellaneous.elia.nn.utils.utils_model import visualize_layers
from miscellaneous.elia.nn import train

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import copy
import pandas as pd
import numpy as np
import random
from miscellaneous.elia.nn.water.make_dataset import make_dataset
from miscellaneous.elia.nn.SabiaNetworkManager import SabiaNetworkManager

def main():

    print()


if __name__ == "__main__":
    main()