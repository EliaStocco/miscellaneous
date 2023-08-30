import torch
#from torch.autograd.functional import jacobian
#from torch.nn import MSELoss
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import json
#os.environ["QT_QPA_PLATFORM"] = "wayland"
# Now you can import PyQt5 or other Qt-related libraries and run your application.

#from miscellaneous.elia.nn.utils.utils_model import visualize_layers
from miscellaneous.elia.nn.hyper_train import hyper_train_at_fixed_model#, _make_dataloader


# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
from copy import copy
import pandas as pd
import numpy as np
import random

from miscellaneous.elia.nn.water.prepare_dataset import prepare_dataset
from miscellaneous.elia.nn.water.normalize_datasets import normalize_datasets
from miscellaneous.elia.nn.SabiaNetworkManager import SabiaNetworkManager

# Documentation
# - https://pytorch.org/docs/stable/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

def main():

    ##########################################
    # Set the seeds of the random numbers generators.
    # This is important for reproducitbility:
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ##########################################
    # some parameters

    reference = True
    OUTPUT = "EF"
    max_radius = 6.0
    folder = "LiNbO3"
    output_folder = "{:s}/results".format(folder)
    ref_index = 0 
    Natoms = 30 # 3 atoms in the water molecule

    ##########################################
    # preparing dataset
    opts = {"prepare":{"restart":False},"build":{"restart":False}}
    datasets, data, dipole, pos = prepare_dataset(ref_index,\
                                                  max_radius,\
                                                  reference,\
                                                  folder=folder,\
                                                  opts=opts)#,\
                                                  #requires_grad=False)#OUTPUT=="EF")
    
    # There is a bug:
    # if requires_grad=True and I build the dataset then at the second epoch the code crash with the following message:
    # Trying to backward through the graph a second time (or directly access saved tensors after 
    # they have already been freed). Saved intermediate values of the graph are freed when 
    # you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward 
    # through the graph a second time or if you need to access saved tensors after calling backward.

    
    ##########################################
    # normalizing dataset
    normalization_factors, datasets = normalize_datasets(datasets)

    ##########################################
    # test
    # # Let's do a simple test!
    # # If your NN is not working, let's focus only on one datapoint!
    # # The NN should train and the loss on the validation dataset get really high
    # # If this does not happen ... there is a bug somewhere
    # # You can also read this post: 
    # # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn

    if False :
        print("\n\tModifying datasets for debugging")
        train_dataset = datasets["train"]
        val_dataset   = datasets["val"]
        test_dataset  = datasets["test"]
        
        train_dataset = train_dataset[0:10] 
        val_dataset   = val_dataset  [0:10] 

        print("\n\tDatasets summary:")
        print("\t\ttrain:",len(train_dataset))
        print("\t\t  val:",len(val_dataset))
        print("\t\t test:",len(test_dataset))

        datasets = {"train":train_dataset,\
                    "val"  :val_dataset,\
                    "test" :test_dataset }

    ##########################################
    # construct the model

    if reference :
        irreps_in = "{:d}x0e+1x1o".format(len(data.all_types()))
    else :
        irreps_in = "{:d}x0e".format(len(data.all_types()))

    if OUTPUT in ["E","EF"]:
        irreps_out = "1x0e"
    elif OUTPUT in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif OUTPUT == "D":
        irreps_out = "1x1o"

    # for layers in [1,2,3,4,5,6]:
    #     for mul in [1,2,3,4,5,6]:
    mul = 1
    layers = 6
    lmax = 1
    
    #####################

    metadata_kwargs = {
        "output":OUTPUT,
        "reference" : reference,
        "dipole" : dipole.tolist(),
        "pos" : pos.tolist(),
        "normalization" : normalization_factors
    }

    # Write the dictionary to the JSON file
    with open("metadata_kwargs.json", "w") as json_file:
        # The 'indent' parameter is optional for pretty formatting
        json.dump(metadata_kwargs, json_file, indent=4)  

    #####################

    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,    # vector (L=1 and odd parity) to output the polarization
        "max_radius":max_radius,    # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":mul,
        "layers":layers,
        "lmax":lmax,
        #"default_dtype" : str(default_dtype),
    }
    # Write
    #  the dictionary to the JSON file
    with open("metadata_kwargs.json", "w") as json_file:
        # The 'indent' parameter is optional for pretty formatting
        json.dump(model_kwargs, json_file, indent=4)

    #####################

    kwargs = {**metadata_kwargs, **model_kwargs}

    net = SabiaNetworkManager(**kwargs)
    print(net)
    N = 0 
    for i in net.parameters():
        N += len(i)
    print("Tot. number of parameters: ",N)

    ##########################################
    # choose the loss function
    if OUTPUT in ["D","E"] :
        loss = net.loss()
    elif OUTPUT == "EF" :
        loss = net.loss(lE=0.1,lF=0.9)

    ##########################################
    # choose the hyper-parameters
    all_bs = [1]#[10,30,60,90]
    all_lr = [1e-3]#[2e-4,1e-3,5e-3]
    
    ##########################################
    # optional settings
    opts = {"plot":{"N":100},"thr":{"exit":1e6}}

    ##########################################
    # hyper-train the model
    hyper_train_at_fixed_model( net,\
                                all_bs,\
                                all_lr,\
                                loss,\
                                datasets,\
                                output_folder,\
                                Natoms=Natoms,\
                                opts=opts)

    print("\nJob done :)")

if __name__ == "__main__":
    main()
