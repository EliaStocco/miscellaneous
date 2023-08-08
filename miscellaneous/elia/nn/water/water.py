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
from miscellaneous.elia.classes import MicroState
#from miscellaneous.elia.nn.utils.utils_model import visualize_layers
from miscellaneous.elia.nn import train#, _make_dataloader
from miscellaneous.elia.nn import compute_normalization_factors, normalize

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
from copy import copy
import pandas as pd
import numpy as np
import random
from miscellaneous.elia.nn.water.make_dataset import make_dataset
from miscellaneous.elia.nn.SabiaNetworkManager import SabiaNetworkManager

# Documentation
# - https://pytorch.org/docs/stable/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

def main():

    ##########################################
    # some parameters

    OUTPUT = "D"
    radial_cutoff = 6.0
    output_folder = "results"

    ##########################################

    RESTART = False
    READ = True
    SAVE = True
    savefile = "data/microstate.pickle"

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "data/i-pi.positions_0.xyz"
        instructions = {"properties" : "data/i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"data/i-pi.forces_0.xyz"}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ########################################## 

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/dataset"

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("building dataset")

        if os.path.exists(savefile+".torch") and not RESTART:
            dataset = torch.load(savefile+".torch")
        else :
            dataset = make_dataset( data=data,radial_cutoff=radial_cutoff)

        # shuffle
        random.shuffle(dataset)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = 1000
        i = 500#int(p_test*len(dataset))
        j = 500#int(p_val*len(dataset))

        train_dataset = dataset[:n]
        val_dataset   = dataset[n:n+j]
        test_dataset  = dataset[n+j:n+j+i]

        del dataset

    else :
        print("reading datasets from file {:s}".format(savefile))
        train_dataset = torch.load(savefile+".train.torch")
        val_dataset   = torch.load(savefile+".val.torch")
        test_dataset  = torch.load(savefile+".test.torch")
        SAVE = False
            
    if SAVE :
        print("saving dataset to file {:s}".format(savefile))
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")

    print("train:",len(train_dataset))
    print("  val:",len(val_dataset))
    print(" test:",len(test_dataset))
    
    ##########################################
    print("computing normalization factors for the 'dipole' variable of the train dataset")
    mu, sigma     = compute_normalization_factors(train_dataset,"dipole")
    print("dipole mean :",mu)
    print("dipole sigma:",sigma)

    metadata = {
        "mean": list(mu),
        "std": list(sigma),
        "cutoff":radial_cutoff,
    }

    # Specify the file path
    file_path = "metadata.json"

    # Write the dictionary to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)  # The 'indent' parameter is optional for pretty formatting


    print("nomalizing the 'dipole' variable of all the dataset")
    train_dataset = normalize(train_dataset,mu,sigma,"dipole")
    val_dataset   = normalize(val_dataset,  mu,sigma,"dipole")
    test_dataset  = normalize(test_dataset ,mu,sigma,"dipole")

    print("final mean and std of the 'dipole' variable of all the dataset")
    mu, sigma     = compute_normalization_factors(train_dataset,"dipole")
    print("train :",mu,",",sigma)
    mu, sigma     = compute_normalization_factors(val_dataset  ,"dipole")
    print("val   :",mu,",",sigma)
    mu, sigma     = compute_normalization_factors(test_dataset ,"dipole")
    print("test  :",mu,",",sigma)

    # # Let's do a simple test!
    # # If your NN is not working, let's focus only on one datapoint!
    # # The NN should train and the loss on the validation dataset get really high
    # # If this does not happen ... there is a bug somewhere
    # # You can also read this post: 
    # # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
    
    # train_dataset = train_dataset[0:1] 
    # val_dataset   = val_dataset  [0:1] 

    # print("train:",len(train_dataset))
    # print("  val:",len(val_dataset))
    # print(" test:",len(test_dataset))

    ##########################################

    irreps_in = "{:d}x0e".format(len(data.all_types()))
    if OUTPUT in ["E","EF"]:
        irreps_out = "1x0e"
    elif OUTPUT in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif OUTPUT == "D":
        irreps_out = "1x1o"

    # for layers in [1,2,3,4,5,6]:
    #     for mul in [1,2,3,4,5,6]:
    mul = 3
    layers = 3
    lmax = 2
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,    # vector (L=1 and odd parity) to output the polarization
        "max_radius":radial_cutoff, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":mul,
        "layers":layers,
        "lmax":lmax,
        "default_dtype" : default_dtype,
    }
    net = SabiaNetworkManager(output=OUTPUT,radial_cutoff=radial_cutoff,**model_kwargs)
    print(net)
    N = 0 
    for i in net.parameters():
        N += len(i)
    print("tot. number of parameters: ",N)

    all_bs = [50]#[10,30,60,90]
    all_lr = [1e-3]#[2e-4,1e-3,5e-3]
    Ntot = len(all_bs)*len(all_lr)
    print("\n")
    print("all batch_size:",all_bs)
    print("all lr:",all_lr)
    print("\n")
    df = pd.DataFrame(columns=["bs","lr","file"],index=np.arange(len(all_bs)*len(all_lr)))

    init_model = copy(net) 

    n = 0
    info = "all good"
    max_try = 5
    for batch_size in all_bs :

        for lr in all_lr:

            df.at[n,"bs"] = batch_size
            df.at[n,"lr"] = lr
            df.at[n,"file"] = "bs={:d}.lr={:.1e}".format(batch_size,lr)
            
            print("\n#########################\n")
            print("\tbatch_size={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(batch_size,lr,n+1,Ntot))

            print("\n\trebuilding network...\n")
            net = copy(init_model)
            
            hyperparameters = {
                'batch_size': batch_size,
                'n_epochs'  : 100,
                'optimizer' : "Adam",
                'lr'        : lr,
                'loss'      : net.loss()#net.loss(lE=1,lF=10) #if OUTPUT == 'P' lE and lF will be ignored
            }

            print("\n\ttraining network...\n")
            count_try = 0
            while (info == "try again" and count_try < max_try) or count_try == 0 :
                model, arrays, corr, info = \
                    train(  model=net,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            hyperparameters=hyperparameters,
                            get_pred=net.get_pred,
                            get_real=lambda X: net.get_real(X=X,output=net.output),
                            #correlation=SabiaNetworkManager.correlation,
                            output=output_folder,
                            name=df.at[n,"file"],
                            opts={"plot":{"N":1},"dataloader":{"shuffle":True}})
                count_try += 1

            if info == "try again":
                print("\nAborted training. Let's go on!\n") 

            df.at[n,"file"] = df.at[n,"file"] + ".pdf"
            n += 1

            df[:n].to_csv("temp-info.csv",index=False)

    # writo information to file 'info.csv'
    try : 
        df.to_csv("info.csv",index=False)

        # remove 'temp-info.csv'
        file_path = "temp-info.csv"  # Replace with the path to your file
        try:
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{e}'")
    except OSError as e:
        print(f"Error writing file '{e}'")

    os.remove("temp-info.csv")

    print("\nJob done :)")

if __name__ == "__main__":
    main()
