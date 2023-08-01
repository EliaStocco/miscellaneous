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

# Documentation
# - https://pytorch.org/docs/stable/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

def main():

    radial_cutoff = 6.0

    ##########################################

    OUTPUT = "EP"

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

        if os.path.exists(savefile+".torch"):
            dataset = torch.load(savefile+".torch")
        else :
            dataset = make_dataset( data=data,radial_cutoff=radial_cutoff)

        # shuffle
        random.shuffle(dataset)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = 2000
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

    print(" test:",len(test_dataset))
    print("  val:",len(val_dataset))
    print("train:",len(train_dataset))

    ##########################################

    for folder in ["results/","results/networks/","results/dataframes","results/images"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    ##########################################

    irreps_in = "{:d}x0e".format(len(data.all_types()))
    if OUTPUT == "E":
        irreps_out = "1x0e"
    elif OUTPUT in ["EP","EPF"]:
        irreps_out = "1x0e + 1x1o"

    radial_cutoff = 6.0
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out, # vector (L=1 and odd parity) to output the polarization
        "max_radius":radial_cutoff, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        #"irreps_node_attr":"1o",
        "mul":10,
        "layers":2,
        "lmax":1,
        #"p":["o","e"],
        "default_dtype" : default_dtype,
    }
    net = SabiaNetworkManager(output=OUTPUT,radial_cutoff=radial_cutoff,**model_kwargs)
    print(net)
    #visualize_layers(net)
    del net

    n = 0
    all_bs = np.arange(30,101,10)
    all_lr = np.logspace(-1, -4.0, num=8)
    Ntot = len(all_bs)*len(all_lr)
    print("\n")
    print("all batch_size:",all_bs)
    print("all lr:",all_lr)
    print("\n")
    df = pd.DataFrame(columns=["bs","lr","file"],index=np.arange(len(all_bs)*len(all_lr)))

    for batch_size in all_bs :

        for lr in all_lr:

            df.at[n,"bs"] = batch_size
            df.at[n,"lr"] = lr
            df.at[n,"file"] = "bs={:d}.lr={:.1e}".format(batch_size,lr)
            
            print("\n#########################\n")
            print("\tbatch_size={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(batch_size,lr,n+1,Ntot))

            print("\n\trebuilding network...\n")
            net = SabiaNetworkManager(output=OUTPUT,radial_cutoff=radial_cutoff,**model_kwargs)#.to(device)

            hyperparameters = {
                'batch_size': batch_size,
                'n_epochs'  : 5,
                'optimizer' : "Adam",
                'lr'        : lr,
                'loss'      : "MSE"
            }

            print("\n\ttraining network...\n")
            out_model, arrays = train(  model=net,\
                                        train_dataset=train_dataset,\
                                        val_dataset=val_dataset,\
                                        hyperparameters=hyperparameters,\
                                        get_pred=net.get_pred,#EPFpred,\
                                        get_real=lambda X: net.get_real(X=X,output=net.output),#EPFreal,\
                                        make_dataloader=None)
            
            savefile = "./results/networks/{:s}.torch".format(df.at[n,"file"])                
            print("saving network to file {:s}".format(savefile))
            torch.save(out_model, savefile)

            savefile = "results/dataframes/{:s}.csv".format(df.at[n,"file"])  
            print("saving arrays to file {:s}".format(savefile))
            arrays.to_csv(savefile,index=False)

            try :
                train_loss = arrays["train_loss"]
                val_loss = arrays["val_loss"]

                print("\n\tplotting losses...\n")
                fig,ax = plt.subplots(figsize=(10,4))
                x = np.arange(len(train_loss))

                ax.plot(x,train_loss,color="blue",label="train",marker=".",linewidth=0.7,markersize=2)
                ax.plot(val_loss,color="red",label="val",marker="x",linewidth=0.7,markersize=2)

                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both",ls="-")
                plt.xlim(0,hyperparameters["n_epochs"])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.title("batch_size={:d}, lr={:.1e}.pdf".format(batch_size,lr))

                plt.tight_layout()
                savefile = "results/images/{:s}.pdf".format(df.at[n,"file"])
                plt.savefig(savefile)
                plt.close(fig)

                plt.figure().clear()
                plt.cla()
                plt.clf()


            except:
                print("Some error during plotting")

            df.at[n,"file"] = df.at[n,"file"] + ".pdf"
            n += 1

            df[:n].to_csv("temp-info.csv",index=False)

    df.to_csv("info.csv",index=False)

    print("\nJob done :)")

if __name__ == "__main__":
    main()
