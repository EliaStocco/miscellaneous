import argparse
import json5 as json
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
import matplotlib.pyplot as plt
from miscellaneous.elia.nn.functions.functions import get_model
from miscellaneous.elia.functions import plot_bisector

#####################

description = "post-process a 'e3nn' model training"

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    # Argument for "input"
    parser.add_argument(
        "--training", action="store", type=str, metavar="\binput.json",
        help="training input file", default="input.json"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--instructions", action="store", type=str, metavar="\binstructions.json",
        help="model input file", default="instructions.json"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--bs", action="store", type=int, metavar="\bbatch_size",
        help="batch size"
    )

    # Argument for "instructions"
    parser.add_argument(
        "--lr", action="store", type=str, metavar="\blearning_rate",
        help="learning rate"
    )

    return parser.parse_args()

def main():
    
    #####################
    # read input
    args = get_args()

    with open(args.training, 'r') as file:
        parameters = json.load(file)
        if "trial" not in parameters:
            parameters["trial"] = None
        elif parameters["trial"] in ["None",0,1]:
            parameters["trial"] = None        

    # with open(args.instructions, 'r') as file:
    #     instructions = json.load(file)

    #####################
    # find bets parameters
    tmp = parameters["output_folder"], parameters["name"], args.bs, args.lr
    file = "{:s}/dataframes/{:s}.bs={:d}.lr={:s}.csv".format(*tmp)
    if not os.path.exists(file):
        raise ValueError("file does not exist")
    loss = pd.read_csv(file)

    epoch = loss["val"].argmin()

    par_folder = "{:s}/parameters/".format(parameters["output_folder"])
    par_files = os.listdir(par_folder)
    best_parameters = None
    best_epoch = 0
    for file in par_files:
        if parameters["name"] in file :
            tmp = int(file.split("epoch=")[1].split(".")[0])
            if tmp > best_epoch and tmp <= epoch :
                best_parameters = file
                best_epoch = tmp

    best_parameters = "{:s}/{:s}".format(par_folder,best_parameters)
    print("\n\tbest file: {:s}".format(best_parameters))

    #####################
    # get model
    model = get_model(args.instructions,best_parameters)

    #####################
    # get train dataset
    datasets = { "train":None, "val":None, "test":None }

    for name in ["train","val","test"]:
        file = "{:s}/dataset.{:s}.torch".format(parameters["folder"],name)
        datasets[name] = torch.load(file)

    #####################
    # evaluate train dataset
    if parameters["output"] == "D":
        real = {
            "train" : np.zeros((len(datasets["train"]),3)),
            "test"  : np.zeros((len(datasets["test"]),3)),
            "val"   : np.zeros((len(datasets["val"]),3)),
        }            
        ncomp = 3
        colors = ["navy","red","green"]
        labels = ["$P_x$","$P_y$","$P_z$"]
        get_real = lambda x : x.dipole
    elif parameters["output"] == "E":
        real = {
            "train" : np.zeros((len(datasets["train"]),1)),
            "test"  : np.zeros((len(datasets["test"]),1)),
            "val"   : np.zeros((len(datasets["val"]),1)),
        }            
        colors = ["navy"]
        labels = ["energy"]
        ncomp = 1
        get_real = lambda x : x.energy
    else :
        raise ValueError("not implemented yet")

    #####################
    pp_folder = "post-processing"

    pred = deepcopy(real)

    for name in ["train","val","test"]:
        for n,X in enumerate(datasets[name]):
            
            real[name][n,:] = get_real(X).detach().numpy()
            pred[name][n,:] = model(X).detach().numpy()

        if not os.path.exists(pp_folder):
            os.mkdir(pp_folder)

        np.savetxt("{:s}/pred.{:s}.txt".format(pp_folder,name),pred[name])
        np.savetxt("{:s}/real.{:s}.txt".format(pp_folder,name),real[name])

    #####################
    # plot correlation
    fig, axs = plt.subplots(ncols=3,figsize=(15,5))
    for n,name in enumerate(datasets.keys()):
        for xyz,c,l in zip(np.arange(ncomp),colors,labels):
            x = real[name][:,xyz]
            y = pred[name][:,xyz]
            axs[n].scatter(x,y,marker=".",color=c,label=l,s=0.5)

        axs[n].title.set_text(name) 
        lgnd = axs[n].legend()
        axs[n].grid()
        axs[n].set_aspect(1)
        plot_bisector(axs[n])

        # lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_sizes([10.0])

    plt.tight_layout()
    plt.savefig("{:s}/correlation.pdf".format(pp_folder))
    #plt.show()

    #####################
    # compute loss
    loss_func = model.loss()
    
    loss = pd.DataFrame(columns=["train","test","val"],index=["loss"])
    for name in ["train","test","val"] :
        x = torch.from_numpy(pred[name])
        y = torch.from_numpy(real[name])
        loss.at["loss",name] = float(loss_func(x,y))
        #info.at["corr"] = pearsonr(pred[name],real[name]).correlation

    loss_file = "{:s}/loss.csv".format(pp_folder)
    loss.to_csv(loss_file,index=False)

    #####################
    # compute correlation    
    corr = pd.DataFrame(columns=["train","test","val"],index=["x","y","z"])
    for name in ["train","test","val"] :
        x = pred[name]
        y = real[name]
        for n,i in enumerate(["x","y","z"]) :
            corr.at[i,name] = pearsonr(x[:,n],y[:,n]).correlation
    
    corr_file = "{:s}/corr.csv".format(pp_folder)
    corr.to_csv(corr_file)

    #####################
    # save output
    output = {
        "parameters" : best_parameters,
        "folder"     : pp_folder,
        "loss"       : loss_file,
        "corr"       : corr_file,
    }

    with open("post-processing.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

    print("\nJob done :)")

#####################

if __name__ == "__main__":
    main()