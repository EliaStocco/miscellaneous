import time
start_time = time.time()

import numpy as np
import random
import json
import argparse
from copy import copy

import torch
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

from miscellaneous.elia.nn.training import hyper_train_at_fixed_model
from miscellaneous.elia.nn.plot import visualize_datasets
from miscellaneous.elia.nn.dataset import prepare_dataset
from miscellaneous.elia.nn.functions import get_data_from_dataset
from miscellaneous.elia.nn.network import SabiaNetworkManager
from miscellaneous.elia.functions import add_default, args_to_dict, str2bool

#----------------------------------------------------------------#
# Documentation
# - https://pytorch.org/docs/stablfe/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

#####################

description = "train a 'e3nn' model"

#####################

default_values = {
        "mul"            : 2,
        "layers"         : 6,
        "lmax"           : 2,
        "name"           : "untitled",
        "reference"      : False,
        # "phases"         : False,
        "output"         : "D",
        "max_radius"     : 6.0,
        "folder"         : "LiNbO3",
        "output_folder"  : "LiNbO3/results",
        "ref_index"      : 0 ,
        "Natoms"         : None,
        "random"         : False,
        "epochs"         : 10000,
        "bs"             : [1],
        "lr"             : [1e-3],
        "grid"           : True,
        "trial"          : None,
        "max_time"       : -1,
        "task_time"      : -1,
        "dropout"        : 0.01,
        "batchnorm"      : True,
        "shift"          : None,
        "restart"        : False,
        "recompute_loss" : False,
        "pbc"            : False,
        "instructions"   : None,
        "debug" : False,
        "indices" : None
    }

#####################

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-i", "--input", action="store", type=str, # metavar="\bjson_file",
        # help="input json file", default=None
    )

    parser.add_argument(
        "--options", action="store", type=str, # metavar="\boptions_file",
        # help="options json file", default=None
    )

    parser.add_argument(
        "--mul", action="store", type=int, # metavar="\bmultiplicity",
        # help="multiplicity for each node (default: 2)", default=default_values["mul"]
    )

    parser.add_argument(
        "--layers", action="store", type=int, # metavar="\bn_layers",
        # help=""debug"number of layers (default: 6)", default=default_values["layers"]
    )

    parser.add_argument(
        "--lmax", action="store", type=int, # metavar="\blmax",
        # help="some description here (default: 2)", default=default_values["lmax"]
    )

    parser.add_argument(
        "--name", action="store", type=str, # metavar="\bname",
        # help="some description here (default: 'untitled')", default=default_values["name"]
    )

    parser.add_argument(
        "--reference", action="store",type=str2bool, # metavar="\buse_ref",
        # help="some description here (default: True)",
        default=default_values["reference"]
    )

    parser.add_argument(
        "--output", action="store", type=str, # metavar="\boutput_folder",
        # help="some description here (default: 'D')", default=default_values["output"]
    )

    parser.add_argument(
        "--max_radius", action="store", type=float, # metavar="\bmax_radius",
        # help="some description here (default: 6.0)", default=default_values["max_radius"]
    )

    parser.add_argument(
        "--folder", action="store", type=str, # metavar="\bdata_folder",
        # help="some description here (default: 'LiNbO3')", default=default_values["folder"]
    )

    parser.add_argument(
        "--output_folder", action="store", type=str, # metavar="\boutput_folder",
        # help="some description here (default: 'LiNbO3/results')", default=default_values["output_folder"]
    )

    parser.add_argument(
        "--ref_index", action="store", type=int, # metavar="\bref_index",
        # help="some description here (default: 0)", default=default_values["ref_index"]
    )

    parser.add_argument(
        "--Natoms", action="store", type=int, # metavar="\bNatoms",
        # help="some description here (default: 30)", default=default_values["Natoms"]
    )

    parser.add_argument(
        "--random", action="store",type=str2bool, # metavar="\brandom",
        # help="some description here (default: True)",
        default=default_values["random"]
    )

    parser.add_argument(
        "--epochs", action="store", type=int, # metavar="\bepochs",
        # help="some description here (default: 10000)", default=default_values["epochs"]
    )

    parser.add_argument(
        "--bs", action="store", type=int, nargs="+", # metavar="\bbatch_sizes",
        # help="some description here (default: [1])", default=default_values["bs"]
    )

    parser.add_argument(
        "--lr", action="store", type=float, nargs="+", # metavar="\blearning_rate",
        # help="some description here (default: [1e-3])", default=default_values["lr"]
    )

    parser.add_argument(
        "--grid", action="store",type=str2bool, # metavar="\bas_grid",
        # help="some description here (default: True)",
        default=default_values["grid"]
    )

    parser.add_argument(
        "--trial", action="store",type=int, # metavar="\bn_trial",
        # help="some description here (default: True)",
        default=default_values["trial"]
    )

    parser.add_argument(
        "--max_time", action="store",type=int, # metavar="\bmax_time",
        # help="some description here (default: True)",
        default=default_values["max_time"]
    )

    parser.add_argument(
        "--task_time", action="store",type=int, # metavar="\btask_time",
        # help="some description here (default: True)",
        default=default_values["task_time"]
    )

    parser.add_argument(
        "--dropout", action="store",type=float, # metavar="\bdropout_prob",
        # help="some description here (default: True)",
        default=default_values["dropout"]
    )

    parser.add_argument(
        "--batchnorm", action="store",type=str2bool, # metavar="\buse_batch_norm",
        # help="some description here (default: True)",
        default=default_values["batchnorm"]
    )

    parser.add_argument(
        "--shift", action="store", type=int, nargs="+", # metavar="\bpahses_shift",
        # help="some description here (default: [1])", default=default_values["shift"]
    )

    parser.add_argument(
        "--restart", action="store",type=str2bool, # metavar="\brestart",
        # help="some description here (default: True)",
        default=default_values["restart"]
    )

    parser.add_argument(
        "--recompute_loss", action="store",type=str2bool, # metavar="\brecompute_loss",
        # help="some description here (default: True)",
        default=default_values["recompute_loss"]
    )

    parser.add_argument(
        "--pbc", action="store",type=str2bool, # metavar="\bpbc",
        # help="some description here (default: True)",
        default=default_values["pbc"]
    )

    parser.add_argument(
        "--instructions", action="store",type=dict, # metavar="\bpbc",
        # help="some description here (default: True)",
        default=default_values["instructions"]
    )

    parser.add_argument(
        "--debug", action="store",type=str2bool, # metavar="\bpbc",
        # help="some description here (default: True)",
        default=default_values["debug"]
    )

    parser.add_argument("--indices", action="store", type=str, default=default_values["indices"])

    return parser.parse_args()

#####################

def check_parameters(parameters):
    
    str2bool_keys = ["reference","random","grid","pbc","recompute_loss","debug"] # "phases"
    for k in str2bool_keys : 
        parameters[k] = str2bool(parameters[k])
    
    if parameters["task_time"] <= 0 :
        parameters["task_time"] = -1

    if parameters["max_time"] <= 0 :
        parameters["max_time"] = -1

    # if parameters["reference"] and parameters["phases"]:
    #     raise ValueError("You can use 'reference'=true or 'phases'=true, not both.")
    
    # if parameters["output"] != "D" and parameters["phases"]:
    #     raise ValueError("You can use 'phases'=true only with 'output'='D'")

#####################

def main():

    start_time 

    ##########################################
    # get user parameters

    args = get_args()

    if args.input is not None :
        # read parameters from file
        try :
            with open(args.input, 'r') as file:
                parameters = json.load(file)

        except :
            raise ValueError("error reading input file")
        # it should not be needed ...
        parameters = add_default(parameters,default_values)
    else :
        parameters = args_to_dict(args)

    ##########################################
    # check that the arguments are okay
    check_parameters(parameters)

    ##########################################
    if not parameters["random"] :
        # Set the seeds of the random numbers generators.
        # This is important for reproducitbility:
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
   
    ##########################################
    # preparing dataset
    opts = {
        "prepare":{
            "restart":False
        },
        "build":{
            "restart":False
        },
        "shift": parameters["shift"],
        "instructions" : parameters["instructions"]
    }
    datasets, data, pos, example, shift = prepare_dataset(ref_index=parameters["ref_index"],\
                                                  max_radius=parameters["max_radius"],\
                                                  reference=parameters["reference"],\
                                                  output=parameters["output"],\
                                                  pbc=parameters["pbc"],\
                                                  indices = parameters["indices"],\
                                                  # variables=variables,\
                                                  folder=parameters["folder"],\
                                                  # phases=parameters["phases"],\
                                                  opts=opts)#,\
                                                  #requires_grad=False)#parameters["output"]=="EF")
    
    # There is a bug:
    # if requires_grad=True and I build the dataset then at the second epoch the code crash with the following message:
    # Trying to backward through the graph a second time (or directly access saved tensors after 
    # they have already been freed). Saved intermediate values of the graph are freed when 
    # you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward 
    # through the graph a second time or if you need to access saved tensors after calling backward.

    ##########################################
    # visualize dataset
    if False :
        visualize_datasets(datasets=datasets,variable="phases",folder="{:s}/images".format(parameters["folder"]))

    ##########################################
    # normalizing dataset
    if False :
        normalization_factors = {
            "dipole":{
                "mean":0,
                "std":1
            },
            "energy":{
                "mean":0,
                "std":1
            }
        }

        if "D" in parameters["output"] :
            dipole = get_data_from_dataset(datasets["all"],"dipole")
            x = torch.mean(dipole,dim=0)
            normalization_factors["dipole"] = {
                "mean" :x.tolist(),
                "std" :torch.norm(dipole-x,dim=1).mean().tolist()
            }

            print("\tmean: ",normalization_factors["dipole"]["mean"])
            print("\t std: ",normalization_factors["dipole"]["std"])

        elif "E" in parameters["output"] :
            raise ValueError("not implemented yet")
            # mean, std = compute_normalization_factors(datasets["train"],"energy")
            # normalization_factors["energy"] = {"mean":mean,"std":std}

        # normalization_factors, datasets = normalize_datasets(datasets)

        match parameters["output"] :
            case "D" :
                normalization_factors = normalization_factors["dipole"]
            case ["E","F"] :
                normalization_factors = normalization_factors["energy"]
            case _ :
                raise ValueError("not implemented yet")

    ##########################################
    # visualize dataset
    if False :
        visualize_datasets(datasets=datasets,variable=variable,folder="{:s}/images-normalized".format(folder))

    ##########################################
    # test
    # # Let's do a simple test!
    # # If your NN is not working, let's focus only on one datapoint!
    # # The NN should train and the loss on the validation dataset get really high
    # # If this does not happen ... there is a bug somewhere
    # # You can also read this post: 
    # # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn

    if parameters["debug"] :
        print("\n\tModifying datasets for debugging")
        train_dataset = datasets["train"]
        val_dataset   = datasets["val"]
        test_dataset  = datasets["test"]
        
        if "n_debug" in parameters :
            train_dataset = train_dataset[0:parameters["n_debug"]["train"]] 
            val_dataset   = val_dataset  [0:parameters["n_debug"]["val"]] 
        else :
            train_dataset = train_dataset[0:1] 
            val_dataset   = val_dataset  [0:1] 

        print("\tDatasets summary:")
        print("\t\ttrain:",len(train_dataset))
        print("\t\t  val:",len(val_dataset))
        print("\t\t test:",len(test_dataset))

        datasets = {"train":train_dataset,\
                    "val"  :val_dataset,\
                    "test" :test_dataset }
        
        parameters["bs"] = [len(train_dataset)]

    ##########################################
    # construct the model

    if parameters["reference"] :
        irreps_in = "{:d}x0e+1x1o".format(len(data.all_types()))
    else :
        irreps_in = "{:d}x0e".format(len(data.all_types()))

    if parameters["output"] in ["E","EF"]:
        irreps_out = "1x0e"
    elif parameters["output"] in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif parameters["output"] == "D":
        irreps_out = "1x1o"
    
    #####################

    metadata_kwargs = {
        "output":parameters["output"],
        "reference" : parameters["reference"],
        # "phases" : parameters["phases"],
        # "normalization" : normalization_factors,
        # "dipole" : dipole.tolist(),
        "pos" : pos.tolist(),  
        # "shift" : list(shift),
    }

    #####################

    model_kwargs = {
        "irreps_in":irreps_in,                  # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,                # vector (L=1 and odd parity) to output the polarization
        "max_radius":parameters["max_radius"],  # Cutoff radius for convolution
        "num_neighbors":2,                      # scaling factor based on the typical number of neighbors
        "pool_nodes":True,                      # We pool nodes to predict total energy
        "num_nodes":2,
        "number_of_basis" : 10,
        "mul":parameters["mul"],
        "layers":parameters["layers"],
        "lmax":parameters["lmax"],
        "dropout_probability" : parameters["dropout"],
        "batchnorm" : parameters["batchnorm"],
        "pbc" : parameters["pbc"]
    }

    #####################

    kwargs = {**metadata_kwargs, **model_kwargs}

    instructions = {
            "kwargs":copy(kwargs),
            "class":"SabiaNetworkManager",
            "module":"miscellaneous.elia.nn.network",
            "chemical-symbols" : example.get_chemical_symbols(),
            "shift" : list(shift),
        }
    
    # del instructions["kwargs"]["normalization"]
    with open("instructions.json", "w") as json_file:
        # The 'indent' parameter is optional for pretty formatting
        json.dump(instructions, json_file, indent=4)

    net = SabiaNetworkManager(**kwargs)
    print(net)
    N = net.n_parameters()
    # N = 0 
    # for i in net.parameters():
    #     if len(i.shape) != 0 :
    #         N += len(i)
    #     else :
    #         N += 1
    print("Tot. number of parameters: ",N)
    
    ##########################################
    # Natoms
    if parameters["Natoms"] is None or parameters["Natoms"] == 'None' :
        parameters["Natoms"] = example.get_global_number_of_atoms() 

    ##########################################
    # choose the loss function
    if parameters["output"] in ["D","E"] :
        loss = net.loss(Natoms=parameters["Natoms"])
    elif parameters["output"] == "EF" :
        loss = net.loss(lE=0.1,lF=0.9)
    
    ##########################################
    # optional settings
    opts = {
            #"name" : parameters["name"],
            "plot":{
                "learning-curve" : {"N":10},
                "correlation" : {"N":-1}
            },
            "thr":{
                "exit":100
            },
            "save":{
                "parameters":1,
                "checkpoint":1,
            },
            "start_time"     : start_time,
            'keep_dataset'   : True,
            "restart"        : parameters["restart"],
            "recompute_loss" : parameters["recompute_loss"],
        }

    if args.options is not None :
        # read parameters from file
        try :
            with open(args.options, 'r') as file:
                options = json.load(file)

        except :
            raise ValueError("error reading options file")
        # it should not be needed ...
        opts = add_default(options,opts)

    ##########################################
    # hyper-train the model
    hyper_train_at_fixed_model( net      = net,\
                                all_bs   = parameters["bs"],\
                                all_lr   = parameters["lr"],\
                                epochs   = parameters["epochs"],\
                                loss     = loss,\
                                datasets = datasets,\
                                opts     = opts,\
                                parameters = parameters)

    print("\nJob done :)")

#####################

if __name__ == "__main__":
    main()
