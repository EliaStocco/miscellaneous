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

from miscellaneous.elia.nn.hyper_train import hyper_train_at_fixed_model
from miscellaneous.elia.nn.visualize_dataset import visualize_datasets
from miscellaneous.elia.nn.prepare_dataset import prepare_dataset
from miscellaneous.elia.nn.normalize_datasets import normalize_datasets
from miscellaneous.elia.nn import SabiaNetworkManager
from miscellaneous.elia.functions import add_default, args_to_dict, str2bool
from miscellaneous.elia.nn import compute_normalization_factors

#----------------------------------------------------------------#
# Documentation
# - https://pytorch.org/docs/stable/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

#####################

description = "train a 'e3nn' model"

#####################

default_values = {
        "mul"           : 2,
        "layers"        : 6,
        "lmax"          : 2,
        "name"          : "untitled",
        "reference"     : True,
        "phases"        : False,
        "output"        : "D",
        "max_radius"    : 6.0,
        "folder"        : "LiNbO3",
        "output_folder" : "LiNbO3/results",
        "ref_index"     : 0 ,
        "Natoms"        : None,
        "random"        : False,
        "epochs"        : 10000,
        "bs"            : [1],
        "lr"            : [1e-3],
        "grid"          : True,
        "trial"         : None,
        "max_time"      : -1,
        "task_time"     : -1
    }

#####################

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    # Argument for "input"
    parser.add_argument(
        "-i", "--input", action="store", type=str, metavar="\bjson_file",
        help="input json file", default=None
    )

    # Argument for "input"
    parser.add_argument(
        "--options", action="store", type=str, metavar="\boptions_file",
        help="options json file", default=None
    )

    # Argument for "mul"
    parser.add_argument(
        "--mul", action="store", type=int, metavar="\bmultiplicity",
        help="multiplicity for each node (default: 2)", default=default_values["mul"]
    )

    # Argument for "layers"
    parser.add_argument(
        "--layers", action="store", type=int, metavar="\bn_layers",
        help="number of layers (default: 6)", default=default_values["layers"]
    )

    # Argument for "lmax"
    parser.add_argument(
        "--lmax", action="store", type=int, metavar="\blmax",
        help="some description here (default: 2)", default=default_values["lmax"]
    )

    # Argument for "name"
    parser.add_argument(
        "--name", action="store", type=str, metavar="\bname",
        help="some description here (default: 'untitled')", default=default_values["name"]
    )

    # Argument for "reference"
    parser.add_argument(
        "--reference", action="store",type=str2bool, metavar="\buse_ref",
        help="some description here (default: True)",
        default=default_values["reference"]
    )

        # Argument for "reference"
    parser.add_argument(
        "--phases", action="store",type=str2bool, metavar="\buse_phases",
        help="some description here (default: True)",
        default=default_values["phases"]
    )

    # Argument for "output"
    parser.add_argument(
        "--output", action="store", type=str, metavar="\boutput_folder",
        help="some description here (default: 'D')", default=default_values["output"]
    )

    # Argument for "max_radius"
    parser.add_argument(
        "--max_radius", action="store", type=float, metavar="\bmax_radius",
        help="some description here (default: 6.0)", default=default_values["max_radius"]
    )

    # Argument for "folder"
    parser.add_argument(
        "--folder", action="store", type=str, metavar="\bdata_folder",
        help="some description here (default: 'LiNbO3')", default=default_values["folder"]
    )

    # Argument for "output_folder"
    parser.add_argument(
        "--output_folder", action="store", type=str, metavar="\boutput_folder",
        help="some description here (default: 'LiNbO3/results')", default=default_values["output_folder"]
    )

    # Argument for "ref_index"
    parser.add_argument(
        "--ref_index", action="store", type=int, metavar="\bref_index",
        help="some description here (default: 0)", default=default_values["ref_index"]
    )

    # Argument for "Natoms"
    parser.add_argument(
        "--Natoms", action="store", type=int, metavar="\bNatoms",
        help="some description here (default: 30)", default=default_values["Natoms"]
    )

    # Argument for "random"
    parser.add_argument(
        "--random", action="store",type=str2bool, metavar="\brandom",
        help="some description here (default: True)",
        default=default_values["random"]
    )

    # Argument for "epochs"
    parser.add_argument(
        "--epochs", action="store", type=int, metavar="\bepochs",
        help="some description here (default: 10000)", default=default_values["epochs"]
    )

    # Argument for "bs"
    parser.add_argument(
        "--bs", action="store", type=int, nargs="+", metavar="\bbatch_sizes",
        help="some description here (default: [1])", default=default_values["bs"]
    )

    # Argument for "lr"
    parser.add_argument(
        "--lr", action="store", type=float, nargs="+", metavar="\blearning_rate",
        help="some description here (default: [1e-3])", default=default_values["lr"]
    )

    # Argument for "reference"
    parser.add_argument(
        "--grid", action="store",type=str2bool, metavar="\bas_grid",
        help="some description here (default: True)",
        default=default_values["grid"]
    )

    # Argument for "reference"
    parser.add_argument(
        "--trial", action="store",type=int, metavar="\bn_trial",
        help="some description here (default: True)",
        default=default_values["trial"]
    )

    # Argument for "max_time"
    parser.add_argument(
        "--max_time", action="store",type=int, metavar="\bmax_time",
        help="some description here (default: True)",
        default=default_values["max_time"]
    )

    # Argument for "task_time"
    parser.add_argument(
        "--task_time", action="store",type=int, metavar="\btask_time",
        help="some description here (default: True)",
        default=default_values["task_time"]
    )

    return parser.parse_args()

#####################

def check_parameters(parameters):
    
    str2bool_keys = ["reference","phases","random","grid"]
    for k in str2bool_keys : 
        parameters[k] = str2bool(parameters[k])
    
    if parameters["task_time"] <= 0 :
        parameters["task_time"] = -1

    if parameters["max_time"] <= 0 :
        parameters["max_time"] = -1

    if parameters["reference"] and parameters["phases"]:
        raise ValueError("You can use 'reference'=true or 'phases'=true, not both.")
    
    if parameters["output"] != "D" and parameters["phases"]:
        raise ValueError("You can use 'phases'=true only with 'output'='D'")

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
    #
    if parameters["output"] in ["E","EF"]:
        variable = "energy"
        variables = ["potential"]
    elif parameters["output"] == "D":
        variable = "dipole"
        variables = ["dipole"]
    else :
        variable = None
        variables = ["potential","dipole"]
    
    ##########################################
    # preparing dataset
    opts = {"prepare":{"restart":False},"build":{"restart":False}}
    datasets, data, pos, example = prepare_dataset(ref_index=parameters["ref_index"],\
                                                  max_radius=parameters["max_radius"],\
                                                  reference=parameters["reference"],\
                                                  output=parameters["output"],\
                                                  variables=variables,\
                                                  folder=parameters["folder"],\
                                                  phases=parameters["phases"],\
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
        visualize_datasets(datasets=datasets,variable=variable,folder="{:s}/images".format(folder))

    ##########################################
    # normalizing dataset
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
        mean, std = compute_normalization_factors(datasets["train"],"dipole")
        normalization_factors["dipole"] = {"mean":0.0,"std":std}

    if "E" in parameters["output"] :
        mean, std = compute_normalization_factors(datasets["train"],"energy")
        normalization_factors["energy"] = {"mean":mean,"std":std}

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

    if False :
        print("\n\tModifying datasets for debugging")
        train_dataset = datasets["train"]
        val_dataset   = datasets["val"]
        test_dataset  = datasets["test"]
        
        train_dataset = train_dataset[0:100] 
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
        "phases" : parameters["phases"],
        "normalization" : normalization_factors,
        # "dipole" : dipole.tolist(),
        "pos" : pos.tolist(),  
    }

    #####################

    model_kwargs = {
        "irreps_in":irreps_in,                  # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,                # vector (L=1 and odd parity) to output the polarization
        "max_radius":parameters["max_radius"],  # Cutoff radius for convolution
        "num_neighbors":2,                      # scaling factor based on the typical number of neighbors
        "pool_nodes":True,                      # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":parameters["mul"],
        "layers":parameters["layers"],
        "lmax":parameters["lmax"],
        "dropout_probability" : 0.3
    }

    #####################

    kwargs = {**metadata_kwargs, **model_kwargs}

    instructions = {
            "kwargs":copy(kwargs),
            "class":"SabiaNetworkManager",
            "module":"miscellaneous.elia.nn",
            "chemical-symbols" : example.get_chemical_symbols()
        }
    
    del instructions["kwargs"]["normalization"]
    with open("instructions.json", "w") as json_file:
        # The 'indent' parameter is optional for pretty formatting
        json.dump(instructions, json_file, indent=4)

    net = SabiaNetworkManager(**kwargs)
    print(net)
    N = 0 
    for i in net.parameters():
        if len(i.shape) != 0 :
            N += len(i)
        else :
            N += 1
    print("Tot. number of parameters: ",N)

    ##########################################
    # choose the loss function
    if parameters["output"] in ["D","E"] :
        loss = net.loss()
    elif parameters["output"] == "EF" :
        loss = net.loss(lE=0.1,lF=0.9)
    
    ##########################################
    # optional settings

    if parameters["Natoms"] is None or parameters["Natoms"] == 'None' :
        parameters["Natoms"] = example.get_global_number_of_atoms()

    opts = {
            #"name" : parameters["name"],
            "plot":{
                "learning-curve" : {"N":10},
                "correlation" : {"N":10}
            },
            "thr":{
                "exit":-1
            },
            #"Natoms" : parameters["Natoms"] ,
            #"output_folder" : parameters["output_folder"],
            "save":{
                "parameters":1,
                "checkpoint":1,
            },
            #"grid" : parameters["grid"],
            #"trial" : parameters["trial"],
            "start_time" : start_time,
            'keep_dataset' : True,
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
