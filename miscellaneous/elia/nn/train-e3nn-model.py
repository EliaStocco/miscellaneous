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
from miscellaneous.elia.nn.dataset import prepare_dataset
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
        "mul"              : 2,
        "layers"           : 6,
        "lmax"             : 2,
        "name"             : "untitled",
        # "reference"        : False,
        "output"           : "D",
        "max_radius"       : 6.0,
        "folder"           : "LiNbO3",
        "output_folder"    : "LiNbO3/results",
        # "ref_index"        : 0 ,
        "Natoms"           : None,
        "random"           : False,
        "epochs"           : 10000,
        "bs"               : [1],
        "lr"               : [1e-3],
        "weight_decay"     : 1e-2,
        "optimizer"        : "adam",
        "grid"             : True,
        "max_time"         : -1,
        "task_time"        : -1,
        "dropout"          : 0.01,
        "batchnorm"        : True,
        "use_shift"        : None,
        "restart"          : False,
        "recompute_loss"   : False,
        "pbc"              : False,
        "instructions"     : None,
        "debug"            : False,
        "indices"          : None,
        "options"          : None,
        "scheduler"        : None,
        "scheduler-factor" : 1e-2,
    }

#####################

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-i", "--input",     action="store", type=str)
    parser.add_argument("--options",         action="store", type=str)
    parser.add_argument("--mul",             action="store", type=int)
    parser.add_argument("--layers",          action="store", type=int)
    parser.add_argument("--lmax",            action="store", type=int)
    parser.add_argument("--name",            action="store", type=str)
    # parser.add_argument("--reference",       action="store", type=str2bool,default=default_values["reference"])
    parser.add_argument("--output",          action="store", type=str)
    parser.add_argument("--max_radius",      action="store", type=float)
    parser.add_argument("--folder",          action="store", type=str)
    parser.add_argument("--output_folder",   action="store", type=str)
    # parser.add_argument("--ref_index",       action="store", type=int)
    parser.add_argument("--Natoms",          action="store", type=int)
    parser.add_argument("--random",          action="store", type=str2bool, default=default_values["random"])
    parser.add_argument("--epochs",          action="store", type=int)
    parser.add_argument("--bs",              action="store", type=int,      nargs="+")
    parser.add_argument("--lr",              action="store", type=float,    nargs="+")
    parser.add_argument("--grid",            action="store", type=str2bool, default=default_values["grid"])
    parser.add_argument("--max_time",        action="store", type=int,      default=default_values["max_time"])
    parser.add_argument("--task_time",       action="store", type=int,      default=default_values["task_time"])
    parser.add_argument("--dropout",         action="store", type=float,    default=default_values["dropout"])
    parser.add_argument("--batchnorm",       action="store", type=str2bool, default=default_values["batchnorm"])
    parser.add_argument("--use_shift",       action="store", type=str2bool, default=default_values["use_shift"])
    parser.add_argument("--restart",         action="store", type=str2bool, default=default_values["restart"])
    parser.add_argument("--recompute_loss",  action="store", type=str2bool, default=default_values["recompute_loss"])
    parser.add_argument("--pbc",             action="store", type=str2bool, default=default_values["pbc"])
    parser.add_argument("--instructions",    action="store", type=dict,     default=default_values["instructions"])
    parser.add_argument("--debug",           action="store", type=str2bool, default=default_values["debug"])
    parser.add_argument("--indices",         action="store", type=str,      default=default_values["indices"])
    parser.add_argument("--weight_decay",    action="store", type=str,      default=default_values["weight_decay"])
    parser.add_argument("--optimizer",       action="store", type=str,      default=default_values["optimizer"])
    parser.add_argument("--scheduler",       action="store", type=str,      default=default_values["scheduler"])
    parser.add_argument("--scheduler-factor",action="store", type=str,      default=default_values["scheduler-factor"])

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

#####################

def get_parameters():
    """get user parameters"""

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

    # check that the arguments are okay
    check_parameters(parameters)

    # print parameters
    print("\n\tParameters:")
    for k in parameters.keys():
        print("\t\t{:20s}: ".format(k),parameters[k])
    
    return parameters

#####################

def main():

    ##########################################
    # print description
    print("\n\t{:s}".format(description))

    ##########################################
    # get user parameters
    parameters = get_parameters()    

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
        #"use_shift": parameters["shift"],
        "instructions" : parameters["instructions"]
    }

    # I should remove this function from the training procedure
    datasets, example = prepare_dataset(  # ref_index  = parameters["ref_index"],
                                        max_radius = parameters["max_radius"],
                                        # reference  = parameters["reference"],
                                        output     = parameters["output"],
                                        pbc        = parameters["pbc"],
                                        indices    = parameters["indices"],
                                        folder     = parameters["folder"],
                                        opts       = opts
                                            )
    
    # There is a bug:
    # if requires_grad=True and I build the dataset then at the second epoch the code crash with the following message:
    # Trying to backward through the graph a second time (or directly access saved tensors after 
    # they have already been freed). Saved intermediate values of the graph are freed when 
    # you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward 
    # through the graph a second time or if you need to access saved tensors after calling backward.

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

    # if parameters["reference"] :
    #     irreps_in = "{:d}x0e+1x1o".format(len(data.all_types()))
    # else :
    types = np.unique(example.get_chemical_symbols())
    irreps_in = "{:d}x0e".format(len(types))

    if parameters["output"] in ["E","EF"]:
        irreps_out = "1x0e"
    elif parameters["output"] in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif parameters["output"] == "D":
        irreps_out = "1x1o"
    
    #####################

    kwargs = {
        "output"              : parameters["output"],
        "irreps_in"           : irreps_in,                  
        "irreps_out"          : irreps_out,                
        "max_radius"          : parameters["max_radius"],  
        "num_neighbors"       : 2,                      
        "pool_nodes"          : True,                      
        "num_nodes"           : 2,
        "number_of_basis"     : 10,
        "mul"                 : parameters["mul"],
        "layers"              : parameters["layers"],
        "lmax"                : parameters["lmax"],
        "dropout_probability" : parameters["dropout"],
        "batchnorm"           : parameters["batchnorm"],
        "pbc"                 : parameters["pbc"],
        "use_shift"           : parameters["use_shift"]
    }

    #####################

    instructions = {
            "kwargs"           : copy(kwargs),
            "class"            : "SabiaNetworkManager",
            "module"           : "miscellaneous.elia.nn.network",
            "chemical-symbols" : example.get_chemical_symbols(),
        }
    
    with open("instructions.json", "w") as json_file:
        json.dump(instructions, json_file, indent=4)

    net = SabiaNetworkManager(**kwargs)
    N = net.n_parameters()
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
            "plot":{
                "learning-curve" : {"N":10},
                "correlation" : {"N":-1}
            },
            "thr":{
                "exit":1e7
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

    if parameters["options"] is not None :
        # read parameters from file
        try :
            with open(parameters["options"], 'r') as file:
                options = json.load(file)

        except :
            raise ValueError("error reading options file")
        # it should not be needed ...
        opts = add_default(options,opts)

    ##########################################
    # hyper-train the model
    hyper_train_at_fixed_model( net        = net,
                                all_bs     = parameters["bs"],
                                all_lr     = parameters["lr"],
                                epochs     = parameters["epochs"],
                                loss       = loss,
                                datasets   = datasets,
                                opts       = opts,
                                parameters = parameters
                            )

    print("\nJob done :)")

#####################

if __name__ == "__main__":
    main()
