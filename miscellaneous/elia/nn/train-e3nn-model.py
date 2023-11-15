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
from miscellaneous.elia.functions import add_default, str2bool

#----------------------------------------------------------------#
# Documentation
# - https://pytorch.org/docs/stablfe/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

description = "train a 'e3nn' model"

#--------------------------------#
default_values = {
        "mul"              : 2,
        "layers"           : 6,
        "lmax"             : 2,
        "name"             : "untitled",
        "output"           : "D",
        "max_radius"       : 6.0,
        "datasets"         : {
            "train" : "data/dataset.train.pth",
            "val"   : "data/dataset.val.pth",
        },
        "output_folder"    : "LiNbO3/results",
        "Natoms"           : None,
        "random"           : False,
        "epochs"           : 10,
        "bs"               : [1],
        "lr"               : [1e-3],
        "weight_decay"     : 1e-2,
        "optimizer"        : "adam",
        "grid"             : True,
        "max_time"         : -1,
        "task_time"        : -1,
        "dropout"          : 0.01,
        # "batchnorm"        : True,
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

#--------------------------------#
def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-i", "--input",action="store", type=str, help="json file with the input parameters")

    return parser.parse_args()

#--------------------------------#
def check_parameters(parameters):
    
    str2bool_keys = ["random","grid","pbc","recompute_loss","debug"]
    for k in str2bool_keys : 
        parameters[k] = str2bool(parameters[k])
    
    if parameters["task_time"] <= 0 :
        parameters["task_time"] = -1

    if parameters["max_time"] <= 0 :
        parameters["max_time"] = -1

    if "chemical-species" not in parameters \
        or parameters["chemical-species"] is None \
            or len(parameters["chemical-species"]) == 0 :
        raise ValueError("Please specify a list of the chemical species that compose the provided atomic structures.")

#--------------------------------#
def get_parameters():
    """get user parameters"""

    args = get_args()

    parameters = None
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
        raise ValueError("no input file provided (--input)")
        # parameters = args_to_dict(args)

    # check that the arguments are okay
    check_parameters(parameters)

    # print parameters
    print("\n\tParameters:")
    for k in parameters.keys():
        print("\t\t{:20s}: ".format(k),parameters[k])
    
    return parameters

#--------------------------------#
# read datasets
def read_datasets(files:dict):
    dataset = dict()
    for k,file in files.items():
        dataset[k] = torch.load(file)
    return dataset

#--------------------------------#
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
   
    # ##########################################
    # # preparing dataset
    # opts = {
    #     "prepare":{
    #         "restart":False
    #     },
    #     "build":{
    #         "restart":False
    #     },
    #     #"use_shift": parameters["shift"],
    #     "instructions" : parameters["instructions"]
    # }
    
    if "datasets" not in parameters:
        raise ValueError("please provide a dict in the input file to specify where to find the datasets.")
    datasets = read_datasets(parameters["datasets"])

    # # I should remove this function from the training procedure
    # datasets, example = prepare_dataset(  # ref_index  = parameters["ref_index"],
    #                                     max_radius = parameters["max_radius"],
    #                                     # reference  = parameters["reference"],
    #                                     output     = parameters["output"],
    #                                     pbc        = parameters["pbc"],
    #                                     indices    = parameters["indices"],
    #                                     folder     = parameters["folder"],
    #                                     opts       = opts
    #                                         )
    
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
        # test_dataset  = datasets["test"]
        
        if "n_debug" in parameters :
            train_dataset = train_dataset[0:parameters["n_debug"]["train"]] 
            val_dataset   = val_dataset  [0:parameters["n_debug"]["val"]] 
        else :
            train_dataset = train_dataset[0:1] 
            val_dataset   = val_dataset  [0:1] 

        print("\tDatasets summary:")
        print("\t\ttrain:",len(train_dataset))
        print("\t\t  val:",len(val_dataset))
        # print("\t\t test:",len(test_dataset))

        datasets = {
            "train":train_dataset,\
            "val"  :val_dataset
        }
        
        parameters["bs"] = [len(train_dataset)]

    ##########################################
    # construct the model

    # if parameters["reference"] :
    #     irreps_in = "{:d}x0e+1x1o".format(len(data.all_types()))
    # else :
    types = parameters["chemical-species"] # np.unique(example.get_chemical_symbols())
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
        # "batchnorm"           : parameters["batchnorm"],
        "pbc"                 : parameters["pbc"],
        "use_shift"           : parameters["use_shift"]
    }

    #####################

    instructions = {
            "kwargs"           : copy(kwargs),
            "class"            : "SabiaNetworkManager",
            "module"           : "miscellaneous.elia.nn.network",
            # "chemical-symbols" : parameters["chemical-symbols"], #example.get_chemical_symbols(),
        }
    
    with open("instructions.json", "w") as json_file:
        json.dump(instructions, json_file, indent=4)

    net = SabiaNetworkManager(**kwargs)
    N = net.n_parameters()
    print("Tot. number of parameters: ",N)
    
    # ##########################################
    # # Natoms
    # if parameters["Natoms"] is None or parameters["Natoms"] == 'None' :
    #     parameters["Natoms"] = example.get_global_number_of_atoms() 

    ##########################################
    # choose the loss function
    if parameters["output"] in ["D","E"] :
        loss = net.loss(Natoms=parameters["Natoms"] if "Natoms" in parameters else None)
    elif parameters["output"] == "EF" :
        raise ValueError("not implemented yet")
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
