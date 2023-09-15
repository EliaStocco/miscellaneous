import numpy as np
import pandas as pd
from copy import copy, deepcopy
import torch
import os
from miscellaneous.elia.nn import train
from ..functions import add_default
from itertools import product

def hyper_train_at_fixed_model( net:torch.nn.Module,\
                                all_bs:list,\
                                all_lr:list,\
                                epochs,\
                                loss:callable,\
                                datasets:dict,\
                                parameters:dict,\
                                opts:dict=None):
    
    ##########################################
    # set optional settings
    default = { #"dataloader":{"shuffle":True},\
                #"disable":False,\
                #"Natoms":1,
                "max_try" : 5,
                #"trial" : None,
                # "name":"untitled",
                #"output_folder":"results"
            }
    opts = add_default(opts,default)

    ##########################################
    # epochs
    if type(epochs) == int:
        rows = len(all_bs)
        cols = len(all_lr)
        data = np.full((rows, cols), epochs)
        epochs = pd.DataFrame(data, index=all_bs, columns=all_lr)

    elif type(epochs) == pd.DataFrame:
        pass
    else :
        raise ValueError("'epochs' type not supported")
    
    ##########################################
    # extract datasets
    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]
    del datasets
    # test_dataset  = datasets["test"]

    

    ##########################################
    # preparing cycle over hyper-parameters
    if parameters["grid"] :
        hyper_pars = product(all_bs, all_lr)
        Ntot = len(all_bs)*len(all_lr)
    else :
        if len(all_bs) != len(all_lr) :
            raise ValueError("batch sizes and learning rates should have the same lenght when 'grid'=True")
        hyper_pars = zip(all_bs, all_lr)
        Ntot = len(all_bs)

    #Ntot = len(hyper_pars)
    df = pd.DataFrame(columns=["bs","lr","file"],index=np.arange(Ntot))

    if parameters["trial"] is not None :
        if parameters["trial"] in [0,1]:
            parameters["trial"] = None
        else :
            df["trial"] = None

    if parameters["task_time"] == -1 and parameters["max_time"] != 1 :
        parameters["task_time"] = ( parameters["max_time"] - 60 ) / Ntot
  

    ##########################################
    # training function
    def run(n,bs,lr):

        info = "all good"

        df.at[n,"bs"] = bs
        df.at[n,"lr"] = lr

        if parameters["trial"] is not None :
            df[n,"trial"] = n_trial
            df.at[n,"file"] = "{:s}.bs={:d}.lr={:.1e}.trial={:d}".format(parameters["name"],bs,lr,n_trial)
        else :
            df.at[n,"file"] = "{:s}.bs={:d}.lr={:.1e}".format(parameters["name"],bs,lr)
        
        print("#########################\n")
        print("bs={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(bs,lr,n+1,Ntot))

        #print("\n\trebuilding network...\n")
        net = deepcopy(init_model)
        
        hyperparameters = {
            'bs': bs,
            'n_epochs'  : epochs.at[bs,lr],
            'optimizer' : "Adam",
            'lr'        : lr,
            'loss'      : loss 
        }

        #print("\n\ttraining network...\n")
        count_try = 0
        while (info == "try again" and count_try < opts["max_try"]) or count_try == 0 :

            if info == "try again":
                print("\nLet's try again\n")

            model, arrays, corr, info = \
                train(  model=net,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        hyperparameters=hyperparameters,
                        get_pred=net.get_pred,
                        get_real=net.get_real, #lambda X: net.get_real(X=X,output=net.output),
                        output=parameters["output_folder"],
                        # correlation = net.correlation,
                        name=df.at[n,"file"],
                        opts=opts,
                        parameters=parameters)
            count_try += 1

        if info == "try again":
            print("\nAborted training. Let's go on!\n") 

        df.at[n,"file"] = df.at[n,"file"] + ".pdf"
        n += 1

        df[:n].to_csv("temp-info.csv",index=False)

        return info
    
    ##########################################
    # looping over all hyper-parameters
    n = 0 
    init_model = deepcopy(net) 
    for bs,lr in hyper_pars :
        if parameters["trial"] is not None : 
            for n_trial in parameters["trial"] : 
                run(n,bs,lr)
                n += 1
        else :       
            run(n,bs,lr)
            n += 1

    ##########################################
    # remove EXIT file if detected
    if os.path.exists("EXIT"):
        os.remove("EXIT")
        print("\n\t'exit' file detected\n")

    ##########################################
    # finish
    try : 
        # write information to file 'info.csv'
        df.to_csv("info.csv",index=False)

        # remove 'temp-info.csv'
        file_path = "temp-info.csv"  # Replace with the path to your file
        try:
            os.remove(file_path)
            print("File '{:s}' deleted successfully.".format(file_path))
        except OSError as e:
            print("Error deleting file '{:s}'".format(e))
    except OSError as e:
        print("Error writing file '{:s}'".format(e))

    pass