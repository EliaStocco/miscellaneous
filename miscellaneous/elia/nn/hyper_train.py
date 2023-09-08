import numpy as np
import pandas as pd
from copy import copy
import torch
import os
from miscellaneous.elia.nn import train
from ..functions import add_default

def hyper_train_at_fixed_model( net:torch.nn.Module,\
                                all_bs:list,\
                                all_lr:list,\
                                epochs,\
                                loss:callable,\
                                datasets:dict,\
                                # output_folder:str,
                                # Natoms:int=1,\
                                opts:dict=None):
    
    ##########################################
    # set optional settings
    default = { #"dataloader":{"shuffle":True},\
                #"disable":False,\
                #"Natoms":1,
                "name":"untitled",
                "output_folder":"results"}
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
    #test_dataset  = datasets["test"]

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
            df.at[n,"file"] = "{:s}.bs={:d}.lr={:.1e}".format(opts["name"],batch_size,lr)
            
            print("#########################\n")
            print("batch_size={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(batch_size,lr,n+1,Ntot))

            #print("\n\trebuilding network...\n")
            net = copy(init_model)
            
            hyperparameters = {
                'batch_size': batch_size,
                'n_epochs'  : epochs.at[batch_size,lr],
                'optimizer' : "Adam",
                'lr'        : lr,
                'loss'      : loss 
            }

            #print("\n\ttraining network...\n")
            count_try = 0
            while (info == "try again" and count_try < max_try) or count_try == 0 :

                if info == "try again":
                    print("\nLet's try again\n")

                model, arrays, corr, info = \
                    train(  model=net,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            hyperparameters=hyperparameters,
                            get_pred=net.get_pred,
                            get_real=lambda X: net.get_real(X=X,output=net.output),
                            output=opts["output_folder"],
                            name=df.at[n,"file"],
                            opts=opts)
                count_try += 1

            if info == "try again":
                print("\nAborted training. Let's go on!\n") 

            df.at[n,"file"] = df.at[n,"file"] + ".pdf"
            n += 1

            df[:n].to_csv("temp-info.csv",index=False)

    # write information to file 'info.csv'
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