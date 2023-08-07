from torch_geometric.loader import DataLoader
from copy import copy
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

__all__ = ["train","_make_dataloader"]

def plot_learning_curves(train_loss,val_loss,file,title=None):
    try :

        matplotlib.use('Agg')
        fig,ax = plt.subplots(figsize=(10,4))
        x = np.arange(len(train_loss))

        ax.plot(x,train_loss,color="red",label="train",marker=".",linewidth=0.7,markersize=2)
        ax.plot(val_loss,color="navy",label="val",marker="x",linewidth=0.7,markersize=2)

        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both",ls="-")
        plt.xlim(0,x.max())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if title is not None :
            plt.title(title)

        plt.tight_layout()
        plt.savefig(file)
        # plt.close(fig)

        # plt.figure().clear()
        # plt.cla()
        # plt.clf()

    except:
        print("Some error during plotting")
    pass

def _make_dataloader(dataset,batch_size=1):
            
        dataloader = DataLoader(dataset,\
                                batch_size=batch_size,\
                                drop_last=True,\
                                shuffle=True)
    
        return dataloader
    
def train(model,\
          train_dataset,\
          val_dataset,\
          #out_shape=None,\
          hyperparameters:dict=None,\
          get_pred:callable=None,\
          get_real:callable=None,\
          make_dataloader:callable=None,\
          correlation:callable=None,\
          output=None,\
          name=None):
    """
    Input:
        model: torch.nn.Modules
            NN to be trained.
            
        train_dataset: list, array
            train dataset.
        
        val_dataset: list, array
            validation dataset.
           
        hyperparameters: dict
            hyperparameters, it has to contain the following keys:
                'batch_size': int
                'n_epochs'  : int
                'optimizer' : str (then converted to torch.optim.Optimizer)
                'lr'        : float
                'loss'      : str (then converted to torch.nn._Loss)
                
        get_pred: None(default), lambda
            lambda function to be applied to the predicted value,
            e.g. (the default is) 'lambda f,x : f(x).flatten()'
            where f is the neural network and x is the input value.
            
        get_real: None(default), lambda
            lambda function to be applied to the input values to get the values to be modeled,
            e.g. (the default is) 'lambda x : x.yreal' 
            assuming that the values are stored in the 'yreal' attribute
    
    Output:
        out_model: torch.nn.Modules
            trained NN
        
        arrays: pandas.DataFrame
            DataFrame the following columns:
                'train_loss': array with the average loss (mean over the mini-batches) of the train dataset 
                'train_std' : array with the loss std-dev (mean over the mini-batches) of the train dataset
                'val_loss'  : array with the average loss (mean over the mini-batches) of the validation dataset
            Each element of these arrays is referred to one epoch. 
    """
    
    print("\nTraining...")

    # set default values
    if get_pred is None :
        get_pred = lambda f,x : f(x).flatten()   
    if get_real is None :
        get_real = lambda x : x.yreal
    if make_dataloader is None:
        make_dataloader = _make_dataloader
    if hyperparameters is None:
        hyperparameters = dict()
    
    # output folder
    if output is None :
        warnings.warn("'output' is None: specify a folder name to print some information to file")

    else :
        if name is None :
            warnings.warn("'name' is None: specify a filename to distinguish the results from other hyperparameters")
            name = "untitled"

        folders = {"networks"     :"{:s}/networks".format(output),\
                   "networks-temp":"{:s}/networks-temp".format(output),\
                   "dataframes"   :"{:s}/dataframes".format(output),\
                   "images"       :"{:s}/images".format(output),\
                   "correlations" :"{:s}/correlations".format(output)}
        
        for folder in [output,*folders.values()]:
            if not os.path.exists(folder):
                print("creating folder '{:s}'".format(folder))
                os.mkdir(folder)

    # hyperparameters    
    if "batch_size" not in hyperparameters:
        hyperparameters["batch_size"] = 32
    if "n_epochs" not in hyperparameters:
        hyperparameters["n_epochs"] = 100
    if "optimizer" not in hyperparameters:
        hyperparameters["optimizer"] = "adam"
    if "lr" not in hyperparameters:
        hyperparameters["lr"] = 1e-2
    if "loss" not in hyperparameters:
        hyperparameters["loss"] = MSELoss()
        
    # print hyperparameters to screen
    def tryprint(obj):
        #if hasattr(obj,'__str__'):
        try :
            if type(obj).__str__ is not object.__str__ or hasattr(obj,'__str__'):
                return obj
            else :
                return "not printable object, sorry for that :("
        except :
            return "some problem, but don't worry :("
        
    print("\nHyperparameters:")
    print("\tbatch_size:{:d}".format(tryprint(hyperparameters["batch_size"])))
    print("\tn_epochs:{:d}".format(tryprint(hyperparameters["n_epochs"])))
    print("\toptimizer:{:s}".format(tryprint(hyperparameters["optimizer"])))
    print("\tlr:{:.2e}".format(tryprint(hyperparameters["lr"])))
    # I had some problems with the loss
    if type(hyperparameters["loss"]) == str : print("\tloss_fn:{:s}".format(tryprint(hyperparameters["loss"])))
       
    # extract hyperparameters for the dict 'hyperparameters'
    batch_size = int(hyperparameters["batch_size"])
    n_epochs   = int(hyperparameters["n_epochs"])
    optimizer  = hyperparameters["optimizer"]
    lr         = float(hyperparameters["lr"])
    loss_fn    = hyperparameters["loss"]
    
    # set default values for some hyperparameters
    if type(optimizer) == str and optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    
    if type(loss_fn) == str and loss_fn.lower() == "mse":
        loss_fn = MSELoss()

    def get_all_dataloader(dataset):
        return next(iter(make_dataloader(dataset,len(dataset))))

    def get_all(dataset):
        all_dataloader = get_all_dataloader(dataset)
        return get_real(all_dataloader)
    
    # prepare the dataloaders for the train and validation datasets
    dataloader_train = make_dataloader(train_dataset,batch_size)
    dataloader_val   = next(iter(make_dataloader(val_dataset,len(val_dataset))))
    batches_per_epoch = len(dataloader_train)
    
    # give a summary of the length of the following for cycles
    print("\nSummary:")
    print("      n. of epochs:",n_epochs)
    print("        batch size:",batch_size)
    print("  n. of iterations:",batches_per_epoch)
    print("train dataset size:",len(train_dataset))
    
    # deepcopy the model into a temporary variable
    in_model = copy(model) 
        
    # some arrays to store information during the training process
    val_loss = train_std = train_loss = np.full(n_epochs,np.nan)
    train_loss_one_epoch = np.full(batches_per_epoch,np.nan)

    # dataframne
    arrays = pd.DataFrame({ "train_loss":train_loss,\
                            "train_std":train_std,\
                            "val_loss":val_loss})

    # compute the real values of the validation dataset only once
    print("\nCompute validation dataset output:")
    yval_real   = get_all(val_dataset)
    ytrain_real = get_all(train_dataset)

    # correlation
    corr = None
    if correlation is not None:
        all_dataloader_train = get_all_dataloader(train_dataset)
        corr = pd.DataFrame(columns=["train","val"],index=np.arange(n_epochs))
    
    # start the training procedure
    for epoch in range(n_epochs):    
        
        with tqdm(enumerate(dataloader_train),\
                  total=batches_per_epoch,\
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as bar:

            for step,X in bar:

                # necessary to train the model
                model.train(mode=True)

                # predict the value for the input X
                y_pred = get_pred(model=model,X=X)
                
                # true value for the value X
                y_train = get_real(X=X)
                
                # compute the loss function
                loss = loss_fn(y_pred,y_train)

                # store the loss function in an array
                train_loss_one_epoch[step] = float(loss)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                # print progress
                if correlation is None :
                    bar.set_postfix(epoch=epoch,
                                    train=np.mean(train_loss_one_epoch[:step+1]),
                                    val=val_loss[epoch-1] if epoch != 0 else np.nan)
                else :
                    bar.set_postfix(epoch=epoch,
                                    train=np.mean(train_loss_one_epoch[:step+1]),
                                    corr_train=corr["train"][epoch-1] if epoch != 0 else np.nan,
                                    val=val_loss[epoch-1] if epoch != 0 else np.nan,
                                    corr_val=corr["val"][epoch-1] if epoch != 0 else np.nan)

            # evaluate model on the test dataset
            # with torch.no_grad():
            # I am not using 'with torch.no_grad()' anymore because 
            # maybe it inferferes with the calculation of the forces
            model.eval()
                
            if output is not None :
                savefile = "{:s}/{:s}.torch".format(folders["networks-temp"],name)
                torch.save(model, savefile)

            # compute the loss function
            # predict the value for the validation dataset
            yval_pred = get_pred(model,dataloader_val)
            val_loss[epoch] = loss_fn(yval_pred,yval_real)

            # set arrays
            train_loss[epoch] = np.mean(train_loss_one_epoch)
            train_std [epoch] = np.std(train_loss_one_epoch)

            arrays.at[epoch,"train_loss"] = train_loss[epoch]
            arrays.at[epoch,"train_std" ] = train_std [epoch]
            arrays.at[epoch,"val_loss"  ] = val_loss  [epoch]

            if output is not None :
                savefile =  "{:s}/{:s}.csv".format(folders["dataframes"],name)
                arrays.to_csv(savefile,index=False)

            if correlation is not None :
                ytrain_pred = model(all_dataloader_train)
                corr["train"][epoch] = correlation(ytrain_pred, ytrain_real)
                corr["val"][epoch] = correlation(yval_pred, yval_real)

                if output is not None :
                    savefile =  "{:s}/{:s}.csv".format(folders["correlations"],name)
                    corr.to_csv(savefile,index=False)

            if output is not None and epoch > 1:
                savefile =  "{:s}/{:s}.pdf".format(folders["images"],name)
                plot_learning_curves(train_loss[:epoch+1],\
                                     val_loss[:epoch+1],\
                                     file=savefile,\
                                     title=name if name != "untitled" else None)

            # print progress
            if correlation is None :
                bar.set_postfix(epoch=epoch,
                                train=train_loss[epoch],
                                val=val_loss[epoch])
            else :
                bar.set_postfix(epoch=epoch,
                                train=train_loss[epoch],
                                corr_train=corr["train"][epoch],
                                val=val_loss[epoch],
                                corr_val=corr["val"][epoch])
            
    # arrays = pd.DataFrame({ "train_loss":train_loss,\
    #           "train_std":train_std,\
    #           "val_loss":val_loss})
            
    # deepcopy the trained model into the output variable
    out_model = copy(model)

    # restore the original value of 'model'
    model = in_model

    # Important message
    print("The following quantities have been saved to these folders:")
    for k in folders:
        print("\t{:<20s}:{:<20s}".format(k,folders[k]))
    
    print("\nTraining done!\n")
    
    return out_model, arrays, corr