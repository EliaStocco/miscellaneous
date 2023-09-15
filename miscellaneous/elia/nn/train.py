# from copy import copy
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import os
from copy import copy
from .make_dataloader import _make_dataloader
from ..functions import add_default, remove_empty_folder
from .plot import plot_learning_curves

__all__ = ["train"]

def train(model:torch.nn.Module,\
          train_dataset:list,\
          val_dataset:list,\
          hyperparameters:dict=None,\
          get_pred:callable=None,\
          get_real:callable=None,\
          make_dataloader:callable=None,\
          correlation:callable=None,\
          output=None,\
          name=None,
          opts=None):
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
    
    print("\nTraining:")
    print("\tPreparing training")

    # information about the status of the training
    info = "all good"

    default = { "plot":{"learning-curve":{"N":10}},\
                "dataloader":{"shuffle":False},\
                "thr":{"exit":10000},\
                "disable":False,\
                "Natoms":1,\
                "save":{"parameters":1}} # ,"networks-temp":-1
    opts = add_default(opts,default)

    # set default values
    if get_pred is None :
        get_pred = lambda f,x : f(x).flatten()   
    if get_real is None :
        get_real = lambda x : x.yreal
    if make_dataloader is None:
        make_dataloader = \
            lambda dataset,batch_size,shuffle=opts["dataloader"]["shuffle"]: \
                _make_dataloader(dataset=dataset,\
                                 batch_size=batch_size,\
                                 shuffle=shuffle)
    if hyperparameters is None:
        hyperparameters = dict()
    
    # output folder
    if output is None :
        warnings.warn("'output' is None: specify a folder name to print some information to file.\n\
                      'results' will be set as default")
        output = 'results'

    # the name of the output files
    if name is None :
        warnings.warn("'name' is None: specify a filename to distinguish the results from other hyperparameters")
        name = "untitled"

    # output folders
    folders = { "networks"        :"{:s}/networks".format(output),\
                # "networks-temp"   :"{:s}/networks-temp".format(output),\
                "parameters"      :"{:s}/parameters".format(output),\
                # "parameters-temp" :"{:s}/parameters-temp".format(output),\
                "dataframes"      :"{:s}/dataframes".format(output),\
                "images"          :"{:s}/images".format(output),\
                "correlations"    :"{:s}/correlations".format(output)}
    
    # create the output folders
    for folder in [output,*folders.values()]:
        if not os.path.exists(folder):
            print("\tCreating folder '{:s}'".format(folder))
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
        
    print("\tHyperparameters:")
    print("\t\tbatch_size:{:d}".format(tryprint(hyperparameters["batch_size"])))
    print("\t\tn_epochs:{:d}".format(tryprint(hyperparameters["n_epochs"])))
    print("\t\toptimizer:{:s}".format(tryprint(hyperparameters["optimizer"])))
    print("\t\tlr:{:.2e}".format(tryprint(hyperparameters["lr"])))
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

    # a useful function
    def get_all_dataloader(dataset):
        """
        Get a data loader for the entire dataset without shuffling.

        This function returns a data loader for the given dataset, allowing access to all data points without shuffling. It's important to set 'shuffle' to False to ensure accurate computation of loss functions.

        Args:
            dataset: Dataset to create the data loader for.

        Returns:
            DataLoader: Data loader for the entire dataset without shuffling.
        """
        # Pay attention!
        # 'shuffle' has to be False!
        # otherwise we will compute the loss function
        # against the wrong data point
        return next(iter(make_dataloader(dataset=dataset,
                                        batch_size=len(dataset),
                                        shuffle=False)))

    # another useful function
    def get_all(dataset):
        """
        Get the real values of the entire dataset.

        This function returns the real values of the entire dataset by utilizing the 'get_all_dataloader' function to retrieve a data loader and then extracting the real values from it.

        Args:
            dataset: Dataset to extract real values from.

        Returns:
            torch.Tensor: Real values of the entire dataset.
        """
        all_dataloader = get_all_dataloader(dataset)
        return get_real(all_dataloader)

    
    # prepare the dataloaders for the train and validation datasets
    dataloader_train = make_dataloader(train_dataset,batch_size)
    #dataloader_val   = next(iter(make_dataloader(val_dataset,len(val_dataset))))
    batches_per_epoch = len(dataloader_train)
    
    # give a summary of the length of the following for cycles
    print("\n\tSummary:")
    print("\t      n. of epochs:",n_epochs)
    print("\t        batch size:",batch_size)
    print("\t  n. of iterations:",batches_per_epoch)
    print("\ttrain dataset size:",len(train_dataset))
    print("\n")
    
    # deepcopy the model into a temporary variable
    # in_model = copy(model) 
        
    # some arrays to store information during the training process
    val_loss = np.full(n_epochs,np.nan)
    train_loss = np.full(n_epochs,np.nan)
    train_loss_one_epoch = np.full(batches_per_epoch,np.nan)

    # dataframe
    arrays = pd.DataFrame({ "train":train_loss,\
                            "val":val_loss})

    # compute the real values of the validation dataset only once
    print("\tCompute validation dataset output (this will save time in the future)")
    yval_real   = get_all(val_dataset)
    all_dataloader_val   = get_all_dataloader(val_dataset)

    print("\tCompute training dataset output (this will save time in the future)")
    ytrain_real = get_all(train_dataset)
    all_dataloader_train = get_all_dataloader(train_dataset)

    savefile = "{:s}/{:s}.init.torch".format(folders["networks"],name)
    print("\tSaving 'model' with dummy parameters to file '{:s}'".format(savefile))
    torch.save(model, savefile)    

    # correlation
    corr = None
    if correlation is not None:        
        corr = pd.DataFrame(columns=["train","val"],index=np.arange(n_epochs))
    
    # start the training procedure
    print("\n\t...and here we go!")
    for epoch in range(n_epochs):    

        if info != "all good":
            break
        
        with tqdm(enumerate(dataloader_train),\
                  total=batches_per_epoch,\
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',\
                  disable=opts["disable"]) as bar:

            for step,X in bar:

                # necessary to train the model
                model.train(mode=True)

                # predict the value for the input X
                y_pred = get_pred(X=X) #get_pred(model=model,X=X)
                
                # true value for the value X
                y_real = get_real(X=X)
                
                # compute the loss function
                loss = loss_fn(y_pred,y_real)

                # store the loss function in an array
                train_loss_one_epoch[step] = float(loss)/opts["Natoms"]

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                # print progress
                if True: #correlation is None :
                    bar.set_postfix(epoch=epoch,
                                    train=np.mean(train_loss_one_epoch[:step+1]),
                                    #train=train_loss_one_epoch[step],
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
            # model.eval()
            with torch.no_grad():
                
                # saving model to temporary file
                # N = opts["save"]["networks-temp"]
                # if N != -1 and epoch % N == 0 :
                #     savefile = "{:s}/{:s}.torch".format(folders["networks-temp"],name)
                #     torch.save(model, savefile)

                # saving parameters to temporary file
                N = opts["save"]["parameters"]
                if N != -1 and epoch % N == 0 :
                    savefile = "{:s}/{:s}.epoch={:d}.pth".format(folders["parameters"],name,epoch)
                    torch.save(model.state_dict(), savefile)

                # compute the loss function
                # predict the value for the validation dataset
                yval_pred = get_pred(all_dataloader_val)# get_pred(model,all_dataloader_val)
                val_loss[epoch] = float(loss_fn(yval_pred,yval_real))/opts["Natoms"]

                # set arrays
                ytrain_pred = get_pred(X=all_dataloader_train) # get_pred(model=model,X=all_dataloader_train)
                train_loss[epoch] = float(loss_fn(ytrain_pred,ytrain_real))/opts["Natoms"]

                if train_loss[epoch] > opts["thr"]["exit"]:
                    info = "try again"
                    break

                arrays.at[epoch,"train"] = train_loss[epoch]
                #arrays.at[epoch,"train_std" ] = train_std [epoch]
                arrays.at[epoch,"val"  ] = val_loss  [epoch]

                # save loss to file
                savefile =  "{:s}/{:s}.csv".format(folders["dataframes"],name)
                arrays.to_csv(savefile,index=False)

                if correlation is not None :
                    # compute correlation
                    ytrain_pred = model(all_dataloader_train)
                    corr["train"][epoch] = correlation(ytrain_pred, ytrain_real)
                    corr["val"][epoch] = correlation(yval_pred, yval_real)

                    # save correlation to file
                    savefile =  "{:s}/{:s}.csv".format(folders["correlations"],name)
                    corr.to_csv(savefile,index=False)

                # produce learning curve plot
                if epoch > 1:
                    savefile =  "{:s}/{:s}.pdf".format(folders["images"],name)
                    plot_learning_curves(train_loss[:epoch+1],\
                                        val_loss[:epoch+1],\
                                        file=savefile,\
                                        title=name if name != "untitled" else None,\
                                        opts=opts["plot"]["learning-curve"])

                # print progress
                if True: #correlation is None :
                    bar.set_postfix(epoch=epoch,
                                    train=train_loss[epoch],
                                    val=val_loss[epoch])
                else :
                    bar.set_postfix(epoch=epoch,
                                    train=train_loss[epoch],
                                    corr_train=corr["train"][epoch],
                                    val=val_loss[epoch],
                                    corr_val=corr["val"][epoch])
    
    # Finished training 
    #print("\n\tTraining done!")

    # Saving some quantities to file
    if info == "all good":
        
        # saving model to file
        savefile = "{:s}/{:s}.torch".format(folders["networks"],name)
        print("\tSaving 'model' to file '{:s}'".format(savefile))
        torch.save(model, savefile)

        # # saving parameters to file
        # savefile = "{:s}/{:s}.torch".format(folders["parameters"],name)
        # print("\tSaving parameters to file '{:s}'".format(savefile))
        # torch.save(model.(), savefile)

        savefile = "{:s}/{:s}.final.pth".format(folders["parameters"],name)
        torch.save(model.state_dict(), savefile)

        # removing initial model 
        savefile = "{:s}/{:s}.init.torch".format(folders["networks"],name)
        if os.path.exists(savefile):
            print("\tDeleting 'model' with dummy parameters: removing file '{:s}'".format(savefile))
            os.remove(savefile)

        # removing empty folders
        _folders = copy(folders)
        for k in folders.keys():
            if remove_empty_folder(folders[k],show=False):
                del _folders[k]
    
        # Important message
        print("\tThe following quantities have been saved to these folders:")
        for k in _folders.keys():
            print("\t{:<20s}: {:<20s}".format(k,folders[k]))
        
        print("\n\tTraining done!\n")

    # Something wrong happened during the training
    # We will let the user know about that through the variable 'info'
    elif info == "try again":
        print("\n\tTraining stopped: we could try again\n")
    
    return model, arrays, corr, info