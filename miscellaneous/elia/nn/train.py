# from copy import copy
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import os
import shutil
from copy import copy
from .make_dataloader import _make_dataloader
from ..functions import add_default, remove_empty_folder
from .plot import plot_learning_curves
import time

__all__ = ["train"]

yval_real = None 
all_dataloader_val   = None

ytrain_real = None
all_dataloader_train = None

def train(model:torch.nn.Module,\
          train_dataset:list,\
          val_dataset:list,\
          parameters:dict,\
          hyperparameters:dict=None,\
          get_pred:callable=None,\
          get_real:callable=None,\
          make_dataloader:callable=None,\
          correlation:callable=None,\
          output=None,\
          name=None,
          opts=None):
    """
    Train a neural network model.

    Args:
        model (torch.nn.Module): 
            The neural network model to be trained.

        train_dataset (list): 
            List of training data.

        val_dataset (list): 
            List of validation data.

        parameters (dict): 
            Dictionary containing various parameters for training.

        hyperparameters (dict, optional): 
            Dictionary containing hyperparameters for training. Defaults to None.

        get_pred (callable, optional): 
            A function that predicts the output given input data. Defaults to None.

        get_real (callable, optional): 
            A function that extracts the real values from the data. Defaults to None.

        make_dataloader (callable, optional): 
            A function that creates a data loader from a dataset. Defaults to None.

        correlation (callable, optional): 
            A function to compute correlation. Defaults to None.

        output (str, optional): 
            Folder name for saving training information. Defaults to None.

        name (str, optional): 
            A filename to distinguish results from other hyperparameters. Defaults to None.

        opts (dict, optional): 
            Additional options for training. Defaults to None.

    Returns:
        tuple: 
            A tuple containing the trained model, arrays, correlation, and information about the training.
    """
   
    start_task_time = time.time()

    print("\nTraining:")
    print("\tPreparing training")

    # information about the status of the training
    info = "all good"

    default = { "plot":{"learning-curve":{"N":10}},\
                "dataloader":{"shuffle":False},\
                "thr":{"exit":10000},\
                "disable":False,\
                #"Natoms":1,\
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
    if "bs" not in hyperparameters:
        hyperparameters["bs"] = 32
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
    print("\t\tbatch_size:{:d}".format(tryprint(hyperparameters["bs"])))
    print("\t\tn_epochs:{:d}".format(tryprint(hyperparameters["n_epochs"])))
    print("\t\toptimizer:{:s}".format(tryprint(hyperparameters["optimizer"])))
    print("\t\tlr:{:.2e}".format(tryprint(hyperparameters["lr"])))
    # I had some problems with the loss
    if type(hyperparameters["loss"]) == str : print("\tloss_fn:{:s}".format(tryprint(hyperparameters["loss"])))
       
    # extract hyperparameters for the dict 'hyperparameters'
    batch_size = int(hyperparameters["bs"])
    n_epochs   = int(hyperparameters["n_epochs"])
    optimizer  = hyperparameters["optimizer"]
    lr         = float(hyperparameters["lr"])
    _loss_fn    = hyperparameters["loss"]
    
    # set default values for some hyperparameters
    if type(optimizer) == str and optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    
    if type(_loss_fn) == str and _loss_fn.lower() == "mse":
        _loss_fn = MSELoss()
    
    # Natoms
    if parameters["Natoms"] > 1 :
        parameters["Natoms"] = torch.tensor(parameters["Natoms"],requires_grad=False)
        def loss_fn(x:torch.tensor,y:torch.tensor) -> torch.Tensor:
            tmp = _loss_fn(x,y)
            return tmp / parameters["Natoms"]
    else :
        loss_fn = _loss_fn

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
    # val_loss = np.full(n_epochs,np.nan)
    # train_loss = np.full(n_epochs,np.nan)
    tmp = np.full(n_epochs,np.nan)
    train_loss_one_epoch = np.full(batches_per_epoch,np.nan)

    # dataframe
    arrays = pd.DataFrame({ "train":copy(tmp),"val":copy(tmp)})
    del tmp

    global yval_real, all_dataloader_val
    # compute the real values of the validation dataset only once
    if yval_real is None or not opts["keep_dataset"]:
        print("\tCompute validation dataset output (this will save time in the future)")
        yval_real   = get_all(val_dataset)
        all_dataloader_val   = get_all_dataloader(val_dataset)

    global ytrain_real, all_dataloader_train
    if ytrain_real is None or not opts["keep_dataset"]:
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

    ##########################################
    # prepare output files
    savefiles = {
        "dataframes"   : "{:s}/{:s}.csv".format(folders["dataframes"],name),
        "correlations" : "{:s}/{:s}.csv".format(folders["correlations"],name),
        "images"       : "{:s}/{:s}.pdf".format(folders["images"],name),
    }

    ##########################################
    # prepare checkpoint
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    start_epoch = 0 

    checkpoint_folder = "checkpoint"
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    checkpoint_file = "{:s}/{:s}.pth".format(checkpoint_folder,name)
    if os.path.exists(checkpoint_file):
        print("\tReading checkpoint from file '{:s}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

        # read array and corr
        if os.path.exists(savefiles["dataframes"]):
            tmp = pd.read_csv(savefiles["dataframes"])
            arrays.iloc[:len(tmp)] = copy(tmp)
            del tmp
        if os.path.exists(savefiles["correlations"]):
            tmp   = pd.read_csv(savefiles["correlations"])
            corr.iloc[:len(tmp)] = copy(tmp)
            del tmp

    def save_checkpoint():
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)

    ##########################################    
    # start the training procedure
    print("\n\t...and here we go!")
    for epoch in range(start_epoch,n_epochs):    

        if info != "all good":
            break

        ##########################################
        # measure elapsed time
        now = time.time()
        if abs(now - opts["start_time"]+60) > parameters["max_time"] and parameters["max_time"] > 0 :
            print("\n\tMaximum time reached: stopping")
            info = "time over"
            break
        if abs(now - start_task_time + 60 ) > parameters["task_time"] and parameters["task_time"] > 0 :
            print("\n\tMaximum time per task reached: stopping task")
            info = "time over"
            break

        if os.path.exists("EXIT"):
            info = "exit file detected"
            break

        if os.path.exists("EXIT-TASK"):
            os.remove("EXIT-TASK")
            info = "exit-task file detected"
            break

        ##########################################
        # training loop per epoch
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
                train_loss_one_epoch[step] = float(loss) # / parameters["Natoms"]

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
                                    val=arrays.at[epoch-1,"val"] if epoch != 0 else np.nan)
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

                model.eval()
                
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
                arrays.at[epoch,"val"] = float(loss_fn(yval_pred,yval_real)) # / parameters["Natoms"]

                # set arrays
                # ytrain_pred = get_pred(X=all_dataloader_train) # get_pred(model=model,X=all_dataloader_train)
                # train_loss[epoch] = float(loss_fn(ytrain_pred,ytrain_real))  /parameters["Natoms"]
                arrays.at[epoch,"train"] = np.mean(train_loss_one_epoch)

                if arrays.at[epoch,"train"] > opts["thr"]["exit"] and opts["thr"]["exit"] > 0:
                    info = "try again"
                    break

                # arrays.at[epoch,"train"] = train_loss[epoch]
                # arrays.at[epoch,"train_std" ] = train_std [epoch]
                # arrays.at[epoch,"val"  ] = val_loss  [epoch]

                # save loss to file
                # savefile = "{:s}/{:s}.csv".format(folders["dataframes"],name)
                arrays[:epoch+1].to_csv(savefiles["dataframes"],index=False)

                if correlation is not None :
                    # compute correlation
                    ytrain_pred = model(all_dataloader_train)
                    corr["train"][epoch] = correlation(ytrain_pred, ytrain_real)
                    corr["val"][epoch] = correlation(yval_pred, yval_real)

                    # save correlation to file
                    # savefile =  "{:s}/{:s}.csv".format(folders["correlations"],name)
                    corr[:epoch+1].to_csv(savefiles["correlations"],index=False)

                # produce learning curve plot
                if epoch >= 1:
                    # savefile =  "{:s}/{:s}.pdf".format(folders["images"],name)
                    plot_learning_curves(   arrays.loc[:epoch,"train"],\
                                            arrays.loc[:epoch+1,"val"],\
                                            file=savefiles["images"],\
                                            title=name if name != "untitled" else None,\
                                            opts=opts["plot"]["learning-curve"])

                # print progress
                if True: #correlation is None :
                    bar.set_postfix(epoch=epoch,
                                    train=arrays.at[epoch,"train"],
                                    val=arrays.at[epoch,"val"])
                else :
                    bar.set_postfix(epoch=epoch,
                                    train=train_loss[epoch],
                                    corr_train=corr["train"][epoch],
                                    val=val_loss[epoch],
                                    corr_val=corr["val"][epoch])
        
        # saving checkpoint to file
        N = opts["save"]["checkpoint"]
        if N != -1 and epoch % N == 0 :
            save_checkpoint()
        
    #
    save_checkpoint()
    
    # Finished training 
    #print("\n\tTraining done!")

    # Saving some quantities to file
    if info == "all good":
        
        # modify checkpoint filename
        if os.path.exists(checkpoint_file):
            new_file = "{:s}/finished-{:s}.pth".format(checkpoint_folder,name)
            os.rename(checkpoint_file,new_file)
        else :
            save_checkpoint()

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
    elif info == "exit-task file detected":
        print("\n\t'exit-task' file detected\n")
    
    return model, arrays, corr, info