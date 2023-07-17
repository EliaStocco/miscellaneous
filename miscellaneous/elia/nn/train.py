from torch_geometric.loader import DataLoader
from copy import copy
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd

def _make_dataloader(dataset,batch_size):
            
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
          make_dataloader:callable=None):
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
    print("\nHyperparameters:")
    print("\tbatch_size:{:d}".format(hyperparameters["batch_size"]))
    print("\tn_epochs:{:d}".format(hyperparameters["n_epochs"]))
    print("\toptimizer:{:s}".format(hyperparameters["optimizer"]))
    print("\tlr:{:.2e}".format(hyperparameters["lr"]))
    print("\tloss_fn:{:s}".format(hyperparameters["loss"]))    
       
    # extract hyperparameters for the dict 'hyperparameters'
    batch_size = int(hyperparameters["batch_size"])
    n_epochs   = int(hyperparameters["n_epochs"])
    optimizer  = str(hyperparameters["optimizer"])
    lr         = float(hyperparameters["lr"])
    loss_fn    = str(hyperparameters["loss"])
    
    # set default values for some hyperparameters
    if optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    
    if loss_fn.lower() == "mse":
        loss_fn = MSELoss()
    
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
    val_loss   = np.full(n_epochs,np.nan)
    train_std  = np.full(n_epochs,np.nan)
    train_loss = np.full(n_epochs,np.nan)
    train_loss_one_epoch = np.full(batches_per_epoch,np.nan)

    # compute the real values of the validation dataset only once
    print("\nCompute validation dataset output:")
    # if out_shape is None:
    #     out_shape = tuple(get_real(val_dataset[0]).shape)
    # y_val = torch.zeros((len(val_dataset),*out_shape))
    # for n,X in enumerate(val_dataset):
    #     y_val[n,:] = torch.tensor( get_real(X) )
    # y_val = y_val#.flatten()
    y_val = get_real(dataloader_val)
    
    # start the training procedure
    for epoch in range(n_epochs):    
        
        with tqdm(enumerate(dataloader_train),\
                  total=batches_per_epoch,\
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as bar:

            for step,X in bar:

                # necessary to train the model
                model.train(True)

                # predict the value for the input X
                y_pred = get_pred(model,X)
                
                # true value for the value X
                y_train = get_real(X)
                
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
                bar.set_postfix(epoch=epoch,\
                                test=val_loss[epoch-1] if epoch != 0 else 0.0,\
                                loss=np.mean(train_loss_one_epoch[:step+1]))

            train_loss[epoch] = np.mean(train_loss_one_epoch)
            train_std [epoch] = np.std(train_loss_one_epoch)

            # evaluate model on the test dataset
            with torch.no_grad():
                model.eval()

                # predict the value for the validation dataset
                y_pred = get_pred(model,dataloader_val)

                # compute the loss function
                val_loss[epoch] = loss_fn(y_pred,y_val)

            # print progress
            bar.set_postfix(epoch=epoch,\
                            val=val_loss[epoch],\
                            loss=train_loss[epoch])
            
    arrays = pd.DataFrame({ "train_loss":train_loss,\
              "train_std":train_std,\
              "val_loss":val_loss})
            
    # deepcopy the trained model into the output variable
    out_model = copy(model)
    # restore the original value of 'model'
    model = in_model

    print("\nTraining done!\n")
    
    return out_model, arrays