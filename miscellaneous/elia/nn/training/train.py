# from copy import copy
import torch
from torch.nn import MSELoss
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import os
import time
from copy import copy
from miscellaneous.elia.functions import (
    add_default,
    remove_empty_folder,
    remove_files_in_folder,
)
from miscellaneous.elia.nn.plot import plot_learning_curves
from miscellaneous.elia.nn.training.functions import (
    save_checkpoint,
    get_all_dataloader,
    get_all,
)


__all__ = ["train"]

yval_real = None
all_dataloader_val = None

ytrain_real = None
all_dataloader_train = None

#----------------------------------------------------------------#
# Attention!
# There is a bit of confusion between 'parameters' and 'parameters':
# some variables are stored in the former and not in the latter (or viceversa) just for simplicity.
#----------------------------------------------------------------#

def train(
    model: torch.nn.Module,
    train_dataset: list,
    val_dataset: list,
    parameters: dict,
    hyperparameters: dict = None,  #   get_pred:callable=None,\
    #   get_real:callable=None,\
    make_dataloader: callable = None,  # correlation:callable=None,\
    output=None,
    name=None,
    opts=None,
):
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

        output (str, optional):
            Folder name for saving training information. Defaults to None.

        name (str, optional):
            A filename to distinguish results from other hyperparameters. Defaults to None.

        opts (dict, optional):
            Additional options for training. Defaults to None.

    Returns:
        tuple:
            A tuple containing the trained model, dataframe, correlation, and information about the training.
    """

    start_task_time = time.time()

    print("\nTraining:")
    print("\tPreparing training")

    # information about the status of the training
    info = "all good"

    ##########################################
    default = {
        "plot": {"learning-curve": {"N": 10}},
        "dataloader": {"shuffle": False},
        "thr": {"exit": 10000},
        "disable": False,
        "restart": False,
        "recompute_loss": False,
        "save": {"parameters": 1},
    }
    opts = add_default(opts, default)

    ##########################################
    # default values
    if make_dataloader is None:
        from miscellaneous.elia.nn.dataset import make_dataloader as _make_dataloader
        make_dataloader = \
            lambda dataset, batch_size, shuffle=opts["dataloader"]["shuffle"]: \
                _make_dataloader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    if hyperparameters is None:
        hyperparameters = dict()

    # output folder
    if output is None:
        warnings.warn("'output' is None: specify a folder name to print some information to file.\n\
                      'results' will be set as default")
        output = "results"

    # the name of the output files
    if name is None:
        warnings.warn("'name' is None: specify a filename to distinguish the results from other hyperparameters")
        name = "untitled"

    ##########################################
    # output folders
    folders = {
        "parameters": "{:s}/parameters".format(output),
        "dataframes": "{:s}/dataframes".format(output),
        "images": "{:s}/images".format(output),
        "correlations": "{:s}/correlations".format(output),
    }

    # create the output folders
    for folder in [output, *folders.values()]:
        if not os.path.exists(folder):
            print("\tCreating folder '{:s}'".format(folder))
            os.mkdir(folder)

    ##########################################
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

    print("\tHyperparameters:")
    for k in hyperparameters.keys():
        try :
            print("\t\t{:20s}: ".format(k),hyperparameters[k])
        except:
            if k == "loss":
                if type(hyperparameters["loss"]) == str:
                    print("\tloss_fn: ", hyperparameters["loss"])

    # print("\t\tbatch_size: ", hyperparameters["bs"])
    # print("\t\tn_epochs: ", hyperparameters["n_epochs"])
    # print("\t\toptimizer: ", hyperparameters["optimizer"])
    # print("\t\tlr: ", hyperparameters["lr"])
    # # I had some problems with the loss
    # if type(hyperparameters["loss"]) == str:
    #     print("\tloss_fn: ", hyperparameters["loss"])
    # if "weight_decay" in hyperparameters:
    #     print("\t\weight_decay: ", hyperparameters["weight_decay"])


    # extract hyperparameters for the dict 'hyperparameters'
    batch_size = int(hyperparameters["bs"])
    n_epochs = int(hyperparameters["n_epochs"])
    optimizer = parameters["optimizer"]
    # lr = float(hyperparameters["lr"])
    loss_fn = hyperparameters["loss"]

    # set default values for some hyperparameters
    if type(optimizer) == str: 
        match optimizer.lower():
            case "adam":
                from torch.optim import Adam
                optimizer = Adam(   params=model.parameters(), 
                                    lr=hyperparameters["lr"])
            case "adamw":
                from torch.optim import AdamW
                optimizer = AdamW(  params=model.parameters(), 
                                    lr=hyperparameters["lr"],
                                    weight_decay=parameters["weight_decay"])

    ##########################################
    # prepare the dataloaders for the train and validation datasets
    dataloader_train = make_dataloader(train_dataset, batch_size)
    batches_per_epoch = len(dataloader_train)
    train_loss_one_epoch = np.full(batches_per_epoch, np.nan)

    # give a summary of the length of the following for cycles
    print("\n\tSummary:")
    print("\t      n. of epochs:", n_epochs)
    print("\t        batch size:", batch_size)
    print("\t  n. of iterations:", batches_per_epoch)
    print("\ttrain dataset size:", len(train_dataset))
    print("\t  val dataset size:", len(val_dataset))
    print("\n")

    ##########################################
    # dataframe
    dataframe = pd.DataFrame(
        np.nan,
        columns=["epoch", "train", "val", "std", "ratio","lr"],
        index=np.arange(n_epochs),
    )
    if opts["recompute_loss"]:
        dataframe["train-2"] = None
        dataframe["ratio-2"] = None
    else:
        dataframe["ratio"] = None

    ##########################################
    # compute the real values of the validation dataset only once
    global yval_real, all_dataloader_val
    yval_real = None
    if yval_real is None or not opts["keep_dataset"]:
        print("\tCompute validation dataset output (this will save time in the future)")
        argv = {
            "dataset": val_dataset,
            "make_dataloader": make_dataloader,
        }
        yval_real = get_all(model, **argv)
        all_dataloader_val = get_all_dataloader(**argv)

    global ytrain_real, all_dataloader_train
    ytrain_real = None
    if ytrain_real is None or not opts["keep_dataset"] and opts["recompute_loss"]:
        print("\tCompute training dataset output (this will save time in the future)")
        argv = {
            "dataset": train_dataset,
            "make_dataloader": make_dataloader,
        }
        ytrain_real = get_all(model, **argv)
        all_dataloader_train = get_all_dataloader(**argv)

    ##########################################
    # prepare output files
    savefiles = {
        "dataframes": "{:s}/{:s}.csv".format(folders["dataframes"], name),
        "images": "{:s}/{:s}.pdf".format(folders["images"], name),
    }

    ##########################################
    # prepare checkpoint
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    start_epoch = 0

    checkpoint_folder = "checkpoint"
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    parameters_folder = "{:s}/{:s}".format(folders["parameters"], name)
    if not os.path.exists(parameters_folder):
        os.mkdir(parameters_folder)
    elif parameters["restart"]:
        print("\tCleaning parameters folder '{:s}'".format(parameters_folder))
        remove_files_in_folder(parameters_folder, "pth")

    checkpoint_file = "{:s}/{:s}.pth".format(checkpoint_folder, name)
    if os.path.exists(checkpoint_file) and not opts["restart"]:
        print("\tReading checkpoint from file '{:s}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        # read array
        if os.path.exists(savefiles["dataframes"]):
            tmp = pd.read_csv(savefiles["dataframes"])
            if not opts["recompute_loss"] and "train-2" in tmp:
                dataframe["train-2"] = None
                dataframe["ratio-2"] = None
            dataframe.iloc[: len(tmp)] = copy(tmp)
            del tmp

        if opts["recompute_loss"] and "train-2" not in dataframe:
            dataframe["train-2"] = None
            dataframe["ratio-2"] = None
    elif not os.path.exists(checkpoint_file):
        print("\tno checkpoint file found")
    elif opts["restart"]:
        print("\trestart=True")

    ##########################################
    print("\n\tOptimizer parameters")
    for k,i in optimizer.param_groups[0].items():
        if k == 'params' : 
            continue
        print("\t\t",k,": ",i)

    ##########################################
    # start the training procedure
    print("\n\t...and here we go!")
    k = copy(start_epoch)
    for epoch in range(start_epoch, n_epochs):

        if info != "all good":
            break

        ##########################################
        # measure elapsed time
        now = time.time()
        if (
            abs(now - opts["start_time"] + 60) > parameters["max_time"]
            and parameters["max_time"] > 0
        ):
            print("\n\tMaximum time reached: stopping")
            info = "time over"
            break
        if (
            abs(now - start_task_time + 60) > parameters["task_time"]
            and parameters["task_time"] > 0
        ):
            print("\n\tMaximum time per task reached: stopping task")
            info = "time over"
            break

        if os.path.exists("EXIT"):
            info = "exit file detected"
            print("\n\t'EXIT' file detected")
            break

        if os.path.exists("EXIT-TASK"):
            os.remove("EXIT-TASK")
            info = "exit-task file detected"
            print("\n\t'EXIT-TASK' file detected")
            break

        ##########################################
        # training loop per epoch
        with tqdm(
            enumerate(dataloader_train),
            total=batches_per_epoch,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=opts["disable"],
        ) as bar:

            ##########################################
            # cycle over mini-batches
            for step, X in bar:

                # necessary to train the model
                model.train(mode=True)

                # predict the value for the input X
                y_pred = model.get_pred(X=X)  # get_pred(model=model,X=X)

                # true value for the value X
                y_real = model.get_real(X=X)

                # compute the loss function
                loss = loss_fn(y_pred, y_real)

                # store the loss function in an array
                train_loss_one_epoch[step] = float(loss)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                
                # print progress
                bar.set_postfix(
                    epoch=epoch,
                    train=np.mean(train_loss_one_epoch[: step + 1]),
                    val=dataframe.at[epoch - 1, "val"] if epoch != 0 else np.nan,
                )

            ##########################################
            # finished cyclying over mini-batches

            if model.use_shift:
                print("\t!! SHIFT:", model.shift.detach().numpy())
                print("\t!! FACTOR:", model.factor.detach().numpy())

            # evaluate model on the test dataset
            # with torch.no_grad():
            # I am not using 'with torch.no_grad()' anymore because
            # maybe it inferferes with the calculation of the forces
            # model.eval()
            # if True:  # with torch.no_grad():

            # save learning rate
            dataframe.at[epoch,"lr"] = float(optimizer.param_groups[0]["lr"])

            model.eval()

            dataframe.at[epoch, "epoch"] = epoch + 1

            # saving parameters to temporary file
            N = opts["save"]["parameters"]
            if N != -1 and epoch % N == 0:
                savefile = "{:s}/epoch={:d}.pth".format(parameters_folder, epoch)
                print("\tsaving parameters to file '{:s}'".format(savefile))
                torch.save(model.state_dict(), savefile)

            dataframe.at[epoch, "train"] = np.mean(train_loss_one_epoch)
            dataframe.at[epoch, "std"] = np.std(train_loss_one_epoch)

            # compute the loss function
            # predict the value for the validation dataset
            yval_pred = model.get_pred(all_dataloader_val)
            dataframe.at[epoch, "val"] = float(loss_fn(yval_pred, yval_real))
            if not opts["recompute_loss"]:
                dataframe.at[epoch, "ratio"] = (
                    dataframe.at[epoch, "train"] / dataframe.at[epoch, "val"]
                )

            # set dataframe
            if opts["recompute_loss"]:
                ytrain_pred = model.get_pred(all_dataloader_train) 
                dataframe.at[epoch, "train-2"] = float(loss_fn(ytrain_pred, ytrain_real))
                dataframe.at[epoch, "ratio-2"] = (dataframe.at[epoch, "train-2"] / dataframe.at[epoch, "val"])

            if dataframe.at[epoch, "train"] > opts["thr"]["exit"] and opts["thr"]["exit"] > 0:
                info = "try again"
                break

            dataframe[: epoch + 1].to_csv(savefiles["dataframes"], index=False)

            # produce learning curve plot
            if epoch >= 1:                
                plot_learning_curves(
                    arrays=dataframe,
                    file=savefiles["images"],
                    title=name if name != "untitled" else None,
                    opts=opts["plot"]["learning-curve"],
                )
            
            # print progress
            bar.set_postfix(
                epoch=epoch,
                train=dataframe.at[epoch, "train"],
                val=dataframe.at[epoch, "val"],
            )

        # saving checkpoint to file
        N = opts["save"]["checkpoint"]
        if N != -1 and epoch % N == 0:
            save_checkpoint(checkpoint_file, epoch, model, optimizer)

        k += 1
    #
    save_checkpoint(checkpoint_file, k - 1, model, optimizer)

    if info == "all good":

        # removing empty folders
        _folders = copy(folders)
        for k in folders.keys():
            if remove_empty_folder(folders[k], show=False):
                del _folders[k]

        # Important message
        print("\tThe following quantities have been saved to these folders:")
        for k in _folders.keys():
            print("\t{:<20s}: {:<20s}".format(k, folders[k]))

        print("\n\tTraining done!\n")

    # Something wrong happened during the training
    # We will let the user know about that through the variable 'info'
    elif info == "try again":
        print("\n\tTraining stopped: we could try again\n")
        if os.path.exists(checkpoint_file):
            print("\tRemoving checkpoint file '{:s}'".format(checkpoint_file))
            os.remove(checkpoint_file)

    elif info == "exit-task file detected":
        print("\n\t'exit-task' file detected\n")

    return model, dataframe, info
