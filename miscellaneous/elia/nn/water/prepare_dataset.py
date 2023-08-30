import os
import json
import torch
import random
from .make_dataset_delta import make_dataset_delta # miscellaneous.elia.nn.water.make_dataset_delta
from .make_dataset import make_dataset # miscellaneous.elia.nn.water.make_dataset
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import add_default

def prepare_dataset(ref_index:int,\
                    max_radius:float,\
                    reference:bool,\
                    variables:list,\
                    folder:str="data",\
                    opts:dict=None,\
                    requires_grad:bool=False):
    
    # Attention:
    # Please keep 'requires_grad' = False
    if opts is None :
        opts = {}
    default = {
                "prepare":{
                    "restart": False,
                    "read":True,
                    "save":True
                },
                "build":{
                    "restart": False,
                    "read":True,
                    "save":True
                },
                "size":
                {
                    "train":1000,
                    "val":100,
                    "test":100,
                },
            }

    opts = add_default(opts,default)

    print("\n\tPreparing datasets:")

    RESTART = opts["prepare"]["restart"]
    READ = opts["prepare"]["read"]
    SAVE = opts["prepare"]["save"]
    savefile = "{:s}/microstate.pickle".format(folder)

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "{:s}/i-pi.positions_0.xyz".format(folder)
        instructions = {"properties" : "{:s}/i-pi.properties.out".format(folder),\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"{:s}/i-pi.forces_0.xyz".format(folder)}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ##########################################
    # show time-series
    f = "{:s}/time-series".format(folder)
    if not os.path.exists(f):
        os.mkdir(f)
    for var in variables:
        filename = "{:s}/{:s}.pdf".format(f,var)

        if var == "electric-dipole":
            _ = data.get_dipole(same_lattice=False)

        data.plot_time_series(what=var,file=filename)

    ########################################## 

    RESTART = opts["build"]["restart"]
    READ = opts["build"]["read"]
    SAVE = opts["build"]["save"]
    savefile = "{:s}/{:s}".format(folder,"dataset-delta" if reference else "dataset")

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("\tBuilding datasets")

        if os.path.exists(savefile+".torch") and not RESTART:
            dataset = torch.load(savefile+".torch")
            if reference:
                dipole  = dataset[ref_index].dipole
                pos     = dataset[ref_index].pos
            else :
                dipole = torch.full((3,),torch.nan)
                pos = torch.full((3,),torch.nan)
        else :
            if reference :
                dataset, dipole, pos = make_dataset_delta(  ref_index = ref_index,
                                                            data = data,
                                                            max_radius = max_radius,\
                                                            requires_grad = requires_grad)
            else :
                dataset = make_dataset( data=data,max_radius=max_radius,requires_grad=requires_grad)
                dipole = torch.full((3,),torch.nan)
                pos = torch.full((3,),torch.nan)
        # shuffle
        random.shuffle(dataset)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = opts["size"]["train"]
        i = opts["size"]["val"] #int(p_test*len(dataset))
        j = opts["size"]["test"]#int(p_val*len(dataset))

        train_dataset = dataset[:n]
        val_dataset   = dataset[n:n+j]
        test_dataset  = dataset[n+j:n+j+i]

        del dataset

    else :
        print("\tReading datasets from file {:s}".format(savefile))
        train_dataset = torch.load(savefile+".train.torch")
        val_dataset   = torch.load(savefile+".val.torch")
        test_dataset  = torch.load(savefile+".test.torch")

        if reference :
            # Open the JSON file and load the data
            with open("reference.json") as f:
                reference = json.load(f)
            dipole = torch.tensor(reference['dipole'])
            pos    = torch.tensor(reference['pos'])
        else :
            dipole = torch.full((3,),torch.nan)
            pos = torch.full((3,),torch.nan)

        SAVE = False
            
    if SAVE :
        print("\tSaving dataset to file {:s}".format(savefile))
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")

        if reference :
            # Write the dictionary to the JSON file
            with open("reference.json", "w") as json_file:
                # The 'indent' parameter is optional for pretty formatting
                json.dump({"dipole":dipole.tolist(),"pos":pos.tolist()}, json_file, indent=4)  

    print("\n\tDatasets summary:")
    print("\t\ttrain:",len(train_dataset))
    print("\t\t  val:",len(val_dataset))
    print("\t\t test:",len(test_dataset))

    datasets = {"train":train_dataset,\
                "val"  :val_dataset,\
                "test" :test_dataset }

    return datasets, data, dipole, pos