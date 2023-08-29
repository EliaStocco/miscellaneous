import os
import json
import torch
import random
from . import make_dataset_delta # miscellaneous.elia.nn.water.make_dataset_delta
from . import make_dataset # miscellaneous.elia.nn.water.make_dataset
from miscellaneous.elia.classes import MicroState

def prepare_dataset(ref_index:int,max_radius:float,reference:bool):

    RESTART = False
    READ = True
    SAVE = True
    savefile = "data/microstate.pickle"

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "data/i-pi.positions_0.xyz"
        instructions = {"properties" : "data/i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"data/i-pi.forces_0.xyz"}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ########################################## 

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/dataset-delta" if reference else "data/dataset"

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("building dataset")

        if os.path.exists(savefile+".torch") and not RESTART:
            dataset = torch.load(savefile+".torch")
            if reference:
                dipole  = dataset[ref_index].dipole
                pos     = dataset[ref_index].pos
            else :
                dipole = None
                pos = None
        else :
            if reference :
                dataset, dipole, pos = make_dataset_delta(  ref_index = ref_index,
                                                            data = data,
                                                            max_radius = max_radius)
            else :
                dataset = make_dataset( data=data,max_radius=max_radius)
                dipole = None
                pos = None
        # shuffle
        random.shuffle(dataset)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = 100
        i = 10#int(p_test*len(dataset))
        j = 10#int(p_val*len(dataset))

        train_dataset = dataset[:n]
        val_dataset   = dataset[n:n+j]
        test_dataset  = dataset[n+j:n+j+i]

        del dataset

    else :
        print("reading datasets from file {:s}".format(savefile))
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
            dipole = None
            pos = None

        SAVE = False
            
    if SAVE :
        print("saving dataset to file {:s}".format(savefile))
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")

        if reference :
            # Write the dictionary to the JSON file
            with open("reference.json", "w") as json_file:
                # The 'indent' parameter is optional for pretty formatting
                json.dump({"dipole":dipole.tolist(),"pos":pos.tolist()}, json_file, indent=4)  

    print("train:",len(train_dataset))
    print("  val:",len(val_dataset))
    print(" test:",len(test_dataset))

    datasets = {"train":train_dataset,\
                "val"  :val_dataset,\
                "test" :test_dataset }

    return datasets, data, dipole, pos