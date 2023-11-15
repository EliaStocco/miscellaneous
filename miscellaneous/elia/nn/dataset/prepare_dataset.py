import os
import json5 as json
import torch
import random
from copy import copy
from ase import Atoms
from miscellaneous.elia.nn.dataset import make_dataset
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import add_default, find_files_by_pattern

# ELIA: modify 'same_lattice' to false
same_lattice = True

def prepare_dataset(# ref_index:int,\
                    max_radius:float,\
                    output:str,\
                    pbc:bool,\
                    # reference:bool,\
                    folder:str,\
                    opts:dict,\
                    indices:str,\
                    requires_grad:bool=False):
    
    # Attention:
    # Please keep 'requires_grad' = False
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
                # "shift" : None,
                "instructions" : None
            }

    opts = add_default(opts,default)

    print("\n\tPreparing datasets:")

    RESTART = opts["prepare"]["restart"]
    READ = opts["prepare"]["read"]
    SAVE = opts["prepare"]["save"]
    savefile = "{:s}/microstate.pickle".format(folder)

    if not READ or not os.path.exists(savefile) or RESTART :
        instructions = opts["instructions"]
        if instructions is None :
            infile = find_files_by_pattern (folder,"positions",1) # "{:s}/i-pi.positions_0.xyz".format(folder)
            instructions = {
                "properties" : find_files_by_pattern (folder,"properties",1), # "{:s}/i-pi.properties.out".format(folder),\
                "positions":infile,
                "types":infile
                }
            if pbc :
                instructions["cells"] = infile
            if "F" in output : 
                instructions["forces"] = find_files_by_pattern (folder,"forces",1) # "{:s}/i-pi.forces_0.xyz".format(folder)
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ##########################################
    # fix polarization
    if "D" in output :
        # shift = [0,0,0]
        if "dipole" not in data.properties :
            data.get_dipole(same_lattice=same_lattice,inplace=True)

    ########################################## 

    RESTART = opts["build"]["restart"]
    READ = opts["build"]["read"]
    SAVE = opts["build"]["save"]

    name = "dataset"
    savefile = "{:s}/{:s}".format(folder,name)

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("\tBuilding datasets")

        if os.path.exists(savefile+".torch") and not RESTART:
            dataset = torch.load(savefile+".torch")
        else :
            dataset = make_dataset( data=data,
                                    max_radius=max_radius,
                                    output=output,
                                    pbc = pbc ,
                                    indices = indices,
                                    requires_grad=requires_grad)
        not_shuffled = copy(dataset)


        n = opts["size"]["train"]
        i = opts["size"]["val"] 
        j = opts["size"]["test"]

        train_dataset   = copy(dataset[:n])
        val_dataset     = copy(dataset[n:n+j])
        test_dataset    = copy(dataset[n+j:n+j+i])
        unused_dataset  = copy(dataset[n+j+i:])

        del dataset

    else :
        print("\tReading datasets from file {:s}".format(savefile))
        not_shuffled   = torch.load(savefile+".all.torch")
        train_dataset  = torch.load(savefile+".train.torch")
        val_dataset    = torch.load(savefile+".val.torch")
        test_dataset   = torch.load(savefile+".test.torch")
        unused_dataset = torch.load(savefile+".unused.torch")

        SAVE = False
            
    if SAVE :
        print("\tSaving dataset to file {:s}".format(savefile))
        torch.save(not_shuffled,savefile+".all.torch")
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")
        torch.save(unused_dataset, savefile+".unused.torch") 

    print("\n\tDatasets summary:")
    print("\t\ttrain:",len(train_dataset))
    print("\t\t  val:",len(val_dataset))
    print("\t\t test:",len(test_dataset))

    datasets = {"train"  : train_dataset,\
                "val"    : val_dataset,\
                "test"   : test_dataset,\
                "unsused": unused_dataset,\
                "all"    : not_shuffled}
    
    pos  = train_dataset[0].pos.numpy()
    symbols = data.types[0]

    if pbc :
        cell = train_dataset[0].lattice[0].numpy()
        # check that the 'cell' format is okay
        example = Atoms(positions=pos,cell=cell.T,symbols=symbols)
    else :
        example = Atoms(positions=pos,symbols=symbols)

    return datasets, example