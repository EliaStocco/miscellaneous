#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from torch_geometric.data import Data
import torch
from miscellaneous.elia.formatting import esfmt, warning
from miscellaneous.elia.show import show_dict

#---------------------------------------#
# Description of the script's purpose
description = "Convert a 'torch' dataset into an 'extxyz' file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, required=True , help="*.pth file with the torch dataset (default: 'dataset.pth')", default="dataset.pth")
    parser.add_argument("-o", "--output"   , **argv,type=str, required=False, help="output extxyz file (default: 'dataset.extxyz')", default="dataset.extxyz")
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading dataset from file '{:s}' ... ".format(args.input), end="")
    dataset = torch.load(args.input)
    print("done")

    N = len(dataset)
    print("\tdataset size: ",N)


    #------------------#
    warnings = dict()
    atoms = [None]*N
    for n,entry in enumerate(dataset):
        print("\tConverting dataset: structure {:d}/{:d}".format(n+1,N),end="\r")
        positions = entry.pos.detach().numpy()
        cell = entry.lattice.detach().numpy()[0,:,:]
        symbols = entry.symbols
        atoms[n] = Atoms(positions=positions,cell=cell,symbols=symbols,pbc=cell is not None)
        Natoms = atoms[n].get_global_number_of_atoms()
        # atoms[n].info["dipole"] = test[n].dipole.detach().numpy()

        for key, value in entry.items():
            if key in ["pos","lattice","pbc","symbols"]:
                continue
            if type(value) in [float,int,str]:
                atoms[n].info["key"] = value
            else:
                if value.ndim == 1:
                    atoms[n].info[key] = value.detach().numpy().reshape((-1,))
                if value.ndim == 2:
                    if value.shape[0] == Natoms :
                        atoms[n].arrays[key] = value.detach().numpy().reshape((Natoms,-1))
                    else:
                        warnings.update({key:list(value.shape)})
                
    print("\t{:s}: it was not possible to save this information to file due to their shape:".format(warning))
    show_dict(warnings,string="\t\t",width=10)

    #------------------#
    print("\n\tWriting atomic structure to file file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=atoms,filename=args.output) # fmt)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

#---------------------------------------#
if __name__ == "__main__":
    main()