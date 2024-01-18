#!/usr/bin/env python
import numpy as np
import os
from miscellaneous.elia.nn.functions import get_model
from miscellaneous.elia.functions import suppress_output
from ase.io import read
from e3nn.o3 import rand_matrix

#---------------------------------------#
# Description of the script's purpose
description = "Check the E(3)-equivariance of a neural network."
closure = "Job done :)"
input_arguments = "Input arguments"
DEBUG = True

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

def prepare_args():
    """Prepare parser of user input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--instructions", type=str, **argv, help="model input file (default: 'instructions.json')", default="instructions.json")
    parser.add_argument("-p" , "--parameters"  , type=str, **argv, help="torch parameters file (default: 'parameters.pth')", default=None)
    parser.add_argument("-s" , "--structure"   , type=str, **argv, help="file with an atomic structure [a.u.]")
    parser.add_argument("-n" , "--number"      , type=int, **argv, help="number of test to perform", default=100)
    return parser.parse_args()

#---------------------------------------#
def main():

    #---------------------------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.structure), end="")
    atoms = read(args.structure)
    print("done")

    #------------------#
    # print("\tLoading model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters)) if args.parameters is not None else None
    with suppress_output(not DEBUG):
        model = get_model(file_in,file_pa)
        model.store_chemical_species(atoms=atoms)
    # print("done")

    #------------------#
    pbc = np.all(atoms.get_pbc())
    pos = atoms.positions
    cell = np.asarray(atoms.cell) if np.all(atoms.get_pbc()) else np.full((3,3),np.nan)   
    def func(array):
        if pbc:
            cell = array[0:3,:].T
            pos  = array[3:,:]
        else:
            cell = None
            pos = array
        y,_ = model.get(pos=pos.reshape((-1,3)),cell=cell)
        return y.detach().numpy()
    
    if pbc:
        array = np.concatenate([cell,pos])
    else:
        array = pos

    #------------------#
    print("\n\tGenerating {:d} random rotation matrices ... ".format(args.number),end="")
    allR = rand_matrix(args.number)
    print("done")

    print("\tComparing 'outputs from rotated inputs' with 'rotated outputs' ... ",end="")
    y = func(array)
    norm = np.zeros(len(allR))
    for n,R in enumerate(allR):
        R = R.numpy()
        tmp = ( R @ array.T ).T
        Rx2y = func(tmp) # Rotated input (x) to output (y)
        Ry = R @ y       # Rotated output (y)
        norm[n] = np.linalg.norm(Rx2y - Ry)
    print("done")
    
    print("\tSummary of the norm between 'outputs from rotated inputs' and 'rotated outputs'")
    print("\t{:>20s}: {:.4e}".format("min norm",norm.min()))
    print("\t{:>20s}: {:.4e}".format("max norm",norm.max()))
    print("\t{:>20s}: {:.4e}".format("mean norm",norm.mean()))

    #------------------#
    print("\n\tGenerating {:d} random translation vectors ... ".format(args.number),end="")
    allT = np.random.rand(args.number, 3)
    print("done")

    print("\tComparing 'outputs from rotated inputs' with 'rotated outputs' ... ",end="")
    y = func(array)
    norm = np.zeros(len(allT))
    for n,T in enumerate(allT):
        if pbc:
            tmp = np.concatenate([cell,pos+T])
        else:
            tmp = pos+T 
        Tx2y = func(tmp) # Translated input (x) to output (y)
        norm[n] = np.linalg.norm(Tx2y - y)
    print("done")
    
    print("\tSummary of the norm between 'outputs from translated inputs' and 'outputs'")
    print("\t{:>20s}: {:.4e}".format("min norm",norm.min()))
    print("\t{:>20s}: {:.4e}".format("max norm",norm.max()))
    print("\t{:>20s}: {:.4e}".format("mean norm",norm.mean()))

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()



print()