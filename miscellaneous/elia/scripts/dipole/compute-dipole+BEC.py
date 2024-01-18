#!/usr/bin/env python
import argparse
import numpy as np
import os
import sys
from ase.io import read
# from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.nn.functions import get_model

#####################

description = "Compute dipole and BEC tensors for a given atomic structure.\n"

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar":"\b"}
    parser.add_argument(
        "-i","--instructions", action="store", type=str, **argv,
        help="model input file (default: 'instructions.json')", default="instructions.json"
    )

    parser.add_argument(
        "-p","--parameters", action="store", type=str, **argv,
        help="torch parameters file (default: 'parameters.pth')", default="parameters.pth",
    )

    parser.add_argument(
        "-q","--positions", action="store", type=str, **argv,
        help="file with positions and cell (in a.u.)"
    )

    parser.add_argument(
        "-o","--output", action="store", type=str, **argv,
        help="prefix for the output files (default: 'test')", default="test"
    )

    # parser.add_argument(
    #     "-f","--folder", action="store", type=str,
    #     help="folder", default='%-.10f'
    # )

    return parser.parse_args()

def main():

    #####################
    # Print the script's description
    print("\n\t{:s}".format(description))
    
    #####################
    # Parse the command-line arguments
    print("\tReading input arguments ... ",end="")
    args = get_args()
    print("done")

    #####################
    # load the model
    # print("\tLoading the model ... ",end="")
    file_in = os.path.normpath("{:s}".format(args.instructions))
    file_pa = os.path.normpath("{:s}".format(args.parameters))
    model = get_model(file_in,file_pa)
    # print("done")

    # ######################
    # # read the positions
    # instructions = { "positions" : args.positions }
    # if model.pbc :
    #     instructions["cells"] = args.cell if args.cell is not None else args.positions

    # print("\n\tReading the positions and cell ... ",end="")
    original_stdout = sys.stdout
    with open('/dev/null', 'w') as devnull:
        sys.stdout = devnull  # Redirect stdout to discard output
        # atoms = MicroState(instructions=instructions)
        atoms = read(args.positions)
    sys.stdout = original_stdout

    model.store_chemical_species(atoms=atoms)

    # if not model.pbc :
    #     atoms.cells = [None]*len(atoms.positions)    

    ######################
    print("\tComputing predicted values ... ",end="")
    # N = len(atoms)
    D = np.full(3,np.nan)
    Z = np.full((len(atoms.positions),3),np.nan)

    # for n,(pos,cell) in enumerate(zip(atoms.positions,atoms.cells)):
    pos = atoms.positions
    cell = np.asarray(atoms.cell).T if np.all(atoms.get_pbc()) else None
    d,z,x = model.get_value_and_jac(pos=pos.reshape((-1,3)),cell=cell)
    D = d.detach().numpy()#.flatten()
    Z = z.detach().numpy()#.flatten()

    print("done")

    ######################
    file = os.path.normpath( "{:s}.dipole.txt".format(args.output))
    print("\tSaving dipole to file '{:s}' ... ".format(file),end="")
    # with open(file,'w') as f:
    # for n in range(N):
    np.savetxt(file,D,delimiter='\t')
    print("done")

    ######################
    file = os.path.normpath("{:s}.bec.txt".format(args.output))
    print("\tSaving BECs to file '{:s}' ... ".format(file),end="")
    # with open(file,'w') as f:
    # for n in range(N):
    np.savetxt(file,Z,delimiter='\t')
    # f.write("\n")
    print("done")

    print("\n\tJob done :)\n")

#####################

if __name__ == "__main__":
    main()