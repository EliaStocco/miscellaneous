#!/usr/bin/env python
from ase.io import write
import numpy as np
import json
from ase import Atoms
from miscellaneous.elia.classes.trajectory import trajectory as Trajectory
from miscellaneous.elia.formatting import esfmt, warning, float_format
from miscellaneous.elia.formatting import matrix2str
from miscellaneous.elia.physics import oxidation_number

#---------------------------------------#
# Description of the script's purpose
description = "Compute the BECs from the nominal oxidation numbers."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="extxyz input file with an atomic structure and oxidation numbers")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-k" , "--keyword"       , **argv, required=False, type=str, help="keyword for the oxidation numbers (default: 'oxidation-numbers')", default='oxidation-numbers')
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str, help="txt output file with the BECs (default: 'bec.txt')", default='bec.txt')
    return parser.parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = Trajectory(args.input,format=args.input_format,index=0)[0]
    print("done")

    #------------------#
    print("\tExtracting oxidation numbers from '{:s}' ... ".format(args.keyword), end="")
    on = atoms.arrays[args.keyword]
    print("done")

    #------------------#
    print("\tCreating BECs ... ", end="")
    Natoms = atoms.get_global_number_of_atoms()
    bec = np.zeros((Natoms,3,3))
    for n in range(Natoms):
        bec[n,:,:] = on[n] * np.eye(3)
    bec = bec.reshape((-1,3))
    print("done")

    #------------------#
    if not np.allclose(bec.sum(axis=0),np.zeros(3)):
        print("\t{:s}: the BECs do not satisfy the acoutsitc sum rule --> subtracting the average.".format(warning))
        bec_old = bec.copy()
        mean = bec.mean()
        print("\tSubtracted average: {:f}".format(mean))
        bec -= mean

    #------------------#    
    args.output = str(args.output)
    print("\n\tSaving BECs to file '{:s}' ... ".format(args.output), end="")
    np.savetxt(args.output, bec,fmt=float_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()