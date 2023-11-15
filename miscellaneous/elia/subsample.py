#!/usr/bin/env python
import argparse
from copy import copy
import numpy as np
from ase.io import write, read
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import str2bool, suppress_output

# Description of the script's purpose
description = "Sumsample an (ASE readable) MD trajectory given a set of indices."


def prepare_args():

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Define command-line arguments

    argv = {"metavar" : "\b"}
    parser.add_argument("-i", "--input",  type=str, default='i-pi.positions_0.xyz', **argv,
                        help="input file containing the MD trajectory (default: 'i-pi.positions_0.xyz')")

    parser.add_argument("-f", "--format", type=str, default='extxyz', **argv, 
                        help="file format (default: extxyz)" )
    
    parser.add_argument("-n", "--indices", type=str, **argv, default='indices.txt',
                        help="txt file with the subsampling indices (default: 'indices.txt')")

    parser.add_argument("-o", "--output", type=str, **argv, 
                        help="output file")



    return parser.parse_args()

def main():
   
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))


    # Read the MicroState data from the input file
    instructions = {
        "cells": args.input,      # Use the input file for 'cells' data
        "positions": args.input,  # Use the input file for 'positions' data
        "types": args.input       # Use the input file for 'types' data
    }

    print("\tReading atomic structures from file '{:s}' using the 'ase.io.read' with format '{:s}' ... ".format(args.input,args.format), end="")
    atoms = read(args.input,format=args.format,index=":")
    print("done")

    print("\tReading subsampling indices from file '{:s}' ... ".format(args.indices), end="")
    indices = np.loadtxt(args.indices).astype(int)
    indices.sort()
    print("done")

    print("\tSubsampling atomic structures ... ".format(args.indices), end="")
    new_atoms = [None]*len(indices)
    for n,i in enumerate(indices):
        atoms[i].set_calculator(None)
        new_atoms[n] = copy(atoms[i])
    # atoms = list(np.array(atoms,dtype=object)[indices])
    print("done")

    # Write the data to the specified output file with the specified format
    print("\tWriting subsampled atomic structures to file '{:s}' with format '{:s}' ... ".format(args.output, args.format), end="")
    try:
        write(args.output, new_atoms, format=args.format)
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")

    # Script completion message
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()