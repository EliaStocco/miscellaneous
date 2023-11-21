#!/usr/bin/env python
import argparse
import numpy as np
from ase.io import write, read
from copy import copy
from miscellaneous.elia.functions import cart2lattice, lattice2cart

description = "Fix the dipole jumps and shift the values of some multitples of the dipole quantum.\n"
message = "\t!Attention:\n"+\
    "\t- you need to provide the data as a 'extxyz' file\n"+\
    "\t- the atomic structure have to represent an MD trajectory (the order matters!)"+\
    "\n"
DEBUG=False

def list_type(s):
    return [ int(i) for i in s.split(" ") ]

def prepare_args():

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}

    parser.add_argument("-i", "--input", type=str, **argv,
                        help="input 'extxyz' file")
    
    parser.add_argument("-o", "--output", type=str, default='shifted&fixed.extxyz', **argv,
                        help="output 'extxyz' file (default: 'shifted&fixed.extxyz')")

    parser.add_argument("-s", "--shift",  type=list_type, default=None, **argv,
                        help="additional arrays to be added to the output file")
    
    return parser.parse_args()

def main():

    ###
    # Parse the command-line arguments
    args = prepare_args()

    ###
    # Print the script's description
    print("\n\t{:s}".format(description))

    if args.shift is not None:
        if len(args.shift) != 3 :
            raise ValueError("You should provide 3 integer numbers as shift vectors")
    
    print(message)

    ###
    # read the MD trajectory from file
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input,format='extxyz',index=":")
    print("done")

    print("\tConverting dipoles from cartesian to lattice coordinates ... ", end="")
    N = len(atoms)
    phases = np.full((N,3),np.nan)
    lenght = np.full((N,3),np.nan)
    # old = np.full((N,3),np.nan)
    for n in range(N):
        atoms[n].set_calculator(None)
        cell = np.asarray(atoms[n].cell.array).T
        lenght[n,:] = np.linalg.norm(cell,axis=0)
        R = cart2lattice(cell)
        dipole = R @ atoms[n].info["dipole"]
        phases[n,:] = dipole / lenght[n,:]
        # old[n,:] = copy(atoms[n].info["dipole"])
    print("done")

    print("\tFixing the dipole jumps {:s}... ", end="")
    for i in range(3):
        phases[:,i] = np.unwrap(2*np.pi*phases[:,i])/(2*np.pi)
    print("done")

    if args.shift is not None:
        shift = np.asarray([ int(i) for i in phases.mean(axis=0) ])
        print("\tShifting the dipoles by the following quantum (on top of the global mean): ",args.shift," ... ", end="")
        shift += args.shift
        for i in range(3):
            phases[:,i] -= shift[i]
        print("done")

    print("\tConverting dipoles from lattice to cartesian coordinates ... ", end="")
    for n in range(N):
        cell = np.asarray(atoms[n].cell.array).T
        R = lattice2cart(cell)
        atoms[n].info["dipole"] = R @ ( phases[n,:] * lenght[n,:] )
    print("done")

    ###
    # writing
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, atoms, format="extxyz")
        print("done")
    except Exception as e:
        print(f"\n\tError: {e}")

    # Script completion message
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()
