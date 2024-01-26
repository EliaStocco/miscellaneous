#!/usr/bin/env python
import numpy as np
from ase.io import write, read
from copy import copy
from miscellaneous.elia.functions import cart2lattice, lattice2cart
from miscellaneous.elia.input import size_type, str2bool

description = "Fix the dipole jumps and shift the values of some multitples of the dipole quantum.\n"
message = "\t!Attention:\n"+\
    "\t- you need to provide the data as a 'extxyz' file\n"+\
    "\t- the atomic structure have to represent an MD trajectory (the order matters!)"+\
    "\n"
DEBUG=False

# def size_type(s):
#     s = s.split("[")[1].split("]")[0].split(",")
#     match len(s):
#         case 3:
#             return np.asarray([ int(k) for k in s ])
#         case _:
#             raise ValueError("You should provide 3 integers") 

def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input", type=str, **argv, help="input 'extxyz' file")
    parser.add_argument("-o", "--output", type=str, default='shifted-and-fixed.extxyz', **argv, help="output 'extxyz' file (default: 'shifted-and-fixed.extxyz')")
    parser.add_argument("-f", "--fix_only", type=str2bool, default=False, **argv, help="whether to fix only the jumps, without shifting (default: false)")
    parser.add_argument("-j", "--jumps", type=str, default=None, **argv, help="output txt file with jumps indeces (default: 'None')")
    parser.add_argument("-s", "--shift",  type=lambda s: size_type(s,dtype=float), default=None, **argv, help="additional vector to be added to the output file (default: [0,0,0])")
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

    print("\n\tConverting dipoles from cartesian to lattice coordinates ... ", end="")
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

    print("\tFixing the dipole jumps ... ", end="")
    old = phases.copy()
    for i in range(3):
        phases[:,i] = np.unwrap(phases[:,i],period=1)
    print("done")

    # if args.shift is not None:
    #     shift = np.asarray([ int(i) for i in phases.mean(axis=0) ]).astype(float)
    #     print("\tShifting the dipoles by the following quantum (on top of the global mean): ",args.shift," ... ", end="")
    #     shift += args.shift
    #     for i in range(3):
    #         phases[:,i] -= shift[i]
    #     print("done")

    if args.fix_only :
        shift = np.asarray([ int(i) for i in phases.mean(axis=0) ]).astype(float)
        print("\tThe dipoles (phases) will be shifted by the average value: ",shift)
        if args.shift is not None:        
            print("\tAdding the user-defined shift (phases): ",args.shift)
            shift += args.shift
        print("\tShifting the dipoles phases by ",shift, " ... ",end="")
        for i in range(3):
            phases[:,i] -= shift[i]
        print("done")

    print("\tConverting dipoles from lattice to cartesian coordinates ... ", end="")
    for n in range(N):
        cell = np.asarray(atoms[n].cell.array).T
        R = lattice2cart(cell)
        atoms[n].info["dipole"] = R @ ( phases[n,:] * lenght[n,:] )
    print("done")

    index = np.where(np.diff(old-phases,axis=0))[0]
    print("\n\tFound {:d} jumps".format(len(index)))
    if args.jumps is not None:
        print("\tSaving the indices of the jumps to file '{:s}' ... ".format(args.jumps), end="")
        np.savetxt(args.jumps,index)
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
