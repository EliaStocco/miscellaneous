#!/usr/bin/env python
import argparse
import os
import numpy as np
from copy import copy
from ase.io import write, read
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import suppress_output, get_one_file_in_folder, str2bool

# ToDo:
# remove dependece on 'MicroState'

description = "Convert the i-PI output files to an extxyz file with the specified properties and arrays.\n"
DEBUG=False
# example:
# python ipi2extxyz.py -p i-pi -f data -aa forces,data/i-pi.forces_0.xyz -ap dipole,potential -o test.extxyz

def arrays_type(s,dtype):
    s = s.split("[")[1].split("]")[0].split(",")
    return np.asarray([ dtype(k) for k in s ])

def prepare_args():

    parser = argparse.ArgumentParser(description=description)

    argv = {"metavar" : "\b",}

    parser.add_argument("-p", "--prefix", type=str, default='i-pi', **argv,
                        help="prefix of the i-PI output files (default: 'i-pi')")
    
    parser.add_argument("-f", "--folder", type=str, default='.', **argv,
                        help="folder (default: '.')")
    
    parser.add_argument("-qf", "--positions_file",  type=str, default=None, **argv,
                        help="input file containing the MD trajectory positions and cells (default: '[prefix].positions_0.xyz')")
    
    parser.add_argument("-pbc", "--pbc",  type=str2bool, default=True, **argv,
                        help="whether the system is periodic (default: True)")

    parser.add_argument("-pf", "--properties_file",  type=str, default=None, **argv,
                        help="input file containing the MD trajectory properties (default: '[prefix].properties.out')")

    parser.add_argument("-if", "--format",  type=str, default='i-pi', **argv,
                        help="input file format (default: 'i-pi')")

    parser.add_argument("-aa", "--additional_arrays",  type=lambda s: arrays_type(s,str), default=None, **argv,
                        help="additional arrays to be added to the output file")
    
    parser.add_argument("-ap", "--additional_properties",  type=lambda s: arrays_type(s,str), default=None, **argv,
                        help="additional properties to be added to the output file")

    parser.add_argument("-o", "--output",  type=str, default='output.extxyz', **argv,
                        help="output file in extxyz format (default: 'output.extxyz')")

    return parser.parse_args()

def main():

    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\tInput arguments:")
    for k in args.__dict__.keys():
        print("\t{:>25s}:".format(k),getattr(args,k))
    print()

    ###
    # atomic structures
    if args.format == "i-pi" :

        if args.positions_file is None:
            if args.prefix is None or args.prefix == "":
                raise ValueError("Please provide a prefix for the i-PI output files (--prefix) or a file with the atomic structures (--positions).")
            else :
                try :
                    args.positions_file = get_one_file_in_folder(folder=args.folder,ext="xyz",pattern="positions")
                except:
                    raise ValueError("Problem deducing the atomic structures file from the i-PI prefix.\n\
                                    Please check that the folder (-f,--folder) and the prefix (-i,--prefix) are correct.\n\
                                    Otherwise we can also directly specify the atomic structures file (-q,--positions).")
        elif not os.path.exists(args.positions_file):
            raise ValueError("File '{:s}' does not exist.".format(args.positions_file))

        # Read the MicroState data from the input file
        instructions = {
            # "cells"     : args.positions_file,  # Use the input file for 'cells' data
            "positions" : args.positions_file,  # Use the input file for 'positions' data
            # "types"     : args.positions_file   # Use the input file for 'types' data
        }

        print("\tReading atomic structures from file '{:s}' using the 'MicroState' class ... ".format(args.positions_file), end="")
        with suppress_output(not DEBUG):
            data = MicroState(instructions=instructions)
            atoms = data.to_ase(pbc=args.pbc)
            del data
        print("done")
    else :
        print("\tReading atomic structures from file '{:s}' using the 'ase.io.read' ... ".format(args.positions_file), end="")
        atoms = read(args.positions_file,format=args.format,index=":")
        if not args.pbc:
            atoms.set_pbc([False, False, False])
            atoms.set_cell()
        print("done")

    if args.additional_arrays is not None:
        arrays = dict()
        for k in args.additional_arrays:
            try :
                file = get_one_file_in_folder(folder=args.folder,ext="xyz",pattern=k)
            except:
                raise ValueError("No file provided or found for array '{:s}'".format(k))
            print("\tReading additional array '{:s}' from file '{:s}' using the 'ase.io.read' ... ".format(k,file), end="")
            tmp = read(file,index=":")
            arrays[k] = np.zeros((len(tmp)),dtype=object)
            for n in range(len(tmp)):
                arrays[k][n] = tmp[n].positions
            print("done")
        
        # atoms.arrays = dict()
        for k in arrays.keys():
            for n in range(len(arrays[k])):
                atoms[n].arrays[k] = arrays[k][n]

    if args.additional_properties is not None:
        properties = list(args.additional_properties)
        print("\tYou specified the following properties to be added to the output file: ",properties)

        ###
        # properties
        if args.properties_file is None:
            if args.prefix is None or args.prefix == "":
                raise ValueError("Please provide a prefix for the i-PI output files (--prefix) or the i-PI file with the properties (--properties).")
            else :
                try :
                    args.properties_file = get_one_file_in_folder(folder=args.folder,ext="out",pattern="properties")
                except:
                    raise ValueError("Problem deducing the properties file from the i-PI prefix.\n\
                                    Please check that the folder (-d,--folder) and the prefix (-i,--prefix) are correct.\n\
                                    Otherwise we can also directly specify the properties file (-p,--properties).")
        elif not os.path.exists(args.properties_file):
            raise ValueError("File '{:s}' does not exist.".format(args.properties_file))
                
        print("\tReading properties from file '{:s}' using the using the 'MicroState' class ... ".format(args.properties_file), end="")
        instructions = {
            "properties" : args.properties_file, 
        }
        with suppress_output(not DEBUG):
            allproperties = MicroState(instructions=instructions)
        print("done")

        print("\tSummary of the read properties:\n")
        df = allproperties.show_properties()

        def line(): print("\t\t-----------------------------------------")
        
        line()
        print("\t\t|{:^15s}|{:^15s}|{:^7s}|".format("name","unit","shape"))
        line()
        for index, row in df.iterrows():
            print("\t\t|{:^15s}|{:^15s}|{:^7d}|".format(row["name"],row["unit"],row["shape"]))
        line()

        tmp = dict()
        for k in properties:
            tmp[k] = allproperties.properties[k]
        properties = copy(tmp)
        del allproperties
        del tmp
    
        for k in properties.keys(): 
            for n in range(len(properties[k])):
                atoms[n].info[k] = properties[k][n]

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
