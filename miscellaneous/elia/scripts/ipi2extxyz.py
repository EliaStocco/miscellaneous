#!/usr/bin/env python
import argparse
import os
import numpy as np
from copy import copy
from ase.io import write, read
from ase import Atoms
from miscellaneous.elia.properties import properties as Properties
from miscellaneous.elia.functions import suppress_output, get_one_file_in_folder, str2bool
from miscellaneous.elia.input import size_type
from miscellaneous.elia.trajectory import trajectory

DEBUG=False
# example:
# python ipi2extxyz.py -p i-pi -f data -aa forces,data/i-pi.forces_0.xyz -ap dipole,potential -o test.extxyz

#---------------------------------------#

# Description of the script's purpose
description = "Convert the i-PI output files to an extxyz file with the specified properties and arrays.\n"
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"


#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA  + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    information     = Fore.YELLOW   + Style.NORMAL + information             + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

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

    parser.add_argument("-aa", "--additional_arrays",  type=lambda s: size_type(s,dtype=str), default=None, **argv,
                        help="additional arrays to be added to the output file (example: [velocities,forces], default: [])")
    
    parser.add_argument("-ap", "--additional_properties",  type=lambda s: size_type(s,dtype=str), default=["all"], **argv,
                        help="additional properties to be added to the output file (example: [potential,dipole], default: [all])")

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

        print("\tReading atomic structures from file '{:s}' ... ".format(args.positions_file), end="")
        with suppress_output(not DEBUG):
            atoms = trajectory(args.positions_file,format="i-pi")
        print("done")
    else :
        raise ValueError("to be implemented yet")
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
        print("\n\tYou specified the following properties to be added to the output file: ",properties)

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
                
        print("\tReading properties from file '{:s}' ... ".format(args.properties_file), end="")
        # instructions = {
        #     "properties" : args.properties_file, 
        # }
        with suppress_output(not DEBUG):
            allproperties = Properties.load(file=args.properties_file)
        print("done")

        print("\n\tSummary:")
        print("\t# atomic structures: {:d}".format(len(atoms)))
        print("\t       # properties: {:d}".format(len(allproperties)))

        if len(allproperties) != len(atoms):
            print("\n\t{:s}: n. of atomic structures and n. of properties differ.".format(warning))
            if len(allproperties) == len(atoms)+1 :
                print("\t{:s}\n\tMaybe you provided a 'replay' input file --> discarding the first properties raw.".format(information))
                allproperties = allproperties[1:]
            else:
                raise ValueError("I would expect n. of atomic structures to be (n. of properties + 1)")

        print("\n\tSummary of the read properties:\n")
        df = allproperties.summary()

        def line(): print("\t\t-----------------------------------------")
        
        line()
        print("\t\t|{:^15s}|{:^15s}|{:^7s}|".format("name","unit","shape"))
        line()
        for index, row in df.iterrows():
            print("\t\t|{:^15s}|{:^15s}|{:^7d}|".format(row["name"],row["unit"],row["shape"]))
        line()

        # all properties
        if "all" in properties:
            _properties = list(allproperties.properties.keys())
            for p in properties:
                p = p.replace(" ","")
                if p[0] == "~":
                    _properties.remove(p[1:])
            properties = copy(_properties)  
            del _properties
        
        print("\n\tStoring the following properties to file: ",properties)

        # 
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
        write(args.output, list(atoms), format="extxyz")
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
