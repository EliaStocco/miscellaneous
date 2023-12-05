#!/usr/bin/env python
from ase.io import read, write
from ase.cell import Cell
# from miscellaneous.elia.functions import str2bool, suppress_output, convert
import argparse
import numpy as np
import contextlib
import sys
import os

#---------------------------------------#

# Description of the script's purpose
description = "Convert the format of a file using 'ASE'"
warning = "***Warning***"
closure = "Job done :)"
keywords = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN  + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.RED    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

# Attention:
# If the parser used in ASE automatically modify the unit of the cell and/or positions,
# then you should add this file format to the list at line 55 so that the user will be warned.

#---------------------------------------#

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

@contextlib.contextmanager
def suppress_output(suppress=True):
    if suppress:
        with open(os.devnull, "w") as fnull:
            sys.stdout.flush()  # Flush the current stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = sys.__stdout__  # Restore the original stdout
    else:
        yield

def convert(what, family=None, _from="atomic_unit", _to="atomic_unit"):
    if family is not None:
        factor = unit_to_internal(family, _from, 1)
        factor *= unit_to_user(family, _to, 1)
        return what * factor
    else :
        return what
#---------------------------------------#

try:
    from ipi.utils.units import unit_to_internal, unit_to_user
    conversion_possible = True
except:
    print("!Warning: this script is not able to import i-PI: it will not be posssible to convert from different units.")
    conversion_possible = False

#---------------------------------------#

def main():

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str     , help="input file")
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str     , help="output file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str     , help="input file format (default: 'None')" , default=None)
    parser.add_argument("-of" , "--output_format",   **argv,type=str     , help="output file format (default: 'None')", default=None)
    parser.add_argument("-d"  , "--debug"        ,   **argv,type=str2bool, help="debug (default: False)"              , default=False)
    parser.add_argument("-iu" , "--input_unit"   ,   **argv,type=str     , help="input positions unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-iuc", "--input_unit_cell", **argv,type=str, help="input cell unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-ou" , "--output_unit" ,    **argv,type=str     , help="output unit (default: atomic_unit)", default=None)
    parser.add_argument("-s"  , "--scaled"      ,    **argv,type=str2bool, help="whether to output the scaled positions (default: False)", default=False)
    parser.add_argument("-r"  , "--rotate" ,         **argv,type=str2bool     , help="whether to rotate the cell s.t. to be compatible with i-PI (default: False)", default=False)

    # Print the script's description
    print("\n\t{:s}".format(description))

    # Parse the command-line arguments
    # print("\n\tReading input arguments ... ",end="")
    args = parser.parse_args()
    end = "" if not args.debug else ""
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    print("\tReading data from input file '{:s}' ... ".format(args.input), end=end)
    with suppress_output(not args.debug):
        # Try to determine the format by checking each supported format
        if args.input_format is None:
            from ase.io.formats import filetype
            args.input_format = filetype(args.input, read=isinstance(args.input, str))

        atoms = read(args.input,format=args.input_format,index=":")
    if not args.debug:
        print("done")

    if args.input_format in ["espresso-in","espresso-out"]:
        args.input_unit = "angstrom"
        args.input_unit_cell = "angstrom"
        if args.output_unit is None:
            print("\n\t{:s}: the file format is '{:s}', then the position ".format(warning,args.input_format)+\
                "and cell are automatically convert to 'angstrom' by ASE.\n\t"+\
                    "Specify the output units (-ou,--output_unit) if you do not want the output to be in 'angstrom'.\n")
        if args.output_format is None or args.output_format == "espresso-in":
            print("\n\t{:s}: the file format is 'espresso-in'.\n\tThen, even though the positions have been converted to another unit, ".format(warning) + \
                    "you will find the keyword 'angstrom' in the output file."+\
                    "\n\t{:s}\n".format(keywords))

    pbc = np.any( [ np.all(atoms[n].get_pbc()) for n in range(len(atoms)) ] )

    print("\tThe atomic structure is {:s}periodic.".format("" if pbc else "not "))

    if args.output_unit is not None :
        if not conversion_possible:
            raise ValueError("It's not possible to convert the units because i-PI was not imported.")
        if args.input_unit is None :
            args.input_unit = "atomic_unit"
        extra = "" if not pbc else "(and lattice parameters) "
        
        factor_pos = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit)
        if pbc : 
            if args.input_unit_cell is None :
                print("\t{:s} The unit of the lattice parameters is not specified ('input_unit_cell'):".format(warning)+\
                      "\n\t\tit will be assumed to be equal to the positions unit")
                args.input_unit_cell = args.input_unit
            # print("\tConverting lattice parameters from '{:s}' to '{:s}'".format(args.input_unit_cell,args.output_unit))
            factor_cell = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit_cell)

        print("\tConverting positions {:s}from '{:s}' to '{:s}' ... ".format(extra,args.input_unit,args.output_unit),end=end)
        for n in range(len(atoms)):
            atoms[n].set_calculator(None)
            atoms[n].positions *= factor_pos
            if np.all(atoms[n].get_pbc()):
                atoms[n].cell *= factor_cell
        print("done")

    # atoms[0].positions - (atoms[0].cell.array @ atoms[0].get_scaled_positions().T ).T 
    if args.rotate:
        print("\tRotating the lattice vectors such that they will be in upper triangular form ... ",end=end)
        for n in range(len(atoms)):
            atom = atoms[n]
            # frac = atom.get_scaled_positions()
            cellpar = atom.cell.cellpar()
            cell = Cell.fromcellpar(cellpar).array

            atoms[n].set_cell(cell,scale_atoms=True)
        print("done")

    # scale
    if args.scaled:
        print("\tReplacing the cartesian positions with the fractional/scaled positions: ... ",end=end)        
        for n in range(len(atoms)):
            atoms[n].set_positions(atoms[n].get_scaled_positions())
        print("done")
        print("\n\t{:s}: in the output file the positions will be indicated as 'cartesian'.".format(warning) + \
              "\n\t{:s}".format(keywords))

    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end=end)
    try:
        write(images=atoms,filename=args.output, format=args.output_format)
        if not args.debug:
            print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
