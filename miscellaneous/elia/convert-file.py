from ase.io import read, write
from miscellaneous.elia.functions import str2bool, suppress_output, convert
# from miscellaneous.elia.classes import MicroState
import argparse

# Description of the script's purpose
description = "Convert the format of a file using 'ASE'"

# Define the command-line argument parser with a description
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-i" , "--input"        , action="store", type=str     , help="input file")
parser.add_argument("-o" , "--output"       , action="store", type=str     , help="output file")
parser.add_argument("-if", "--input_format" , action="store", type=str     , help="input file format (default: 'None')" , default=None)
parser.add_argument("-of", "--output_format", action="store", type=str     , help="output file format (default: 'None')", default=None)
parser.add_argument("-d" , "--debug"        , action="store", type=str2bool, help="debug (default: False)"              , default=False)
parser.add_argument("-iu" , "--input_unit"  , action="store", type=str     , help="unit of the input file positions (default: atomic_unit)"  , default=None)
parser.add_argument("-ou" , "--output_unit" , action="store", type=str     , help="unit of the output file positions (default: atomic_unit)", default=None)


# Print the script's description
print("\n\t{:s}".format(description))

# Parse the command-line arguments
print("\n\tReading input arguments ... ",end="")
args = parser.parse_args()
end = "" if not args.debug else ""
print("done")


# Read the MicroState data from the input file
instructions = {
    "cells": args.input,      # Use the input file for 'cells' data
    "positions": args.input,  # Use the input file for 'positions' data
    "types": args.input       # Use the input file for 'types' data
}

print("\tReading data from input file '{:s}' ... ".format(args.input), end=end)
with suppress_output(not args.debug):
    atoms = read(args.input,format=args.input_format,index=":")
if not args.debug:
    print("done")

if args.output_unit is not None :
    if args.input_unit is None :
        args.input_unit = "atomic_unit"
    print("\tConverting positions from '{:s}' to {:s} ... ".format(args.input_unit,args.output_unit), end=end)
    factor = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit)
    if type(atoms) == list :
        for n in range(len(atoms)):
            atoms[n].positions *= factor
    else :
        atoms.positions *= factor
    print("done")

# Write the data to the specified output file with the specified format
print("\tWriting data to file '{:s}' ... ".format(args.output), end=end)
try:
    write(images=atoms,filename=args.output, format=args.output_format)
    if not args.debug:
        print("done")
except Exception as e:
    print("\n\tError: {:s}".format(e))

# Script completion message
print("\n\tJob done :)\n")
