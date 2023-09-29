from ase.io import read, write
from miscellaneous.elia.functions import str2bool, suppress_output
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

args = parser.parse_args()
end = "" if not args.debug else ""

# Print the script's description
print("\n\t{:s}".format(description))

# Parse the command-line arguments
print("\n\tRead input arguments ... ")


# Read the MicroState data from the input file
instructions = {
    "cells": args.input,      # Use the input file for 'cells' data
    "positions": args.input,  # Use the input file for 'positions' data
    "types": args.input       # Use the input file for 'types' data
}

print("\tReading data from input file '{:s}' ... ".format(args.input), end=end)
with suppress_output(not args.debug):
    atoms = read(args.input,format=args.input_format)
if not args.debug:
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
