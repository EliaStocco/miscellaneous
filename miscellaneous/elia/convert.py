import argparse
import os
from ase.io import write
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import str2bool, suppress_output

# Description of the script's purpose
description = "Convert an i-PI MD trajectory to an ASE-compatible file with a specified format."

# Define the command-line argument parser with a description
parser = argparse.ArgumentParser(description=description)

# Define command-line arguments

# Input file containing the MD trajectory (default: 'i-pi.positions_0.xyz')
parser.add_argument("-i", "--input", action="store", type=str, default='i-pi.positions_0.xyz',
                    help="input file containing the MD trajectory (default: 'i-pi.positions_0.xyz')")

# Output file (default: 'output.xsf')
parser.add_argument("-o", "--output", action="store", type=str, default='output.xsf',
                    help="output file (default: 'output.xsf')")

# Output file format (default: xsf)
parser.add_argument("-f", "--format", action="store", type=str, default='xsf',
                    help="output file format (default: xsf)")

# Debug mode (default: False)
parser.add_argument("-d", "--debug", action="store", type=str2bool, default=False,
                    help="debug mode (default: False)")

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

print("\tReading data from input file '{:s}' using the 'MicroState' class ... ".format(args.input), end=end)
with suppress_output(not args.debug):
    data = MicroState(instructions=instructions)
if not args.debug:
    print("done")

print("\tConverting data from 'MicroState' to 'ase.Atoms' ... ", end=end)
with suppress_output(not args.debug):
    data = data.to_ase()
if not args.debug:
    print("done")

# Write the data to the specified output file with the specified format
print("\tWriting output to file '{:s}' with format '{:s}' ... ".format(args.output, args.format), end=end)
try:
    write(args.output, data, format=args.format)
    if not args.debug:
        print("done")
except Exception as e:
    print(f"\n\tError: {e}")

# Script completion message
print("\n\tJob done :)\n")
