import argparse
import numpy as np
import os
import pandas as pd
from ipi.utils.units import unit_to_internal, unit_to_user

def convert(what,family,_from,_to):
    factor  = unit_to_internal(family,_from,1)
    factor *= unit_to_user(family,_to,1)
    return what * factor

# Description of the script's purpose
description = "Create a summary from the vibrational analysis results.\
\n!Pay attention: this script needs i-PI to be installed!"

# Define the command-line argument parser with a description
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-i" , "--input"  , action="store", type=str, help="eigvals file"  , default="i-pi.phonons.eigval")
parser.add_argument("-o" , "--output" , action="store", type=str, help="output file (csv)"                   , default=None)
args = parser.parse_args()

# Print the script's description
print("\n{:s}\n".format(description))

if not os.path.exists(args.input):
    raise ValueError("file '{:s}' does not exists".format(args.input))

print("Reading eigvals from file '{:s}' ... ".format(args.input), end="")
eigvals = np.loadtxt(args.input)
print("done")

print("Computing vibrational analysis summary ... ",end="")
df = pd.DataFrame()
df["eigvals [a.u.]"] = eigvals
df["w [a.u.]"]  = [ np.sqrt(i) if i > 0. else None for i in eigvals ]
df["w [THz]"]   = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="thz")
df["w [cm^-1]"] = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="inversecm")
df["T [a.u.]"]  = 2*np.pi / df["w [a.u.]"]
df["T [ps]"]    = convert(df["T [a.u.]"],"time",_from="atomic_unit",_to="picosecond")
print("done")


print("\nPreview:")
print(df.to_string(index=False,justify="right"))

if args.output is not None :
    print("\nWriting dataframe to file '{:s}' ... ".format(args.output),end="")
    df.to_csv(args.output,index=False)
    print("done")

# Script completion message
print("\nJob done :)\n")
