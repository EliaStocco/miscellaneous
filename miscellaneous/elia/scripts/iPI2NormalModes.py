#!/usr/bin/env python
from miscellaneous.elia.normal_modes import NormalModes
# from miscellaneous.elia.formatting import matrix2str
# from miscellaneous.elia.functions import convert
# from miscellaneous.elia.functions import output_folder
# from miscellaneous.elia.input import size_type
# from miscellaneous.elia.functions import phonopy2atoms
import argparse
import numpy as np
# import yaml
import pandas as pd
# from icecream import ic
# import os
from ase.io import read
# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from i-PI."
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
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass
#---------------------------------------#
def prepare_args():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-r",  "--reference",       type=str, **argv, 
                        help="reference structure w.r.t. which the phonons are computed [a.u.] (default: 'start.xyz')", default="start.xyz")
    parser.add_argument("-f",  "--folder",         type=str, **argv, 
                        help="folder with the output files of the i-PI vibrational analysis (default: '.')", default=".")
    parser.add_argument("-o",  "--output",        type=str, **argv, 
                        help="output file (default: 'vibrational-modes.pickle')", default="vibrational-modes.pickle")
    return parser.parse_args()
#---------------------------------------#
def main():

    #---------------------------------------#
    # print the script's description
    print("\n\t{:s}".format(description))

    #---------------------------------------#
    # parse the user input
    args = prepare_args()
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #---------------------------------------#
    # read reference atomic structure
    print("\tReading reference atomic structure from input '{:s}' ... ".format(args.reference), end="")
    reference = read(args.reference)
    print("done")

    #---------------------------------------#
    # phonon modes
    print("\n\tReading vibrational modes from folder '{:s}' ... ".format(args.folder))
    gamma = (0,0,0)
    pm = pd.DataFrame(index=[gamma],columns=["q","freq","modes"])
    nm = NormalModes.load(args.folder)
    pm.at[gamma,"q"]     = gamma
    pm.at[gamma,"modes"] = nm 
    nm.reference = reference
    print("done")

    #---------------------------------------#
    print("\n\tWriting vibrational modes to file '{:s}' ... ".format(args.output), end="")
    pm.to_pickle(args.output)
    print("done")
    
    #---------------------------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
