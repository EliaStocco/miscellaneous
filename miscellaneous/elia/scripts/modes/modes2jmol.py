#!/usr/bin/env python
from miscellaneous.elia.classes.normal_modes import NormalModes
from miscellaneous.elia.formatting import matrix2str
from miscellaneous.elia.tools import convert
import argparse
from ase.io import read
import numpy as np
import yaml
import pandas as pd
#---------------------------------------#
""" 
13.12.2023 Elia Stocco
    Some changes to the previous script:
    - using 'argparse'
    - verbose output to screen
    - defined a function callable from other scripts

04.05.2020 Karen Fidanyan
    This script takes the relaxed geometry in XYZ format
    and the .mode, .eigval files produced by i-PI,
    and builds a .xyz_jmol file to visualize vibrations.

"""
#---------------------------------------#
# Description of the script's purpose
description = "Create a JMOL file with the normal modes from the '*.mode' output file of a i-PI vibrational analysis."
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
    parser.add_argument("-i",  "--input",        type=str, **argv, help="atomic structure file [angstrom,xyz]")
    parser.add_argument("-m",  "--modes",        type=str, **argv, help="file with vibrational modes displacements [a.u.] (default: 'i-pi.phonons.mode')", default="i-pi.phonons.mode")
    parser.add_argument("-w",  "--eigenvalues",  type=str, **argv, help="file with vibrational modes eigenvalues [a.u.] (default: None)", default=None)
    parser.add_argument("-o",  "--output",       type=str, **argv, help="JMOL output file (default: 'vibmodes.jmol')", default="vibmodes.jmol")
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
    # read input file
    print("\tReading atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    #---------------------------------------#
    # read vibrational modes
    print("\tReading vibrational modes displacements from file '{:s}' ... ".format(args.modes), end="")
    modes = np.loadtxt(args.modes)
    print("done")

    if atoms.positions.flatten().shape[0] != modes.shape[0]:
        raise ValueError("positions and modes shapes do not match.")

    #---------------------------------------#
    # read eigenvalues
    if args.eigenvalues is not None:
        print("\tReading vibrational modes eigenvalues from file '{:s}' ... ".format(args.eigenvalues), end="")
        eigvals = np.loadtxt(args.eigenvalues)
        print("done")
    else:
        print("\tNo vibrational modes eigenvalues provided: setting them to zero ... ", end="")
        eigvals = np.zeros(modes.shape[0])
        print("done")

    if eigvals.shape[0] != modes.shape[0]:
        raise ValueError("eigvals and modes shapes do not match.")

    #---------------------------------------#
    # frequencies
    print("\tComputing frequencies ... ", end="")
    freqs = np.sqrt(eigvals) 
    freqs = convert(freqs,"frequency","atomic_unit","inversecm")
    print("done")

    #---------------------------------------#
    # write JMOL file
    print("\tWriting vibrational modes to file '{:s}' ... ".format(args.output), end="")
    np.set_printoptions(formatter={'float': '{: .8f}'.format})
    with open(args.output, 'w') as fdout:
        for b, vec in enumerate(modes.T):
            disp = vec.reshape(-1, 3)
            fdout.write("%i\n# %f cm^-1, branch # %i\n"
                        % (len(atoms), freqs[b], b))
            for i, atom in enumerate(atoms.positions):
                fdout.write("%s  " % atoms[i].symbol
                            + ' '.join(map("{:10.8g}".format, atom)) + "  "
                            + ' '.join(map("{:12.8g}".format, disp[i])) + "\n")
    print("done")

    # Script final message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
