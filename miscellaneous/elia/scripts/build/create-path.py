#!/usr/bin/env python
import numpy as np
from ase.io import read, write
from miscellaneous.elia.tools import segment

#---------------------------------------#
description     = "Create a path bridging two atomic structures (useful for NEB calculations)."
warning         = "***Warning***"
closure         = "Job done :)"
keywords        = "It's up to you to modify the required keywords."
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
def prepare_parser():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a", "--structure_A",  type=str,**argv,help="atomic structure A")
    parser.add_argument("-b", "--structure_B",  type=str,**argv,help="atomic structure B")
    parser.add_argument("-n", "--number"     ,  type=int,**argv,help="number of inner structures")
    parser.add_argument("-o", "--output"     ,  type=str,**argv,help="output file with the path", default="path.xyz")
    options = parser.parse_args()
    return options

#---------------------------------------#
def main():

    #-------------------#
    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #-------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A,index=0)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B,index=0)
    print("done")

    #-------------------#
    if not np.allclose(A.get_cell(),B.get_cell()):
        raise ValueError("The two structures do not have the same cell.")

    #-------------------#
    print("\n\tComputing the path positions '{:s}' ... ", end="")
    pathpos = segment(A.positions,B.positions,N=args.number)
    print("done")

    #-------------------#
    N = pathpos.shape[0]
    print("\tn. of structures in the path: '{:d}'".format(N))

    #-------------------#
    print("\tCreating the path '{:s}' ... ", end="")
    path = [None]*N
    for n in range(N):
        path[n] = A.copy()
        path[n].set_positions(pathpos[n])
    print("done")

    #-------------------#
    print("\n\tSaving the path to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=path,filename=args.output)
    except Exception as e:
        print("\n\tError: {:s}".format(e))
    print("done")

    #-------------------#
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()
