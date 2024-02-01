#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import argparse
import numpy as np
from ase.io import read
from miscellaneous.elia.formatting import matrix2str
from miscellaneous.elia.tools import find_transformation
from miscellaneous.elia.input import str2bool
from miscellaneous.elia.tools import sort_atoms

#---------------------------------------#
description     = "Compute the difference between two atomic structures. "
warning         = "***Warning***"
closure         = "Job done :)"
keywords        = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
divisor         = "-"*100

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
    divisor         = Fore.CYAN   + Style.NORMAL + divisor         + Style.RESET_ALL
except:
    pass
    
#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a", "--structure_A",  type=str    ,**argv,help="atomic structure A [au]")
    parser.add_argument("-b", "--structure_B",  type=str     ,**argv,help="atomic structure B [au]")
    parser.add_argument("-s", "--sort"       ,  type=str2bool,**argv,help="whether to sort the second structure (dafault: true)", default=True)
    return parser.parse_args()

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

    #-------------------#
    print("\tReading atomic structure A from file '{:s}' ... ".format(args.structure_A), end="")
    A = read(args.structure_A)
    print("done")

    #-------------------#
    print("\tReading atomic structure B from file '{:s}' ... ".format(args.structure_B), end="")
    B = read(args.structure_B)
    print("done")

    #-------------------#
    # sort
    if args.sort:
        print("\n\tSorting the atoms of the second structure  ... ", end="")
        B, indices = sort_atoms(A, B)
        print("done")

    #-------------------#
    # cells
    _ , M = find_transformation(A,B)
    print("\n\tTrasformation matrix M(A->B):")
    line = matrix2str(M,digits=2,col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    #---------------------------------------#
    # positions
    print("\n\tPositions differences:")
    diff = A.positions - B.positions
    M = np.concatenate([diff,np.linalg.norm(diff,axis=1)[:, np.newaxis]], axis=1)
    #M = np.concatenate(A.positions - B.positions],
    line = matrix2str(M,digits=3,col_names=["x","y","z","norm"],cols_align="^",width=8,row_names=A.get_chemical_symbols())
    print(line)
    
    #-------------------#
    print("\n\t{:s}\n".format(closure))
    return

if __name__ == "__main__":
    main()