#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
from miscellaneous.elia.formatting import matrix2str
from ase import Atoms
from ase.io import read
import argparse
import numpy as np

#---------------------------------------#
description     = "Compute the trasformation matrix M(A->B) between the lattice vector of the atomic configurations A and B."
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
def find_trasformation(A:Atoms,B:Atoms):
    M = np.asarray(B.cell).T @ np.linalg.inv(np.asarray(A.cell).T)
    size = M.round(0).diagonal().astype(int)
    return size, M

#---------------------------------------#
def find_A2B(file_A,file_B):
    # structure A
    print("\tReading structure A from input file '{:s}' ... ".format(file_A), end="")
    A = read(file_A)
    print("done")

    print("\tCell A:")
    cell = np.asarray(A.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # structure B
    print("\tReading structure B from input file '{:s}' ... ".format(file_B), end="")
    B = read(file_B)
    print("done")

    print("\tCell B:")
    cell = np.asarray(B.cell).T
    line = matrix2str(cell.round(4),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    # trasformation
    size, M = find_trasformation(A,B)
    print("\tTrasformation matrix M(A->B):")
    line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    det = np.linalg.det(M)
    print("\tdet(M): {:6.4f}".format(det))

    return

#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-a", "--structure_A",  type=str,**argv,
        help="atomic structure A [cell]"
    )
    parser.add_argument(
        "-b", "--structure_B",  type=str,**argv,
        help="atomic structure B [supercell]"
    )
    options = parser.parse_args()
    return options

#---------------------------------------#
def main():

    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    find_A2B(args.structure_A,args.structure_B)
    
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()