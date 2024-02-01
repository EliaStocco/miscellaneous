#!/usr/bin/env python
from ase.io import read
import numpy as np
from ase import Atoms
from miscellaneous.elia.classes.dipole import dipoleLM
from miscellaneous.elia.classes.trajectory import trajectory as Trajectory
from miscellaneous.elia.input import size_type

#---------------------------------------#
# Description of the script's purpose
description = "Create a linar model for the dipole of a system given the Born Effective Charges of a reference configuration."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA  + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    information     = Fore.YELLOW   + Style.NORMAL + information             + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, help="file with the atomic configurations [a.u]")
    parser.add_argument("-n", "--index"    , **argv,type=int, help="index of the reference configuration",default=None)
    parser.add_argument("-r", "--reference", **argv,type=str, help="file with the reference configuration [a.u.,xyz]")
    parser.add_argument("-d", "--dipole"   , **argv,type=lambda a:size_type(a,float,3), help="dipole of the reference configuration [a.u.] (default: None --> specify -n,--index)",default=None)
    parser.add_argument("-z", "--bec"      , **argv,type=str, help="file with the BEC tensors of the reference configuration [txt] (default: None --> specify -n,--index)",default=None)
    parser.add_argument("-f", "--frame"    , **argv,type=str, help="frame [eckart,global] (default: global)", default="global")
    parser.add_argument("-o", "--output"   , **argv,type=str, help="output file with the dipole linear model (default: 'dipoleLM.pickle')", default="dipoleLM.pickle")
    return parser.parse_args()

#---------------------------------------#
def main():

    #------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = Trajectory(args.input)
    print("done")

    #------------------#
    # index
    ref = None
    bec = None
    dipole = None
    if args.index is not None:
        reference = trajectory[args.index]
        try: bec = np.asarray(reference.arrays["bec"]) 
        except: pass
        try: dipole = np.asarray(reference.info["dipole"])
        except: pass
        try: ref = Atoms(reference)#.get_positions()
        except: pass
        del reference

    if ref is None:
        ref = read(args.reference)#.get_positions()
    if bec is None:
        bec = np.loadtxt(args.bec)
    if dipole is None:
        dipole = np.loadtxt(args.dipole).reshape(3)

    #------------------#
    print("\n\tCreating the linear model for the dipole ... ", end="")
    model = dipoleLM(ref=ref,dipole=dipole,bec=bec)
    print("done")

    #------------------#
    print("\n\tSaving the model to file '{:s}' ... ".format(args.output), end="")
    model.to_pickle(args.output)
    print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()