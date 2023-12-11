#!/usr/bin/env python
from miscellaneous.elia.normal_modes import NormalModes
from miscellaneous.elia.formatting import matrix2str
from miscellaneous.elia.functions import convert
import argparse
import numpy as np
import yaml
import pandas as pd
#---------------------------------------#
# Description of the script's purpose
description = "Check results obtained using 'phonopy'."
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
    warning         = Fore.RED    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass
#---------------------------------------#
def prepare_args():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-q", "--qpoints", type=str, **argv, help="qpoints file (default: 'qpoints.yaml')", default="qpoints.yaml")
    parser.add_argument("-i", "--input",   type=str, **argv, help="general phonopy file (default: 'phonopy.yaml')", default="phonopy.yaml")
    parser.add_argument("-o", "--output",  type=str, **argv, help="csv output file (default: 'phonon-modes.csv')", default="phonon-modes.csv")
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
    print("\tReading data from input file '{:s}' ... ".format(args.input), end="")
    with open(args.input) as f:
        info = yaml.safe_load(f)
    print("done")

    print("\t{:<10s} : ".format("dim"),info["phonopy"]["configuration"]["dim"])
    print("\t{:<10s} : ".format("qpoints"),info["phonopy"]["configuration"]["qpoints"].split(" "))
    print("\t{:<10s} : ".format("masses"),np.asarray([ a["mass"] for a in info["unit_cell"]["points"] ]).round(2))   

    size = np.asarray([ int(a) for a in info["phonopy"]["configuration"]["dim"].split(" ") ])
    factor = convert(1,"mass","dalton","atomic_unit")
    mass = factor * np.asarray([ [a["mass"]]*3 for a in info["unit_cell"]["points"] ]).flatten()
    
    #---------------------------------------#
    # read qpoints
    print("\n\tReading qpoints from input file '{:s}' ... ".format(args.qpoints), end="")
    with open(args.qpoints) as f:
        qpoints = yaml.safe_load(f)
    print("done")
    
    print("\t{:<20s} : {:<10d}".format("# q points",qpoints["nqpoint"]))
    print("\t{:<20s} : {:<10d}".format("# atoms",qpoints["natom"]))
    print("\t{:<20s} :".format("reciprocal vectors"))
    tmp = np.asarray(qpoints["reciprocal_lattice"]).T
    line = matrix2str(tmp,digits=6,exp=False,width=12)
    print(line)

    #---------------------------------------#
    # phonon modes
    index = [ tuple(a["q-position"]) for a in qpoints["phonon"] ]
    pm = pd.DataFrame(index=index,columns=["q","freq","modes"])
    for n,phonon in enumerate(qpoints["phonon"]):
        q = tuple(phonon["q-position"])
        print("\tphonon {:d}, with q point".format(n),q)
        N = len(phonon["band"])
        nm = NormalModes(N,N)
        # nm.set_dynmat(phonon["dynamical_matrix"],mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")
        nm.masses = mass
        # nm.set_eigvals(phonon["band"],mode="phonopy")

        pm.at[q,"q"]     = tuple(q)
        pm.at[q,"freq"]  = [ a["frequency"] for a in phonon["band"] ]
        pm.at[q,"modes"] = nm.build_supercell_normal_modes(size=size).modes

        # print("\t\tfreq : ",np.asarray(pm.at[q,"freq"]).round(2))

    #---------------------------------------#
    print("\n\tWriting phonon modes to file '{:s}' ... ".format(args.output), end="")
    pm.to_csv(args.output,index=False)
    print("done")
    
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
