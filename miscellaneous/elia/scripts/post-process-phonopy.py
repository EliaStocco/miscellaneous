#!/usr/bin/env python
from miscellaneous.elia.normal_modes import NormalModes
from miscellaneous.elia.formatting import matrix2str
from miscellaneous.elia.functions import convert
from miscellaneous.elia.functions import output_folder
from miscellaneous.elia.input import size_type
import argparse
import numpy as np
import yaml
import pandas as pd
from icecream import ic
import os
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
    parser.add_argument("-q",  "--qpoints",       type=str, **argv, 
                        help="qpoints file (default: 'qpoints.yaml')", default="qpoints.yaml")
    parser.add_argument("-i",  "--input",         type=str, **argv, 
                        help="general phonopy file (default: 'phonopy.yaml')", default="phonopy.yaml")
    parser.add_argument("-o",  "--output",        type=str, **argv, 
                        help="csv output file (default: 'phonon-modes.pickle')", default="phonon-modes.pickle")
    parser.add_argument("-of", "--output_folder", type=str, **argv, 
                        help="output folder for *.mode, *.eigvec and *.eigval files (default: None)", default=None)
    parser.add_argument("-m",  "--matrices",      type=lambda x: size_type(x,str), **argv, 
                        help="matrices/vectors to print (default: ['eigval','eigvec','mode'])", default=['eigval','eigvec','mode'])
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
    # read input file ('phonopy.yaml')
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
    # read qpoints file ('qpoints.yaml')
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
    print("\n\tReading phonon modes and related quantities ... ")
    index = [ tuple(a["q-position"]) for a in qpoints["phonon"] ]
    pm = pd.DataFrame(index=index,columns=["q","freq","modes"])
    factor = convert(1,"frequency","inversecm","atomic_unit")
    for n,phonon in enumerate(qpoints["phonon"]):
        q = tuple(phonon["q-position"])
        print("\t\tphonons {:d}: q point".format(n),q)
        N = len(phonon["band"])
        nm = NormalModes(N,N)
        # nm.set_dynmat(phonon["dynamical_matrix"],mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")
        nm.masses = mass

        pm.at[q,"q"]     = tuple(q)
        pm.at[q,"freq"]  = [ a["frequency"] for a in phonon["band"] ]

        nm.eigval = (factor*np.asarray(pm.at[q,"freq"]))**2
               
        pm.at[q,"modes"] = nm.build_supercell_displacement(size=size,q=q)

    #---------------------------------------#
    print("\n\tWriting phonon modes to file '{:s}' ... ".format(args.output), end="")
    pm.to_pickle(args.output)
    print("done")

    if args.output_folder is not None:
        print("\n\tWriting phonon modes to i-PI-like files in folder '{:s}':".format(args.output_folder))

        # create directory
        output_folder(args.output_folder)
        k = 0 
        for n,row in pm.iterrows():
            # ic(row)
            q = str(row["q"]).replace(" ","")
            print("\n\t\tphonons {:d}, with q point {:s}:".format(k,q))
            for name in args.matrices:
                matrix = getattr(row["modes"],name)
                if np.any(np.isnan(matrix)):
                    print("\t\t\t{:s}: {:s} matrix/vector containes np.nan values: it willnot be saved to file".format(warning,name))
                    continue
                file = os.path.normpath("{:s}/{:s}.{:s}.{:s}".format(args.output_folder,"phonopy",q,name))
                print("\t\t\t- {:>10s} saved to file '{:s}'".format(name,file))
                np.savetxt(file,matrix)
            k += 1
    
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
