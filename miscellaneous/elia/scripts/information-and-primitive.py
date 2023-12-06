#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
# from ase.io import read
import argparse
import numpy as np
from miscellaneous.elia.functions import matrix2str
from ase.cell import Cell
from ase import Atoms
from ase.io import write
from gims.structure import Structure, read
from gims.structure_info import StructureInfo

#---------------------------------------#
description     = "Show general information of a given atomic structure and find its primitive cell structure using GIMS."
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
    warning         = Fore.RED    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
    divisor         = Fore.CYAN   + Style.NORMAL + divisor         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def find_trasformation(A:Atoms,B:Atoms):
    M = np.asarray(B.cell).T @ np.linalg.inv(np.asarray(A.cell).T)
    size = M.round(0).diagonal().astype(int)
    return size, M

#---------------------------------------#
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
#---------------------------------------#
def prepare_parser():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-i", "--input",  type=str,**argv,
        help="atomic structure input file"
    )
    parser.add_argument(
        "-t", "--threshold",  type=float,**argv,
        help="threshold for GIMS (default: 1e-3)", default=1e-3
    )
    parser.add_argument(
        "-r"  , "--rotate" , type=str2bool, **argv,
        help="whether to rotate the cell to the upper triangular form compatible with i-PI (default: True)", default=True
    )
    parser.add_argument(
        "-p"  , "--primitive" , type=str2bool, **argv,
        help="whether to compute the primitive structure (default: False)", default=False
    )
    parser.add_argument(
        "-o", "--output",  type=str,**argv,
        help="output file of the primitive structure (default: 'None')", default=None
    )
    parser.add_argument(
        "-of" , "--output_format",   type=str, **argv,
        help="output file format (default: 'None')", default=None
    )
    return parser.parse_args()

#---------------------------------------#
def main():

    args = prepare_parser()

    # Print the script's description
    print("\n\t{:s}".format(description))
    # print("done")
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))

    #---------------------------------------#
    print("\n\t{:s}".format(divisor))
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    if args.rotate:
        print("\tRotating the lattice vectors of the atomic structure such that they will be in upper triangular form ... ",end="")
        # frac = atom.get_scaled_positions()
        cellpar = atoms.cell.cellpar()
        cell = Cell.fromcellpar(cellpar).array
        if np.allclose(cell,atoms.cell):
            print("done")
            print("\tThe lattice vectors are already in upper triangular form.")
        else:
            atoms.set_cell(cell,scale_atoms=True)
            print("done")

    print("\n\tComputing general information of the atomic structure using GIMS ... ",end="")
    #  = StructureInfo(atoms,1e-3).get_info()
    structure = Structure(atoms)
    info = str(StructureInfo(structure,args.threshold))
    print("done")
    info = info.replace("System Info","\nOriginal structure information:")
    info = info.replace("\n","\n\t")
    print("\t"+info)

    print("\tCell:")
    line = matrix2str(structure.cell.array.T,col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    if args.primitive:
        #---------------------------------------#
        print("\n\t{:s}".format(divisor))
        print("\n\tComputing the primitive cell using GIMS ... ",end="")
        primive_structure = structure.get_primitive_cell(args.threshold)
        print("done")

        if args.rotate:
            print("\tRotating the lattice vectors of the primitive structure such that they will be in upper triangular form ... ",end="")
            # frac = atom.get_scaled_positions()
            cellpar = primive_structure.cell.cellpar()
            cell = Cell.fromcellpar(cellpar).array
            if np.allclose(cell,primive_structure.cell):
                print("done")
                print("\tThe lattice vectors are already in upper triangular form.")
            else:
                primive_structure.set_cell(cell,scale_atoms=True)
                print("done")    

        print("\n\tComputing general information of the primitive structure using GIMS ... ",end="")
        info = str(StructureInfo(primive_structure,args.threshold))
        print("done")    
        info = info.replace("System Info","\nPrimitive cell structure information:")
        info = info.replace("\n","\n\t")
        print("\t"+info)

        print("\tCell:")
        line = matrix2str(primive_structure.cell.array.T,col_names=["1","2","3"],cols_align="^",width=6)
        print(line)
        
        #---------------------------------------#
        # trasformation
        print("\n\t{:s}".format(divisor))
        size, M = find_trasformation(primive_structure,structure)
        print("\tTrasformation matrix from primitive to original cell:")
        line = matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

        #---------------------------------------#
        # Write the data to the specified output file with the specified format
        if args.output is not None:            
            print("\n\t{:s}".format(divisor))
            print("\n\tWriting primitive structure to file '{:s}' ... ".format(args.output), end="")
            try:
                write(images=primive_structure,filename=args.output, format=args.output_format)
                print("done")
            except Exception as e:
                print("\n\tError: {:s}".format(e))

    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()