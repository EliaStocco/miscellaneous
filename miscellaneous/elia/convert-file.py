from ase.io import read, write
from ase.cell import Cell
from miscellaneous.elia.functions import str2bool, suppress_output, convert
import argparse
import numpy as np

# Description of the script's purpose
description = "Convert the format of a file using 'ASE'"


def main():

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-i"  , "--input"        ,   type=str     , help="input file")
    parser.add_argument("-o"  , "--output"       ,   type=str     , help="output file")
    parser.add_argument("-if" , "--input_format" ,   type=str     , help="input file format (default: 'None')" , default=None)
    parser.add_argument("-of" , "--output_format",   type=str     , help="output file format (default: 'None')", default=None)
    parser.add_argument("-d"  , "--debug"        ,   type=str2bool, help="debug (default: False)"              , default=False)
    parser.add_argument("-iu" , "--input_unit"   ,   type=str     , help="input positions unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-iuc", "--input_unit_cell", type=str, help="input cell unit (default: atomic_unit)"  , default=None)
    parser.add_argument("-ou" , "--output_unit" ,    type=str     , help="output unit (default: atomic_unit)", default=None)
    parser.add_argument("-r"  , "--rotate" ,         type=str2bool     , help="whether to rotate the cell s.t. to be compatible with i-PI (default: False)", default=False)

    # Print the script's description
    print("\n\t{:s}".format(description))

    # Parse the command-line arguments
    print("\n\tReading input arguments ... ",end="")
    args = parser.parse_args()
    end = "" if not args.debug else ""
    print("done\n")

    print("\tReading data from input file '{:s}' ... ".format(args.input), end=end)
    with suppress_output(not args.debug):
        atoms = read(args.input,format=args.input_format,index=":")
    if not args.debug:
        print("done\n")

    pbc = np.any( [ np.all(atoms[n].get_pbc()) for n in range(len(atoms)) ] )

    print("\tThe read atomic structions is {:s}periodic".format("" if pbc else "not "))

    if args.output_unit is not None :
        if args.input_unit is None :
            args.input_unit = "atomic_unit"
        extra = "" if not pbc else "(and lattice parameters) "
        print("\tConverting positions from '{:s}' to '{:s}'".format(extra,args.input_unit,args.output_unit))
        factor_pos = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit)
        if pbc : 
            if args.input_unit_cell is None :
                print("\t***Warning*** The unit of the lattice parameters is not specified ('input_unit_cell'):\n\t\tit will be assumed to be equal to the positions unit")
                args.input_unit_cell = args.input_unit
            print("\tConverting lattice parameters from '{:s}' to '{:s}'".format(args.input_unit_cell,args.output_unit))
            factor_cell = convert(what=1,family="length",_to=args.output_unit,_from=args.input_unit_cell)

        for n in range(len(atoms)):
            atoms[n].set_calculator(None)
            atoms[n].positions *= factor_pos
            if np.all(atoms[n].get_pbc()):
                atoms[n].cell *= factor_cell

    # atoms[0].positions - (atoms[0].cell.array @ atoms[0].get_scaled_positions().T ).T 
    if args.rotate:
        print("\tRotating the lattice vectors such that they will be in upper triangular form ... ",end=end)
        for n in range(len(atoms)):
            atom = atoms[n]
            # frac = atom.get_scaled_positions()
            cellpar = atom.cell.cellpar()
            cell = Cell.fromcellpar(cellpar).array

            atoms[n].set_cell(cell,scale_atoms=True)
        print("done")

    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end=end)
    try:
        write(images=atoms,filename=args.output, format=args.output_format)
        if not args.debug:
            print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

    # Script completion message
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()

