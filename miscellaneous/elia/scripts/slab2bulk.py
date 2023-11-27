import argparse
from ase import io
from ase.build import make_supercell

description="Generate bulk structure from a slab file using ASE."

def generate_bulk_structure(input_file, output_file, input_format, supercell_size):
    # Load the slab structure from a file with the specified format
    slab = io.read(input_file, format=input_format)

    # Generate the bulk structure by replicating the slab in three dimensions
    bulk = make_supercell(slab, supercell_size)

    # Save the bulk structure to the specified output file
    io.write(output_file, bulk)

def main():

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-i","input", help="input file with the slab structure")
    parser.add_argument("-o","output", help="output file with the bulk structure")
    parser.add_argument("-if","--input_format", default=None,help="input file format (default: None)")
    parser.add_argument("-s","--supercell", nargs=3, type=int, default=[1,1,1],help="Supercell size in each dimension (default: 1 1 1)")

    args = parser.parse_args()

    print("\n\t{:s}".format(description))

    generate_bulk_structure(args.input, args.output, args.input_format, args.supercell)

    print("\n\tJob done :)\n")


if __name__ == "__main__":
    main()

