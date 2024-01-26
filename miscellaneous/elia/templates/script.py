#!/usr/bin/env python
from ase.io import read, write
#---------------------------------------#
# Description of the script's purpose
description = "Template for a script."
error = "***Error***"
closure = "Job done :)"
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def prepare_args():
    # Define the command-line argument parser with a description
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"        ,   **argv,type=str     , help="input file")
    parser.add_argument("-o"  , "--output"       ,   **argv,type=str     , help="output file")
    parser.add_argument("-if" , "--input_format" ,   **argv,type=str     , help="input file format (default: 'None')" , default=None)
    parser.add_argument("-of" , "--output_format",   **argv,type=str     , help="output file format (default: 'None')", default=None)
    return parser.parse_args()

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

    #
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    atoms = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    # Write the data to the specified output file with the specified format
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    try:
        write(images=atoms,filename=args.output, format=args.output_format)
        print("done")
    except Exception as e:
        print("\n\tError: {:s}".format(e))

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))
    #

#---------------------------------------#
if __name__ == "__main__":
    main()
