import argparse
import torch
from miscellaneous.elia.nn.functions import get_model

#####################

description = "Create a standalone model from a torch.nn.Module \n"

def get_args():
    """Prepare parser of user input arguments."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-i","--instructions", action="store", type=str,
        help="model input file", default="instructions.json"
    )

    parser.add_argument(
        "-p","--parameters", action="store", type=str,
        help="(torch) parameters file", default="parameters.pth",
    )

    parser.add_argument(
        "-o","--output", action="store", type=str,
        help="prefix for the output files", default="test"
    )

    return parser.parse_args()

def main():

    # Print the script's description
    print("\n\t{:s}".format(description))
    
    # Parse the command-line arguments
    print("\tReading input arguments ... ",end="")
    args = get_args()
    print("done")

    # load the model
    model = get_model(args.instructions,args.parameters)

    # torch.jit.script
    print("\tCreating a standalone model by using 'torch.jit.script' ... ",end="")
    scripted_model = torch.jit.script(model)
    print("done")

    # save to file
    print("\tSaving the standalone model to file '{:s}' ... ".format(args.output),end="")
    torch.jit.save(scripted_model, args.output)
    print("done")
    
    print("\n\tJob done :)\n")

#####################

if __name__ == "__main__":
    main()