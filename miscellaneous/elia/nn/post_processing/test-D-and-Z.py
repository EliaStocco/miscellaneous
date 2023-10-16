import argparse
import numpy as np
import os
import sys
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.nn.functions import get_model

#####################

description = "Compute dipole and BEC tensors\n"

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
        "-q","--positions", action="store", type=str,
        help="positions file (in a.u.)"
    )

    parser.add_argument(
        "-c","--cell", action="store", type=str, 
        help="cell file (in a.u.)", default=None
    )

    parser.add_argument(
        "-o","--output", action="store", type=str,
        help="prefix for the output files", default="test"
    )

    parser.add_argument(
        "-f","--format", action="store", type=str,
        help="format", default='%-.10f'
    )

    return parser.parse_args()

def main():

    #####################
    # Print the script's description
    print("\n\t{:s}".format(description))
    
    #####################
    # Parse the command-line arguments
    print("\tReading input arguments ... ",end="")
    args = get_args()
    print("done")

    #####################
    # load the model
    # print("\tLoading the model ... ",end="")
    model = get_model(args.instructions,args.parameters)
    # print("done")

    ######################
    # read the positions
    instructions = { "positions" : args.positions }
    if model.pbc :
        instructions["cells"] = args.cell if args.cell is not None else args.positions

    # print("\n\tReading the positions and cell ... ",end="")
    original_stdout = sys.stdout
    with open('/dev/null', 'w') as devnull:
        sys.stdout = devnull  # Redirect stdout to discard output
        data = MicroState(instructions=instructions)
    sys.stdout = original_stdout

    if not model.pbc :
        data.cells = [None]*len(data.positions)    

    ######################
    print("\tComputing predicted values ... ",end="")
    N = len(data.positions)
    D = np.full((N,3),np.nan)
    Z = np.full((N,len(data.positions[0]),3),np.nan)

    for n,(pos,cell) in enumerate(zip(data.positions,data.cells)):
        d,z,x = model.get_value_and_jac(pos=pos.reshape((-1,3)),cell=cell)
        D[n] = d.detach().numpy()#.flatten()
        Z[n] = z.detach().numpy()#.flatten()

    print("done")

    ######################
    file = os.path.normpath( "{:s}.dipole.txt".format(args.output))
    print("\tSaving dipole to file '{:s}' ... ".format(file),end="")
    with open(file,'w') as f:
        for n in range(N):
            np.savetxt(f,D[n],delimiter='\t', fmt=args.format)
    print("done")

    ######################
    file = os.path.normpath("{:s}.bec.txt".format(args.output))
    print("\tSaving BECs to file '{:s}' ... ".format(file),end="")
    with open(file,'w') as f:
        for n in range(N):
            np.savetxt(file,Z[n],delimiter='\t', fmt=args.format)
            f.write("\n")
    print("done")

    print("\n\tJob done :)\n")

#####################

if __name__ == "__main__":
    main()