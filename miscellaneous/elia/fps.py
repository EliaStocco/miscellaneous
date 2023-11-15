from ase.io import read, write
import chemiscope
import numpy as np
import json

from rascal.representations import SphericalInvariants as SOAP
from skmatter.preprocessing import StandardFlexibleScaler
from skmatter.feature_selection import FPS
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import argparse
from miscellaneous.elia.functions import add_default

def main():

    description = "Process atomic structures and select a diverse subset using the Farthest Point Sampling (FPS) algorithm."

    message = "!Pay attention that the provided positions should be in angstrom!"

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser()
    argv = {"metavar" : "\b"}
    parser.add_argument("-i" , "--input"         , action="store", type=str , **argv, help="input file")
    parser.add_argument("-o" , "--output"        , action="store", type=str , **argv, help="output file with the selected structures ")
    parser.add_argument("-oi", "--output_indices", action="store", type=str , **argv, help="output file with indices of the selected structures (default: 'indices.txt')",default='indices.txt')
    parser.add_argument("-if", "--input_format"  , action="store", type=str , **argv, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-of", "--output_format" , action="store", type=str , **argv, help="output file format (default: 'None')", default=None)
    parser.add_argument("-n" , "--number"        , action="store", type=int , **argv, help="number of desired structure (default: '100')", default=None)
    parser.add_argument("-sh", "--soap_hyper"    , action="store", type=str , **argv, help="JSON file with the SOAP hyperparameters", default=None)

    # Print the script's description
    print("\n\t{:s}".format(description))

    # Parse the command-line arguments
    print("\n\tReading input arguments ... ",end="")
    args = parser.parse_args()
    print("done")

    print("\t{:s}".format(message))

    #
    print("\n\tReading positions from file '{:s}' ... ".format(args.input),end="")
    frames = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
    available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

    print('\tNumber of frames: ', len(frames))
    print('\tNumber of atoms/frame: ', len(frames[0]))
    print('\tAvailable structure properties: ', available_structure_properties)
    print('\tAvailable atom-level properties: ', available_atom_level_properties)

    if args.soap_hyper is not None :
        print("\n\tReading the SOAP hyperparameters from file '{:s}' ... ".format(args.soap_hyper),end="")
        with open(args.soap_hyper, 'r') as file:
            # Load the JSON data from the file
            user_soap_hyper = json.load(file)
        print("done")
    else:
        user_soap_hyper = None


    print("\n\tPreparing SOAP object ... ",end="")
    SOAP_HYPERS = {
        "interaction_cutoff": 3.5,
        "max_radial": 6,
        "max_angular": 6,
        "gaussian_sigma_constant": 0.4,
        "cutoff_smooth_width": 0.5,
        "gaussian_sigma_type": "Constant",
    }

    SOAP_HYPERS = add_default(user_soap_hyper,SOAP_HYPERS)

    #
    numbers = list(sorted(set([int(n) for frame in frames for n in frame.numbers])))

    # initialize SOAP
    soap = SOAP(
        global_species=numbers,
        expansion_by_species_method='user defined',
        **SOAP_HYPERS
    )
    print("done")

    X = None
    print("\tComputing SOAP features ... ")
    for i, frame in enumerate(tqdm(frames)):
        # normalize cell for librascal input
        if np.linalg.norm(frame.cell) < 1e-16:
            extend = 1.5 * (np.max(frame.positions.flatten()) - np.min(frame.positions.flatten()))
            frame.cell = [extend, extend, extend]
            frame.pbc = True
        frame.wrap(eps=1e-16)

        x = soap.transform(frame).get_features(soap).mean(axis=0) # here it takes mean over atoms in the frame
        if X is None:
            X = np.zeros((len(frames), x.shape[-1]))
        X[i] = x

    print(f"\tSOAP features shape: {X.shape}")

    file = 'full-featurization.npy'
    print("\tSaving features to file '{:s}' ... ".format(file),end="")
    np.save(file, X)
    print("done")


    #
    if args.number is None :
        n_FPS = min(100,len(frames)) # number of structures to select
    else :
        n_FPS = args.number
    struct_idx = FPS(n_to_select=n_FPS, progress_bar = True, initialize = 'random').fit(X.T).selected_idx_
    X_fps = X[struct_idx]

    print("\tFPS selected indices: {:d}".format(struct_idx.shape[0]))
    print(f"\tOriginal: {X.shape} ---> FPS: {X_fps.shape}")

    # Saving the fps selected structure
    if args.output is not None :
        print("\tSaving FPS selected structures to file '{:s}' ... ".format(args.output),end="")
        frames_fps = [frames[i] for i in struct_idx]
        write(args.output, frames_fps, format=args.output_format)
        print("done")
    else :
        print("\tOutput file (-o/--output) for the selected structures not specified: they will not be saved to file")

    # Saving the fps selected structure
    if args.output_indices :
        print("\tSaving indices of selected structures to file '{:s}' ... ".format(args.output_indices),end="")
        indices = np.asarray([ int(i) for i in struct_idx],dtype=int)
        np.savetxt(args.output_indices,indices,fmt='%d')
        print("done")
    else :
        print("\tOutput file (-oi/--output_indices) for the indeces of the selected structures not specified: they will not be saved to file")

    print("\n\tJob done :)\n")

    return

    # Visualizing using the principal componenets of the selected dataset and the original dataset using the chemiscope
    X = StandardFlexibleScaler(column_wise=False).fit_transform(X)
    T = PCA(n_components=2).fit_transform(X)

    #
    np.savetxt('PES_PCA.txt', np.concatenate([T, np.array([frame.info['energy'] for frame in frames]).reshape(-1, 1)], axis=1))

    #
    available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
    available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

    print("Available structure-level properties", available_structure_properties)
    print("Available atom-level properties", available_atom_level_properties)

    #


    properties = {
        "PCA": {
            # change the following line if your map is per-atom
            "target": "structure",
            "values": T,

            # change the following line to describe your map
            "description": "PCA of structure-averaged representation",
        },

        # this is an example of how to add structure-level properties
        "energy": {
            "target": "structure",
            "values": [frame.info['energy'] for frame in frames],

            # change the following line to correspond to the units of your property
            "units": "eV",
        },

        # this is an example of how to add atom-level properties
        "numbers": {
            "target": "atom",
            "values": np.concatenate([frame.arrays['numbers'] for frame in frames]),
        },
    }

    chemiscope.write_input(
        path=f"PES_PCA-chemiscope.json.gz",
        frames=frames,
        properties=properties,

        # # This is required to display properties with `target: "atom"`
        # # Without this, the chemiscope will show only structure-level properties
        # environments=chemiscope.all_atomic_environments(frames),
    )

    #

if __name__ == "__main__":
    main()
