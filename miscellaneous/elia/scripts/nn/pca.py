#!/usr/bin/env python
from ase.io import read, write
import chemiscope
import numpy as np
import json
import os

from rascal.representations import SphericalInvariants as SOAP
from skmatter.preprocessing import StandardFlexibleScaler
from skmatter.feature_selection import FPS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import argparse
from miscellaneous.elia.functions import add_default
from miscellaneous.elia.show import show_dict
from miscellaneous.elia.tools import convert
from miscellaneous.elia.input import str2bool


#---------------------------------------#
# Description of the script's purpose
description = "Process atomic structures and select a diverse subset using the Farthest Point Sampling (FPS) algorithm."
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
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i"  , "--input"             , type=str  , **argv, help="input file [au]")
    parser.add_argument("-o"  , "--output"            , type=str  , **argv, help="output file with the selected structures")
    parser.add_argument("-oi" , "--output_indices"    , type=str  , **argv, help="output file with indices of the selected structures (default: 'indices.txt')",default='indices.txt')
    parser.add_argument("-if" , "--input_format"      , type=str  , **argv, help="input file format (default: 'None')" , default=None)
    parser.add_argument("-of" , "--output_format"     , type=str  , **argv, help="output file format (default: 'None')", default=None)
    parser.add_argument("-ff" , "--features_file"     , type=str  , **argv, help="features output file [*.npy] (default: 'None')", default=None)
    parser.add_argument("-n"  , "--number"            , type=int  , **argv, help="number of desired structure (default: '100')", default=100)
    parser.add_argument("-s"  , "--sort"              , type=str2bool , **argv, help="whether to sort the indices (default: true)", default=True)
    parser.add_argument("-rc" , "--cutoff_radius"     , type=float, **argv, help="cutoff radius [au] (default: 6)", default=6)
    parser.add_argument("-sh" , "--soap_hyper"        , type=str  , **argv, help="JSON file with the SOAP hyperparameters", default=None)
    parser.add_argument("-pca", "--pca"               , type=str2bool  , **argv, help="whether to perform PCA or not (default: true)", default=True)
    parser.add_argument("-npca", "--number_pca"       , type=int  , **argv, help="number of components to be used in PCA (default: 2)", default=2)
    parser.add_argument("-fpca", "--feature_pca"      , type=str  , **argv, help="feature to be analysed using PCA (default: 'dipole')", default="dipole")
    parser.add_argument("-opca", "--output_pca"       , type=str  , **argv, help="output file for PCA (default: 'pca.txt')", default="pca.txt")
    parser.add_argument("-och" , "--output_chemiscope", type=str  , **argv, help="output file for chemiscope (default: 'chemiscope.json.gz')", default="chemiscope.json.gz")
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
    frames = read(args.input, index=':', format=args.input_format)  #eV
    print("done")

    print("\n\tConverting the atomic structures positions and cells from atomic unit to angstrom ... ",end="")
    factor = convert(1,"length","atomic_unit","angstrom")
    for n in range(len(frames)):
        frames[n].positions *= factor
        if np.any(frames[n].pbc):
            cell = factor * frames[n].get_cell()
            frames[n].set_cell(cell)
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

    print("\n\tUsing the following SOAP hyperparameters:")
    SOAP_HYPERS = {
        "interaction_cutoff": 3.5,
        "max_radial": 8,
        "max_angular": 6,
        "gaussian_sigma_constant": 0.4,
        "cutoff_smooth_width": 0.5,
        "gaussian_sigma_type": "Constant",
    }

    SOAP_HYPERS = add_default(user_soap_hyper,SOAP_HYPERS)
    # SOAP_HYPERS["interaction_cutoff"] = args.cutoff_radius
    show_dict(SOAP_HYPERS,string="\t\t")

    #
    print("\n\tPreparing SOAP object ... ",end="")
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

    print(f"\n\tSOAP features shape: {X.shape}")

    if args.features_file is not None:
        args.features_file = str(args.features_file)
        if not args.features_file.endswith("npy"):
            raise ValueError("'features_file' (-ff, --features_file) must have 'npy' extension.")
        print("\tSaving features to file '{:s}' ... ".format(args.features_file),end="")
        np.save(file, X)
        print("done")

    #
    struct_idx = FPS(n_to_select=args.number, progress_bar = True, initialize = 'random').fit(X.T).selected_idx_
    X_fps = X[struct_idx]

    print("\n\tFPS selected indices: {:d}".format(struct_idx.shape[0]))
    print(f"\tOriginal: {X.shape} ---> FPS: {X_fps.shape}")

    indices = np.asarray([ int(i) for i in struct_idx],dtype=int)

    if args.sort:
        print("\n\tSorting indices ... ",end="")
        indices = np.sort(indices)
        print("done")

    # Saving the fps selected structure
    if args.output_indices :
        print("\n\tSaving indices of selected structures to file '{:s}' ... ".format(args.output_indices),end="")
        np.savetxt(args.output_indices,indices,fmt='%d')
        print("done")

    # Saving the fps selected structure
    if args.output is not None :
        print("\n\tSaving FPS selected structures to file '{:s}' ... ".format(args.output),end="")
        frames_fps = [frames[i] for i in indices]
        write(args.output, frames_fps, format=args.output_format) # fmt)
        print("done")
    # else :
    #     print("\tOutput file (-o/--output) for the selected structures not specified: they will not be saved to file")


    # else :
    #     print("\tOutput file (-oi/--output_indices) for the indeces of the selected structures not specified: they will not be saved to file")

    # return

    if args.pca is not None:
        print("\n\tPerforming PCA:")

        X = X[struct_idx]
        frames_fps = [frames[i] for i in indices]
        frames = frames_fps.copy()

        #
        available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
        available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

        print("\t\tAvailable structure-level properties:", available_structure_properties)
        print("\t\t     Available atom-level properties:", available_atom_level_properties)

        # Visualizing using the principal componenets of the selected dataset and the original dataset using the chemiscope
        print("\t\tStandardyzing features ... ",end="")
        X = StandardFlexibleScaler(column_wise=False).fit_transform(X)
        print("done")

        print("\t\tApplying PCA with {:d} components ... ".format(args.number_pca),end="")
        T = TSNE(n_components=args.number_pca,learning_rate='auto', init='random', perplexity=3).fit_transform(X)
        print("done")

        #
        print("\t\tExtracting '{:s}' from the atomic structures ... ".format(args.feature_pca),end="")
        try :
            features = np.array([frame.info[args.feature_pca] for frame in frames])
        except:
            raise ValueError("encountered problems extracting '{:s}' info from the atomic structures".format(args.feature))
        print("done")

        tmp = np.concatenate([T, features], axis=1)
        print("\t\tSaving PCA results to file '{:s}'".format(args.output_pca),end="")
        np.savetxt(args.output_pca,tmp,fmt='%24.18e')
        print("done")

        properties = {
            "PCA": {
                # change the following line if your map is per-atom
                "target": "structure",
                "values": T,

                # change the following line to describe your map
                "description": "PCA of structure-averaged representation",
            },

            # this is an example of how to add structure-level properties
            args.feature_pca: {
                "target": "structure",
                "values": features[:,0].tolist(),

                # change the following line to correspond to the units of your property
                "units": "atomic_unit",
            },

            # # this is an example of how to add atom-level properties
            "numbers": {
                "target": "atom",
                "values": np.concatenate([frame.arrays['numbers'] for frame in frames]),
            },
        }

        print("\tSaving results for chemiscope to file '{:s}' ... ".format(args.output_chemiscope),end="")
        chemiscope.write_input(
            path=args.output_chemiscope,
            frames=frames,
            properties=properties,

            # # This is required to display properties with `target: "atom"`
            # # Without this, the chemiscope will show only structure-level properties
            # environments=chemiscope.all_atomic_environments(frames),
        )
        print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))
    #

#---------------------------------------#
if __name__ == "__main__":
    main()
