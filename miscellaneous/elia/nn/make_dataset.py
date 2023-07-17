import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list
from miscellaneous.elia.classes import MicroState
import numpy as np

def get_type_onehot_encoding(species):
    type_encoding = {}
    for n,s in enumerate(species):
        type_encoding[s] = n
    type_onehot = torch.eye(len(type_encoding))
    return type_onehot, type_encoding

# def make_dataset(data:MicroState,\
#                  output:np.ndarray,\
#                  radial_cutoff:float):
    
#     species = data.all_types()
#     type_onehot, type_encoding = get_type_onehot_encoding(species)    

#     systems = data.to_ase()
    
#     # better to do this
#     output = torch.tensor(output)

#     #print("output:",output)
#     #print("output value shape:",output[0].shape)

#     dataset = [None] * len(systems)
#     n = 0 
#     bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
#     for crystal, out in tqdm(zip(systems, output),total=len(systems), bar_format=bar_format):
#         # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
#         # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
#         edge_src, edge_dst, edge_shift = \
#             neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)

#         data = Data(
#             pos=torch.tensor(crystal.get_positions()),
#             lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
#             x=type_onehot[[type_encoding[atom] for atom in crystal.get_chemical_symbols()]],
#             symbols = crystal.get_chemical_symbols(),
#             edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
#             # edge_src=torch.tensor(edge_src,dtype=int),
#             # edge_dst=torch.tensor(edge_dst,dtype=int),
#             edge_shift=torch.tensor(edge_shift),
#             yreal=out  # polarization (??assumed to be normalized "per atom" ??)
#         )

#         dataset[n] = data
#         n += 1
#     return dataset#,dtype=torch_geometric.data.Data)

# def make_datapoint(crystal,radial_cutoff):

#     species = np.unique(crystal.get_chemical_symbols())
#     type_onehot, type_encoding = get_type_onehot_encoding(species)

#     edge_src, edge_dst, edge_shift = \
#         neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
#     return Data(
#             pos=torch.tensor(crystal.get_positions()),
#             lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
#             x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
#             edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
#             edge_shift=torch.tensor(edge_shift),
#         )