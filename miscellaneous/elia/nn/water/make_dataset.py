import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list, primitive_neighbor_list
from miscellaneous.elia.classes import MicroState
from . import get_type_onehot_encoding
from ase import Atoms
import numpy as np


#----------------------------------------------------------------#

def preprocess(lattice, positions, symbols, radial_cutoff, default_dtype):
        
    species = np.unique(symbols)
    type_onehot, type_encoding = get_type_onehot_encoding(species)
    x=type_onehot[[type_encoding[atom] for atom in symbols]]

    pos=positions.reshape((-1,3))
    batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
    
    crystal = Atoms(cell=lattice,positions=pos,symbols=symbols)
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)

    # I need these lines here before calling 'einsum'
    edge_shift = torch.tensor(edge_shift,dtype=default_dtype)
    lattice = torch.tensor(lattice,dtype=default_dtype).unsqueeze(0)# We add a dimension for batching

    edge_batch = batch[edge_src]
    edge_vec = (pos[edge_dst]
                - pos[edge_src]
                + torch.einsum('ni,nij->nj',
                            edge_shift,
                            lattice[edge_batch]))
    edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)
    x = type_onehot[[type_encoding[atom] for atom in symbols]]

    #pos = pos.type(default_dtype)
    #edge_vec = torch.tensor(edge_vec)
    #edge_index = torch.tensor(edge_index)

    return pos, lattice, x, edge_vec, edge_index

#----------------------------------------------------------------#

def make_dataset(data:MicroState,
                 radial_cutoff:float,
                 default_dtype=torch.float64):
    
    # species = data.all_types()
    # type_onehot, type_encoding = get_type_onehot_encoding(species)    

    systems = data.to_ase()

    energy       = torch.tensor(data.properties["potential"])
    polarization = torch.tensor(data.properties["totalpol"])
    forces       = torch.tensor(data.forces)

    dataset = [None] * len(systems)
    n = 0 
    for crystal, e, p, f in tqdm(zip(systems,energy,polarization,forces),
                                 total=len(systems), 
                                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        # edge_src, edge_dst, edge_shift = \
        #     neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)

        pos, lattice, x, edge_vec, edge_index = preprocess( lattice=crystal.cell.array,
                                                            positions=torch.tensor(crystal.get_positions()),#.flatten(),
                                                            symbols=crystal.get_chemical_symbols(),
                                                            radial_cutoff=radial_cutoff,
                                                            default_dtype=default_dtype)

        
        # pos     = torch.tensor(crystal.get_positions())
        # lattice = torch.tensor(crystal.cell.array).unsqueeze(0) # We add a dimension for batching
        # x       = type_onehot[[type_encoding[atom] for atom in crystal.get_chemical_symbols()]]

        # edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)

        data = Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            radial_cutoff = radial_cutoff,
            symbols = crystal.get_chemical_symbols(),
            edge_index=edge_index,
            edge_vec=edge_vec,
            energy=e, # energy
            polarization=p, # polarization
            forces=f, # forces
            Natoms=crystal.get_global_number_of_atoms(), # valid only if all the structures have the same number of atoms
        )

        dataset[n] = data
        n += 1
    return dataset

#----------------------------------------------------------------#

def make_datapoint(lattice, positions, symbols, radial_cutoff, default_dtype=torch.float64):#, fake=None):

    pos, lattice, x, edge_vec, edge_index = preprocess(lattice, positions, symbols, radial_cutoff, default_dtype)
    
    return Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            edge_index=edge_index,
            edge_vec=edge_vec,
        )