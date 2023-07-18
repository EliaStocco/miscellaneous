import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list, primitive_neighbor_list
from miscellaneous.elia.classes import MicroState
from . import get_type_onehot_encoding
from ase import Atoms
import numpy as np

def make_dataset(data:MicroState,\
                 radial_cutoff:float):
    
    species = data.all_types()
    type_onehot, type_encoding = get_type_onehot_encoding(species)    

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
        edge_src, edge_dst, edge_shift = \
            neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
        
        pos     = torch.tensor(crystal.get_positions())
        lattice = torch.tensor(crystal.cell.array).unsqueeze(0) # We add a dimension for batching
        x       = type_onehot[[type_encoding[atom] for atom in crystal.get_chemical_symbols()]]

        edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)

        data = Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            radial_cutoff = radial_cutoff,
            symbols = crystal.get_chemical_symbols(),
            edge_index=edge_index,
            edge_shift=torch.tensor(edge_shift),
            energy=e, # energy
            polarization=p, # polarization
            forces=f, # forces
            Natoms=crystal.get_global_number_of_atoms(), # valid only if all the structures have the same number of atoms
        )

        dataset[n] = data
        n += 1
    return dataset

#----------------------------------------------------------------#

def make_datapoint(lattice, positions,fake,radial_cutoff, symbols):

    from copy import copy

    #with torch.no_grad():
    #lattice = lattice#.unsqueeze(0) # We add a dimension for batching
    positions = positions.reshape((-1,3))
    fake = fake.reshape((-1,3))

    #pos = np.asarray(positions.detach())
    crystal = Atoms(cell=lattice,positions=fake,symbols=symbols)

    species = np.unique(symbols)
    type_onehot, type_encoding = get_type_onehot_encoding(species)

    # def neighbor_list(quantities, a, cutoff, self_interaction=False,
    #                   max_nbins=1e6):
    

    # edge_src, edge_dst, edge_shift = primitive_neighbor_list(quantities="ijS",
    #                         pbc= (True,True,True),
    #                         cell=lattice,
    #                         positions=positions,
    #                         cutoff=radial_cutoff,
    #                         numbers=Atoms(symbols=symbols).numbers,
    #                         self_interaction=True,
    #                         max_nbins=1e6)

    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
    
    return Data(
            pos=positions.reshape((-1,3)),
            lattice=lattice.unsqueeze(0),  # We add a dimension for batching
            x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            edge_shift=torch.tensor(edge_shift),
        )