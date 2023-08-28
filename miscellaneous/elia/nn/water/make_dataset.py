import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list, primitive_neighbor_list
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.functions import add_default
from . import symbols2x
from ase import Atoms
import numpy as np


#----------------------------------------------------------------#

def my_neighbor_list(lattice,pos,max_radius):

    # detach
    if isinstance(lattice, torch.Tensor):
        lattice = lattice.detach().numpy()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().numpy()

    edge_src, edge_dst, edge_shift = primitive_neighbor_list(
        quantities="ijS",
        pbc=[True]*3,
        cell=lattice,
        positions=pos,
        cutoff=max_radius,
        self_interaction=True
    )

    return edge_src, edge_dst, edge_shift

#----------------------------------------------------------------#

def preprocess(lattice, positions, symbols, max_radius, default_dtype,requires_grad=None):
        
    default = { "pos"        : True,\
                "lattice"    : True,\
                "x"          : None,\
                "edge_vec"   : None,\
                "edge_index" : None }

    requires_grad = add_default(requires_grad,default)

    x = symbols2x(symbols)

    pos=positions.reshape((-1,3))
    if requires_grad["pos"] is not None :
        if isinstance(pos,torch.Tensor):
            pos.requires_grad_(requires_grad["pos"])
        else :
            pos = torch.tensor(pos,requires_grad=requires_grad["pos"])
    pos = pos.to(default_dtype)

    batch = pos.new_zeros(pos.shape[0], dtype=torch.long)

    edge_src, edge_dst, edge_shift = my_neighbor_list(lattice,pos,max_radius)
    
    # crystal = Atoms(cell=lattice,positions=pos,symbols=symbols,pbc=True)
    # edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=crystal, cutoff=max_radius, self_interaction=True)

    # edge_src, edge_dst, edge_shift = primitive_neighbor_list(
    #     quantities="ijS",
    #     pbc=[True]*3,
    #     cell=lattice,
    #     positions=pos,
    #     cutoff=max_radius,
    #     self_interaction=True
    # )

    # I need these lines here before calling 'einsum'
    edge_shift = torch.tensor(edge_shift,dtype=default_dtype)
    #lattice = torch.tensor(lattice,dtype=default_dtype).unsqueeze(0)# We add a dimension for batching
    if requires_grad["lattice"] is not None :
        if isinstance(pos,torch.Tensor):
            lattice.requires_grad_(requires_grad["lattice"])
        else :
            lattice = torch.tensor(lattice,requires_grad=requires_grad["lattice"])
    lattice = lattice.to(default_dtype).unsqueeze(0)

    edge_batch = batch[edge_src]
    edge_vec = (pos[edge_dst]
                - pos[edge_src]
                + torch.einsum('ni,nij->nj',
                            edge_shift,
                            lattice[edge_batch]))
    if requires_grad["edge_vec"] is not None:
        edge_vec.requires_grad_(requires_grad["edge_vec"])

    edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)
    if requires_grad["edge_index"] is not None:
        edge_index.requires_grad_(requires_grad["edge_index"])

    #x = type_onehot[[type_encoding[atom] for atom in symbols]]
    if requires_grad["x"] is not None :
        x = torch.tensor(x).requires_grad_(requires_grad["x"])

    #pos = pos.type(default_dtype)
    #edge_vec = torch.tensor(edge_vec)
    #edge_index = torch.tensor(edge_index)

    return pos, lattice, x, edge_vec, edge_index, edge_shift

#----------------------------------------------------------------#

def make_dataset(data:MicroState,
                 max_radius:float,
                 default_dtype=torch.float64):
    
    # species = data.all_types()
    # type_onehot, type_encoding = get_type_onehot_encoding(species)    

    systems = data.to_ase()

    energy       = torch.tensor(data.properties["potential"])
    dipole       = torch.tensor(data.get_dipole(same_lattice=False))
    forces       = torch.tensor(data.forces)

    dataset = [None] * len(systems)
    n = 0 
    for crystal, e, d, f in tqdm(zip(systems,energy,dipole,forces),
                                 total=len(systems), 
                                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        # edge_src, edge_dst, edge_shift = \
        #     neighbor_list("ijS", a=crystal, cutoff=max_radius, self_interaction=True)

        pos, lattice, x, edge_vec, edge_index, edge_shift = \
            preprocess( lattice=torch.from_numpy(crystal.cell.array),
                        positions=torch.from_numpy(crystal.get_positions()),#.flatten(),
                        symbols=crystal.get_chemical_symbols(),
                        max_radius=max_radius,
                        default_dtype=default_dtype,
                        requires_grad={"pos":False,"lattice":False})

        
        # pos     = torch.tensor(crystal.get_positions())
        # lattice = torch.tensor(crystal.cell.array).unsqueeze(0) # We add a dimension for batching
        # x       = type_onehot[[type_encoding[atom] for atom in crystal.get_chemical_symbols()]]

        # edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)

        data = Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            max_radius = max_radius,
            symbols = crystal.get_chemical_symbols(),
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_shift = edge_shift,
            energy=e, # energy
            dipole=d, # dipole
            forces=f, # forces
            Natoms=torch.tensor(crystal.get_global_number_of_atoms()).to(int), # valid only if all the structures have the same number of atoms
        )

        dataset[n] = data
        n += 1
    return dataset

#----------------------------------------------------------------#

def make_datapoint(lattice, positions, symbols, max_radius, default_dtype=torch.float64,**argv)->Data:

    pos, lattice, x, edge_vec, edge_index, edge_shift = \
        preprocess(lattice, positions, symbols, max_radius, default_dtype,**argv)
    
    return Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_shift=edge_shift
        )