import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list
from miscellaneous.elia.functions import add_default
from miscellaneous.elia.nn.functions import symbols2x
from ase import Atoms
import numpy as np

#----------------------------------------------------------------#
def compute_edge_vec(pos,lattice,edge_src,edge_dst,edge_shift,pbc):
    edge_vec = pos[edge_dst] - pos[edge_src]
    if pbc :
        batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        edge_batch = batch[edge_src]
        edge_vec = edge_vec + torch.einsum('ni,nij->nj',edge_shift,lattice[edge_batch])
    return edge_vec
#----------------------------------------------------------------#

def my_neighbor_list(lattice,pos,max_radius,pbc):

    # detach
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().numpy()

    nl = lambda q,a : neighbor_list( quantities=q,
                                     a = a,       
                                     cutoff=max_radius,
                                     self_interaction=True)

    if pbc :
        if isinstance(lattice, torch.Tensor):
            lattice = lattice.detach().numpy()
        atoms = Atoms(positions=pos,cell=lattice,pbc=[True]*3)
        edge_src, edge_dst, edge_shift = nl("ijS",atoms)
    else :
        atoms = Atoms(positions=pos,pbc=[False]*3)
        edge_src, edge_dst = nl("ij",atoms)
        edge_shift = None

    # edge_src, edge_dst, edge_shift = neighbor_list( quantities="ijS",
    #                                                 a = atoms,
    #                                                 cutoff=max_radius,
    #                                                 self_interaction=True)

    # edge_src, edge_dst, edge_shift = primitive_neighbor_list(
    #     quantities="ijS",
    #     pbc=[pbc]*3,
    #     cell=lattice,
    #     positions=pos,
    #     cutoff=max_radius,
    #     self_interaction=True
    # )

    return edge_src, edge_dst, edge_shift

#----------------------------------------------------------------#

def preprocess(lattice, positions, symbols, max_radius, default_dtype, pbc, requires_grad=None):

    # lattice has to be in 'ase' format:
    # | a_1x a_1y a_1z |
    # | a_2x a_2y a_2z |
    # | a_3x a_3y a_3z |
        
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

    edge_src, edge_dst, edge_shift = my_neighbor_list(lattice,pos,max_radius,pbc)
    
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

    # 'lattice' can be None for non-periodic systems
    if lattice is not None :
        if requires_grad["lattice"] is not None :
            if isinstance(lattice,torch.Tensor):
                lattice.requires_grad_(requires_grad["lattice"])
            else :
                lattice = torch.tensor(lattice,requires_grad=requires_grad["lattice"])
        lattice = lattice.to(default_dtype).unsqueeze(0)

    # I need these lines here before calling 'einsum'
    if pbc :
        edge_shift = torch.tensor(edge_shift,dtype=default_dtype)

    edge_vec = compute_edge_vec(pos,lattice,edge_src,edge_dst,edge_shift,pbc)

    if requires_grad["edge_vec"] is not None:
        edge_vec.requires_grad_(requires_grad["edge_vec"])

    edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)
    if requires_grad["edge_index"] is not None:
        edge_index.requires_grad_(requires_grad["edge_index"])

    #x = type_onehot[[type_encoding[atom] for atom in symbols]]
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(default_dtype)
    if requires_grad["x"] is not None :
        x.requires_grad_(requires_grad["x"])

    return pos, lattice, x, edge_vec, edge_index, edge_shift

#----------------------------------------------------------------#

def make_dataset(systems:Atoms,max_radius:float,disable:bool=False):

    dataset = [None] * len(systems)
    with tqdm(enumerate(systems),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',disable=disable) as bar:
        for n,crystal in bar:

            pbc = np.any(crystal.get_pbc())
            lattice = torch.from_numpy(crystal.cell.array) if pbc else None

            pos, lattice, x, edge_vec, edge_index, edge_shift = \
                preprocess( lattice=lattice,
                            positions=torch.from_numpy(crystal.get_positions()),
                            symbols=crystal.get_chemical_symbols(),
                            max_radius=max_radius,
                            pbc = pbc,
                            default_dtype=torch.float64,
                            requires_grad={"pos":True,"lattice":True})
        
            data_dict = {
                "pos": pos,
                "lattice": lattice,       # it should be None if pbc=False
                "edge_shift": edge_shift, # it should be None if pbc=False
                "x": x,
                "pbc" : pbc,
                "max_radius": max_radius,
                "symbols": crystal.get_chemical_symbols(),
                "edge_index": edge_index,
                "edge_vec": edge_vec,
                "Natoms": int(crystal.get_global_number_of_atoms()),
            }

            for k in ["numbers","positions"]:
                if k in crystal.arrays.keys():
                    del crystal.arrays[k]
                else :
                    raise ValueError("array '{:s}' should be in 'crystal'".format(k))
                
            for k,array in crystal.arrays.items():
                data_dict[k] = torch.tensor(array)
            for k,info in crystal.info.items():
                data_dict[k] = torch.tensor(info)

            data = Data(**data_dict)

            dataset[n] = data

    return dataset

# #----------------------------------------------------------------#

# def make_dataset(data:MicroState,
#                  max_radius:float,
#                  output:str,
#                  pbc:bool,
#                  indices:list=None,
#                  default_dtype=torch.float64,
#                  requires_grad:bool=True):#,\
#                  #output_method:callable=None):
    
#     # species = data.all_types()
#     # type_onehot, type_encoding = get_type_onehot_encoding(species)    

#     # if output_method is None:
#     #     # ELIA: modify 'same_lattice' to 'false'
#     #     output_method = lambda : data.get_dipole(same_lattice=True)

#     if indices is not None :
#         indices = np.loadtxt(indices).astype(int)
#         indices = list(indices)
#         data = data.subsample(indices)

#     systems = data.to_ase()

#     if "E" in output:
#         energy = torch.from_numpy(data.properties["potential"])
    
#     if "D" in output:
#         dipole = torch.from_numpy(data.properties["dipole"]) #torch.tensor(output_method())

#     if "F" in output :
#         forces = torch.from_numpy(data.forces)

#     dataset = [None] * len(systems)
#     n = 0 
#     for n in tqdm(range(len(systems)),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

#         crystal = systems[n]

#     # for crystal, e, d, f in tqdm(zip(systems,energy,dipole,forces),
#     #                              total=len(systems), 
#     #                              bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        
#         # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
#         # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
#         # edge_src, edge_dst, edge_shift = \
#         #     neighbor_list("ijS", a=crystal, cutoff=max_radius, self_interaction=True)

#         lattice = torch.from_numpy(crystal.cell.array) if pbc else None

#         pos, lattice, x, edge_vec, edge_index, edge_shift = \
#             preprocess( lattice=lattice,
#                         positions=torch.from_numpy(crystal.get_positions()),#.flatten(),
#                         symbols=crystal.get_chemical_symbols(),
#                         max_radius=max_radius,
#                         pbc = pbc,
#                         default_dtype=default_dtype,
#                         requires_grad={"pos":requires_grad,"lattice":requires_grad})

#         # if 'pbc' == True then 'lattice' should be 'None'
    
#         data_dict = {
#             "pos": pos,
#             "lattice": lattice,       # it should be None if pbc=False
#             "edge_shift": edge_shift, # it should be None if pbc=False
#             "x": x,
#             "max_radius": max_radius,
#             "symbols": crystal.get_chemical_symbols(),
#             "edge_index": edge_index,
#             "edge_vec": edge_vec,
#             "Natoms": torch.tensor(crystal.get_global_number_of_atoms()).to(int),  # valid only if all the structures have the same number of atoms
#             "index" : n # to be able to reconstruct the dataset
#         }

#         # if pbc :
#         #     data_dict["lattice"] = lattice
#         if "E" in output :
#             data_dict["energy"] = energy[n]
#         if "D" in output :
#             data_dict["dipole"] = dipole[n]    
#         if "F" in output :
#             data_dict["forces"] = forces[n]

#         data = Data(**data_dict)

#         dataset[n] = data
#         # n += 1
#     return dataset

#----------------------------------------------------------------#

def make_datapoint(lattice, positions, symbols, max_radius, default_dtype, pbc, **argv)->Data:

    # lattice has to be in 'ase' format:
    # | a_1x a_1y a_1z |
    # | a_2x a_2y a_2z |
    # | a_3x a_3y a_3z |

    pos, lattice, x, edge_vec, edge_index, edge_shift = \
        preprocess(lattice, positions, symbols, max_radius, default_dtype,pbc,**argv)
    
    return Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_shift=edge_shift,
            batch = torch.full((len(pos),),0) # I need this in i-PI
        )