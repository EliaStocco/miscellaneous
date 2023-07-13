import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list

def make_dataset(systems,\
                 type_onehot,\
                 type_encoding,\
                 output,\
                 radial_cutoff,\
                 default_dtype=torch.float64):
    
    # better to do this
    output = torch.tensor(output)

    #print("output:",output)
    #print("output value shape:",output[0].shape)

    dataset = [None] * len(systems)
    n = 0 
    bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
    for crystal, out in tqdm(zip(systems, output),total=len(systems), bar_format=bar_format):
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        edge_src, edge_dst, edge_shift = \
            neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)

        data = Data(
            pos=torch.tensor(crystal.get_positions()),
            lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
            x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            # edge_src=torch.tensor(edge_src,dtype=int),
            # edge_dst=torch.tensor(edge_dst,dtype=int),
            edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
            yreal=out  # polarization (??assumed to be normalized "per atom" ??)
        )

        dataset[n] = data
        n += 1
    return dataset#,dtype=torch_geometric.data.Data)