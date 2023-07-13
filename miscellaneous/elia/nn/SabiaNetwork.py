from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter
from e3nn import o3
import torch
import torch_geometric
from .MessagePassing import MessagePassing
from typing import Dict, Union

# https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html

class SabiaNetwork(torch.nn.Module):
    def __init__(self,
        irreps_in,
        irreps_out,
        max_radius,
        #irreps_node_attr,
        num_neighbors,
        num_nodes,
        mul=10,
        layers=1,
        lmax=1,
        number_of_basis=10,
        p=[1,-1],
        debug=False,
        pool_nodes=True) -> None:
        
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.debug = debug
        
        if self.debug:
            print()
            print("irreps_in:",irreps_in)
            print("irreps_out:",irreps_out)

        irreps_node_hidden = o3.Irreps([(mul, (l, pp)) for l in range(lmax + 1) for pp in p])
        if self.debug:
            print("irreps_node_hidden:",irreps_node_hidden)
        #irreps_node_hidden = o3.Irreps([(mul, (l, 1)) for l in range(lmax + 1) ])

        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output
    
#     @reloading
#     def evaluate(self,dataset):
#         self.eval()
#         with torch.no_grad():
#             temp = self(dataset[0])
#             output = torch.zeros((len(dataset),*temp.shape))
#             output[0] = temp
#             for n in range(1,len(dataset)):
#                 output[n] = self(dataset[n])
#             return output

    # Overwriting preprocess method of SimpleNetwork to adapt for periodic boundary data
    def preprocess(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination

        # We need to compute this in the computation graph to backprop to positions
        # We are computing the relative distances + unit cell shifts from periodic boundaries
        edge_batch = batch[edge_src]
        edge_vec = (data['pos'][edge_dst]
                    - data['pos'][edge_src]
                    + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][edge_batch]))

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor: 
        if self.debug:
            print("SabiaNetwork:1")
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data
        if self.debug:
            print("SabiaNetwork:2")

        edge_attr = o3.spherical_harmonics( range(self.lmax + 1), edge_vec, True, normalization="component")
        if self.debug:
            print("SabiaNetwork:3")
        
        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)
        if self.debug:
            print("SabiaNetwork:4")

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)
        if self.debug:
            print("SabiaNetwork:5")

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        if self.debug:
            print("SabiaNetwork:6")
        
        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs