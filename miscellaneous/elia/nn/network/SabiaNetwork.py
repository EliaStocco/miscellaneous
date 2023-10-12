from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import BatchNorm
import torch
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from .MessagePassing import MessagePassing
from miscellaneous.elia.nn.dataset import compute_edge_vec
import warnings
from typing import Dict, Union
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetwork')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

# https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html

class SabiaNetwork(torch.nn.Module):

    default_dtype = torch.float64 #: torch.dtype
    # lmax : int
    # max_radius: float
    # number_of_basis : int
    # num_nodes : int
    # pool_nodes : bool
    # debug : bool
    # mp : MessagePassing
    # irreps_in : o3.Irreps
    # irreps_out : o3.Irreps

    def __init__(self:T,
                irreps_in:str,
                irreps_out:str,
                max_radius:float,
                num_neighbors:int,
                num_nodes:int,
                irreps_node_attr:str="0e",
                mul:int=10,
                layers:int=1,
                lmax:int=1,
                number_of_basis:int=10,
                p:list=["o","e"],
                debug:bool=False,
                pool_nodes:bool=True,
                dropout_probability:float=0,
                batchnorm:bool=True,
                pbc:bool=True,
                **argv) -> None:
                
        super().__init__(**argv)

        # self.default_dtype = default_dtype
        torch.set_default_dtype(self.default_dtype)
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.debug = debug
        self.pbc = pbc

        # https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html
        if self.pool_nodes :
            self.num_nodes = 1
        
        # if self.debug: print("irreps_in:",irreps_in)
        # if self.debug: print("irreps_out:",irreps_out)

        tmp = ["{:d}x{:d}{:s}".format(mul,l,pp) for l in range(lmax + 1) for pp in p]
        irreps_node_hidden = o3.Irreps("+".join(tmp))
        # if self.debug: print("irreps_node_hidden:",tmp)
        
        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
            dropout_probability=dropout_probability,
            batchnorm=batchnorm
        )

        # it's not clear to me what this class actually does
        self._sh = o3.SphericalHarmonics( range(self.lmax + 1), True, normalization="component")

        # batch normalization to normalize output
        self._bn = BatchNorm(irreps=self.mp.irreps_out,affine=True)
        # self.factor = torch.nn.Parameter(torch.tensor(1.0))
        
        self.irreps_in  = self.mp.irreps_in
        self.irreps_out = self.mp.irreps_out

        pass
    
    # Overwriting preprocess method of SimpleNetwork to adapt for periodic boundary data
    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:

        msg = "The author does no longer trust 'SabiaNetwork.preprocess' method. \
            Use 'make_dataset.preprocess' instead."
        
        if 'batch' in data:
            batch = data['batch']
        else:
            # warnings.warn("'batch' is missing. " + msg )
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if "edge_index" in data:
            edge_src = data['edge_index'][0]  # Edge source
            edge_dst = data['edge_index'][1]  # Edge destination
        else :
            warnings.warn("'edge_index' is missing. " + msg )
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

        if "edge_vec" in data and ( data["pos"].requires_grad == data["edge_vec"].requires_grad ):
            edge_vec = data['edge_vec']
        
        else :
            # warnings.warn("'edge_vec' is missing. " + msg )
            
            # We need to compute this in the computation graph to backprop to positions
            # We are computing the relative distances + unit cell shifts from periodic boundaries
            edge_vec = compute_edge_vec(pos=data['pos'],
                                        lattice=data['lattice']       if self.pbc else None,
                                        edge_shift=data['edge_shift'] if self.pbc else None,
                                        edge_src=edge_src,
                                        edge_dst=edge_dst,
                                        pbc=self.pbc)
            # edge_batch = batch[edge_src]
            # if self.pbc : 
            #     edge_vec = (data['pos'][edge_dst]
            #                 - data['pos'][edge_src]
            #                 + torch.einsum('ni,nij->nj',
            #                             data['edge_shift'].type(self.default_dtype),
            #                             data['lattice'][edge_batch].type(self.default_dtype)))
            # else :
            #     edge_vec = data['pos'][edge_dst] - data['pos'][edge_src]

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor: 
        
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        # del data

        # edge_attr = o3.spherical_harmonics( range(self.lmax + 1), edge_vec, True, normalization="component")
        edge_attr = self._sh(edge_vec)
        
        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        # Elia: this can be improved
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        
        if self.pool_nodes:
            # Elia: understand what 'scatter' does
            node_outputs = scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        # else:
        #     y = node_outputs

        # return node_outputs * self.factor
        return self._bn(node_outputs)
        