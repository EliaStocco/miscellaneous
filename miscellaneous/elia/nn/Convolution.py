from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from torch_scatter import scatter
import torch
from e3nn import o3

@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    # irreps_node_input : o3.Irreps
    # irreps_node_output : o3.Irreps
    # irreps_node_attr : o3.Irreps
    # irreps_edge_attr : o3.Irreps
    # debug : bool
    # num_neighbors: int
    # sc : FullyConnectedTensorProduct
    # lin1 : FullyConnectedTensorProduct
    # fc : FullyConnectedNet
    # tp : TensorProduct
    # lin2 : FullyConnectedTensorProduct
    # lin3 : FullyConnectedTensorProduct


    def __init__(
        self,\
        irreps_node_input,\
        irreps_node_attr,\
        irreps_edge_attr,\
        irreps_node_output,\
        fc_neurons,\
        num_neighbors,
        debug=False,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors
        self.debug = debug

        if self.debug: 
            print()
            print("Convolution.irreps_node_input (in-1)",self.irreps_node_input)
            print("Convolution.irreps_node_attr (in-2)",self.irreps_node_attr)
            print("Convolution.irreps_node_output (out)",self.irreps_node_output)
        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        if self.debug: 
            print("Convolution.irreps_node_input (in-1)",self.irreps_node_input)
            print("Convolution.irreps_node_attr (in-2)",self.irreps_node_attr)
            print("Convolution.irreps_node_input (out)",self.irreps_node_input)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        if self.debug: 
            print("TensorProduct.tp.irreps_node_input (in-1)",self.irreps_node_input)
            print("TensorProduct.tp.irreps_edge_attr (in-2)",self.irreps_edge_attr)
            print("TensorProduct.tp.irreps_mid (out)",irreps_mid)
        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)

        # Elia: undertand if the final '0e' is okay even when I add a vectorial input to the nodes
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        if self.debug: print("Convolution:1")
        weight = self.fc(edge_scalars)

        if self.debug: 
            print("Convolution:2")
            print("node_input:",node_input)
            print("node_attr:",node_attr)
        node_self_connection = self.sc(node_input, node_attr)
        
        if self.debug: print("Convolution:3")
        node_features = self.lin1(node_input, node_attr)

        if self.debug: 
            print("Convolution:4")
            print("node_features[edge_src]:",node_features[edge_src])
            print("edge_attr:",node_attr)
            print("weight:",weight)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        
        if self.debug: print("Convolution:5")
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

        if self.debug: print("Convolution:6")
        node_conv_out = self.lin2(node_features, node_attr)
        
        if self.debug: print("Convolution:7")
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection
        
        if self.debug: print("Convolution:8")
        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out
