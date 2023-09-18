from torch_geometric.data import Data
import torch
from torch.func import jacrev
import numpy as np

from .iPIinterface import iPIinterface
from .Methods4Training import EDFMethods4Training
from .Methods4AngularOutput import Methods4AngularOutput
from miscellaneous.elia.good_coding import froze
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetworkManager')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

__all__ = ["SabiaNetworkManager"]

# @froze
class SabiaNetworkManager(EDFMethods4Training,iPIinterface):

    def __init__(self: T, 
                 output: str = "ED",
                 reference:bool=False,
                 # dipole:torch.tensor=None,
                 pos:torch.tensor=None,
                 shift=None,
                 phases:bool=False,
                 **kwargs) -> None:
        
        # iPIinterface.__init__(self,**kwargs)
        # SabiaNetwork.__init__(self,**kwargs)

        # if debug is None :
        #     debug = list()
        
        super().__init__(**kwargs)

        self.output = output
        if self.output not in ["E", "D", "ED", "EF", "EDF"]:
            raise ValueError("'output' must be 'E', 'D', 'EF', 'ED' or 'EDF'")

        #self.lattice = torch.Tensor()
        #self.symbols = list()
        #self.R = torch.Tensor()

        self.reference = reference
        self.shift = shift

        self.phases = phases
        if self.output != "D" and self.phases :
            raise ValueError("You can use 'phases'=true only with 'output'='D'")
        
        # overwrite loss
        if self.phases :
            self.loss = self.angular_loss

        # self.debug = debug

        self._forces = None
        self._bec = None
        self._X = None

        if self.reference :

            # if dipole is None :
            #     raise ValueError("'dipole' can not be 'None'")
            # self.ref_dipole = torch.tensor(dipole)
            
            if pos is None :
                raise ValueError("'pos' can not be 'None'")
            self.ref_pos = torch.tensor(pos)

        pass

    # https://stackoverflow.com/questions/2345944/exclude-objects-field-from-pickling-in-python
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["_forces"]
        del state["_bec"]
        del state["_X"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self._forces = None
        self._bec = None
        self._X = None
        pass

    @staticmethod
    def batch(X):
        return len(torch.unique(X.batch))

    def _dummy_output_R(self: T, R:torch.tensor) -> torch.tensor:
        """Potential Energy Surface
        Input:
            - R: (N,3) tensor of positions
        Output:
            - potential energy
        Comments:
            - this methods will be automatically differentiated to get the forces"""

        # We reshape the positions so we can exploit the batches
        self._X.pos = R.reshape((-1,3))

        # I need to delete this attribute since the NN depends on the
        # relative positions 'edge_vec', but these HAVE to be "related" 
        # to the input 'R', so we simply recompute them.

        if hasattr(self._X,"edge_vec"):
            del self._X.edge_vec

        # Compute the output
        y = self(self._X)

        return y
    
    def energy(self: T, X) -> torch.tensor:

        if self.output not in ["E","ED","EDF"]:
            raise ValueError("'energy' not present in the output of this 'torch.nn.Module'")
        
        # Compute the output
        y = self(X)
        
        # Extract the energy in case we are evaluating also the dipole
        y = y if self.output == "E" else y[:,0]

        return y

    def _pes(self: T, R:torch.tensor) -> torch.tensor:
        """Return the potential energy for a given set of nuclear coordinates."""

        if self.output not in ["E","ED","EDF","EF"]:
            raise ValueError("'energy' not present in the output of this 'torch.nn.Module'")

        # Compute the output
        y = self._dummy_output_R(R)
        
        # Extract the energy in case we are evaluating also the dipole
        y = y if self.output == "E" else y[:,0]

        return y

    def forces(self: T, X, recompute=False) -> torch.tensor:
        """Compute the forces by automatic differentiating the energy"""

        # Compute the jacobian of the energy
        if ( not hasattr(self,"_forces") or self._forces is None) or recompute:
            self._forces = jacrev(self._pes)

        # Save the information into a global variable
        # that will be used inside 'self._pes'
        self._X = X

        # Reshape the positions according to the batches
        batch_size = self.batch(X)
        R = X.pos.reshape((batch_size,-1,3))

        # I do not know if I actually need this line
        #R.requires_grad_(True)

        # Evaluate the jacobian.
        # The positions are reshaped inside 'self._pes', but in this way
        # we get 'y' with the batches in a separate dimension/axis.
        y = self._forces(R)

        # Since 'jacrev' can not distinguish between batches and positions
        # the first dimension/axis of 'y' is spurious:
        # R.shape = [batch,Na,3]  -->  y = self._forces(R).shape = [batch,batch,Na,3]
        # with y[i,i] = ['non zero'] but y[i,j] = [0,...0]
        #  
        # It's easier to get this bigger tensor and extract its diagonal (y --> y[i,i])
        # than make a 'for' loop since we should modify also all the attributes of 'self._X'!

        # We need to take the diagonal of 'y' since one dimension is spurious as we said
        y = torch.diagonal(y,offset=0,dim1=0,dim2=1)
        
        # The 'diagonalized' dimension is put at the end of the tensor 
        # according to 'torch.diagonal' documentation.
        # Then we simply permute the tensor dimensions/axis .
        y = y.permute(2,0,1)

        # Reshape the forces so we have only two axis: 
        # - the first for the batches
        # - the second with all the coordinates
        y = y.reshape((batch_size,-1))

        return y
        
    def dipole(self: T, X) -> torch.tensor:
        """Return the dipole of the system"""

        if self.output not in ["D","ED","EDF"]:
            raise ValueError("'dipole' not present in the output of this 'torch.nn.Module'")
        
        # Compute the output
        y = self(X)
        
        # Extract the dipole in case we are evaluating also the energy (and the forces)
        if self.output in ["ED","EDF"]:
            y = y[:,1:4]

        return y
        
        # batch_size = self.batch(X)
        # y = torch.zeros((batch_size,3), requires_grad=requires_grad)

        # for n in range(batch_size):

        #     # prepare data
        #     x,R = self._prepare(X,n)
        
        #     tmp = self(self._X)

        #     if self.output == "D":
        #         y.data[n,:] = tmp
        #     elif self.output in ["ED","EDF"]:
        #         y.data[n,:] = tmp[:,1:4]

        # return y

    def _dipole(self: T, R:torch.tensor) -> torch.tensor:
        """Return the dipole of the system for a given set of nuclear coordinates."""

        if self.output not in ["D","ED","EDF"]:
            raise ValueError("'dipole' not present in the output of this 'torch.nn.Module'")

        # Compute the output
        y = self._dummy_output_R(R)
        
        # Extract the dipole in case we are evaluating also the energy (and the forces)
        if self.output in ["ED","EDF"]:
            y = y[:,1:4]
        return y

    def BEC(self: T, X, recompute=False) -> torch.tensor:
        """Compute the Born Effective Charges tensors by automatic differentiating the polarization"""

        # Compute the jacobian of the dipole: 
        # Z^i_j = \frac{\Omega}{q_e} \frac{\partial P^i}{\partial R_j}
        #       = \frac{1}{q_e} \frac{\partial d^i}{\partial R_j}
        # We assume that the dipole and the coordinates are given in atomic units
        # then q_e = 1 and we can compute Z as directly the jacobian of the dipole.
        #
        if ( not hasattr(self,"_bec") or self._bec is None) or recompute:
            self._bec = jacrev(self._dipole)

        # Save the information into a global variable
        # that will be used inside 'self._pes'
        self._X = X

        # Reshape the positions according to the batches
        batch_size = self.batch(X)
        R = X.pos.reshape((batch_size,-1,3))

        # I do not know if I actually need this line
        #R.requires_grad_(True)

        # Evaluate the jacobian.
        # The positions are reshaped inside 'self._pes', but in this way
        # we get 'y' with the batches in a separate dimension/axis.
        y = self._bec(R)

        # Since 'jacrev' can not distinguish between batches and positions
        # the first dimension/axis of 'y' is spurious:
        # R.shape = [batch,Na,3]  -->  y = self._forces(R).shape = [batch,batch,Na,3]
        # with y[i,i] = ['non zero'] but y[i,j] = [0,...0]
        #  
        # It's easier to get this bigger tensor and extract its diagonal (y --> y[i,i])
        # than make a 'for' loop since we should modify also all the attributes of 'self._X'!

        # put all the coordinates in the last axis
        y = y.reshape((batch_size,3,batch_size,-1))

        # We need to take the diagonal of 'y' since one dimension is spurious as we said
        y = torch.diagonal(y,offset=0,dim1=0,dim2=2)
        
        # The 'diagonalized' dimension is put at the end of the tensor 
        # according to 'torch.diagonal' documentation.
        # Then we simply permute the tensor dimensions/axis .
        y = y.permute(2,0,1)

        # Reshape the forces so we have only two axis: 
        # - the first for the batches
        # - the second with all the coordinates
        # y = y.reshape((batch_size,3,-1))

        return y

    # def check_equivariance(self:T,X: Data,angles=None)->np.ndarray:

    #     from e3nn import o3

    #     with torch.no_grad():

    #         batch_size = len(np.unique(X.batch))
    #         y = np.zeros(batch_size)

    #         irreps_out = o3.Irreps(self.irreps_out)

    #         if angles is None :
    #             angles = o3.rand_angles()

    #         for n in range(batch_size):

    #             irreps_in = "{:d}x1o".format(int(X.Natoms[n]))
    #             irreps_in = o3.Irreps(irreps_in)

    #             # prepare data -> prepare self.R and self._X

    #             # rotate output
    #             x,R = self._prepare(X,n)
    #             R_out = irreps_out.D_from_angles(*angles)
    #             y0 = self(x)
    #             rot_out = torch.einsum("ij,zj->zi",R_out,y0)

    #             # rotate input
    #             R_in  = irreps_in.D_from_angles(*angles)
    #             x,R = self._prepare(X,n,R_in)
    #             #R = R.reshape((1,-1))

                
                
    #             # R = x.pos.reshape((1,-1))                
    #             # rot_in  = torch.einsum("ij,zj->zi",R_in,R)

    #             # l = x.lattice.reshape((1,-1))
    #             # l_in  = torch.einsum("ij,zj->zi",R_in,l)
                
    #             # x.pos     = rot_in.reshape((-1,3))
    #             # x.lattice = l_in.reshape((-1,3))

    #             #x,R_ = self._prepare(X,n,replace_pos=rot_in.reshape((-1,3)))
    #             out_rot = self(x)

    #             thr = torch.norm(rot_out - out_rot)
                
    #             y[n] = thr
        
    #     return y

        
def main():

    import os
    from miscellaneous.elia.classes import MicroState
    from miscellaneous.elia.nn.dataset import make_dataset
    from miscellaneous.elia.nn import _make_dataloader

    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)

    # Changing the current working directory
    os.chdir('./water/')

    max_radius = 6.0

    ##########################################

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/microstate.pickle"

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "data/i-pi.positions_0.xyz"
        instructions = {"properties" : "data/i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"data/i-pi.forces_0.xyz"}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ########################################## 

    RESTART = False 
    READ = True
    SAVE = False
    savefile = "data/dataset"

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("building dataset")

        if os.path.exists(savefile+".torch"):
            dataset = torch.load(savefile+".torch")
        else :
            dataset = make_dataset( data=data,max_radius=max_radius)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = 2000
        i = 500#int(p_test*len(dataset))
        j = 500#int(p_val*len(dataset))

        train_dataset = dataset[:n]
        val_dataset   = dataset[n:n+j]
        test_dataset  = dataset[n+j:n+j+i]

        del dataset

    else :
        print("reading datasets from file {:s}".format(savefile))
        train_dataset = torch.load(savefile+".train.torch")
        val_dataset   = torch.load(savefile+".val.torch")
        test_dataset  = torch.load(savefile+".test.torch")
        SAVE = False
            
    if SAVE :
        print("saving dataset to file {:s}".format(savefile))
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")

    print(" test:",len(test_dataset))
    print("  val:",len(val_dataset))
    print("train:",len(train_dataset))

    ##########################################

    for folder in ["results/","results/networks/","results/dataframes","results/images"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    ##########################################

    OUTPUT = "E"

    irreps_in = "{:d}x0e".format(len(data.all_types()))
    if OUTPUT == "E":
        irreps_out = "1x0e"
    elif OUTPUT in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif OUTPUT == "D":
        irreps_out = "1x1o"

    print("irreps_out:",irreps_out)

    max_radius = 6.0
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,    # vector (L=1 and odd parity) to output the dipole
        "max_radius":max_radius, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        # "num_nodes":2,
        "mul":10,
        "layers":2,
        "lmax":1,
        "default_dtype" : default_dtype,
    }
    net = SabiaNetworkManager(output=OUTPUT,**model_kwargs)
    print(net)
    print("net.irreps_in:",net.irreps_in)   #>> net.irreps_in: 2x0e
    print("net.irreps_out:",net.irreps_out) #>> net.irreps_out: 1x0e+1x1o

    # this line does not affect the results (at least the code does not complain)
    # net.eval()

    # it works for any batch_size
    train_dataloader = _make_dataloader(train_dataset,batch_size=50)
    for X in train_dataloader:

        # the water molecule has 3 atoms
        # this means that all the coordinates have lenght 9
        # if we predict energy and dipole the output will have lenght 4

        y = net(X)
        print("y.shape:",y.shape)           #>> y.shape: torch.Size([50, 4])

        E = net.energy(X)
        print("E.shape:",E.shape)           #>> E.shape: torch.Size([50])

        F = net.forces(X)
        print("F.shape:",F.shape)           #>> F.shape: torch.Size([50, 9])
        
        if OUTPUT in ["ED","D"] :
            P = net.dipole(X)
            print("P.shape:",P.shape)           #>> P.shape: torch.Size([50, 3])
            
            BEC = net.BEC(X)
            print("BEC.shape:",BEC.shape)       #>> BEC.shape: torch.Size([50, 3, 9])

        diff = net.check_equivariance(X)
        print("diff:",diff.mean())          #>> diff: 6.481285499819072e-15

        break

    print("\n Job done :)")

if __name__ == "__main__":
    main()