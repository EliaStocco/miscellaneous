from torch_geometric.data import Data
import torch
from copy import copy
import numpy as np
from .SabiaNetwork import SabiaNetwork
from miscellaneous.elia.nn.water.make_dataset import make_datapoint
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetworkManager')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

class SabiaNetworkManager(SabiaNetwork):

    lattice : torch.Tensor
    radial_cutoff  : float
    _radial_cutoff : float
    symbols : list

    def __init__(self:T,radial_cutoff:float=0.0,output:str="EP",**kwargs)->None:
        super(SabiaNetworkManager,self).__init__(**kwargs)

        self.output = output
        if self.output not in ["E", "EP", "EPF"]:
            raise ValueError("'output' must be 'E', 'EP' or 'EPF'")
        
        self.grad = None
        self.lattice = None
        self.radial_cutoff = None
        self._radial_cutoff = radial_cutoff
        self.symbols = None
        pass

    def train(self:T,mode:bool)->T:
        if self.grad is not None:
            del self.grad     # delete the gradient
            self.grad = None  # build an empty gradient
        return super(SabiaNetworkManager,self).train(mode)
    
    # def evaluate(self:T,positions,lattice,radial_cutoff,symbols)->torch.tensor:
    #     x = make_datapoint( lattice=lattice,\
    #                         radial_cutoff=radial_cutoff,\
    #                         symbols=symbols,\
    #                         positions=positions)
    #     y = self(x)
    #     return y if self.output == "E" else y[:,0]
    
    #@torch.jit.script
    def PES(self:T,R:torch.tensor)->torch.tensor:
        """Potential Energy Surface
        Input:
            - R: (N,3) tensor of positions
        Output:
            - potential energy
        Comments:
            - this methods will be automatically differentiated to get the forces"""
        x = make_datapoint( lattice=self.lattice,\
                            radial_cutoff=self.radial_cutoff,\
                            symbols=self.symbols,\
                            positions=R,
                            fake=R.detach())
        y = self(x)
        return y if self.output == "E" else y[:,0]
    
    def forces(self:T,X:Data)->torch.tensor:

        batch_size = len(np.unique(X.batch))
        y = torch.zeros(X.pos.shape,requires_grad=True).reshape((batch_size,-1))

        for n in range(batch_size):

            index = X.batch == n
            self.lattice       = X.lattice[n]
            self.symbols       = X.symbols[n]
            self.radial_cutoff = X.radial_cutoff if hasattr(X,"radial_cutoff") else self._radial_cutoff
            # R = X.pos[index].requires_grad_(True)
            R = X.pos[index].requires_grad_(True).flatten()

            # if self.grad is None:
            #     # test = self.PES(R)
            #     self.grad = torch.autograd.functional.jacobian(self.PES,R,create_graph=True)
            
            # jac = self.grad(R)
            y.data[n,:] = torch.autograd.functional.jacobian(self.PES,R,create_graph=True)

        return y

    @staticmethod
    def get_pred(model:T,X:Data)->torch.Tensor:
        """return Energy, Polarization and Forces"""

        N = {"E":1, "EP":4, "EPF":1+3+3*X.Natoms[0]}
        N = N[model.output]
        batch_size = len(np.unique(X.batch))

        if model.output in ["E","EP"]:
            y = torch.zeros((batch_size,N))
            y = model(X)

        elif model.output == "EPF":
            y = torch.zeros((batch_size,N))
            EP = model(X)
            y[:,0]   = EP[:,0]         # 1st column  for the energy
            y[:,1:4] = EP[:,1:4]       # 3rd columns for the polarization
            y[:,4:]  = model.forces(X) # 3rd columns for the forces
        
        return y
    
    @staticmethod
    def get_real(X:Data,output:str="E")->torch.Tensor:
        """return Energy, Polarization and Forces"""

        # 'EPF' has to be modified in case we have different molecules in the dataset
        N = {"E":1, "EP":4, "EPF":1+3+3*X.Natoms[0]}
        N = N[output]
        batch_size = len(np.unique(X.batch))

        # if batch_size > 1 :          

        y = torch.zeros((batch_size,N))

        y[:,0] = X.energy

        if output in ["EP","EPF"]:
            y[:,1:4] = X.polarization.reshape((batch_size,-1))

        elif output == "EPF":
            y[:,4:]  = X.forces.reshape((batch_size,-1))

        # else:
        #     y = torch.zeros(N)
        #     y[0]   = X.energy

        #     if output in ["EP","EPF"]:
        #         y[1:4] = X.polarization.flatten()

        #     elif output == "EPF":
        #         y[4:]  = X.forces.flatten()

        return y