import torch
from .iPIinterface import iPIinterface
# from .Methods4Training import EDFMethods4Training
from torch_geometric.data import Data
from torch.nn import MSELoss
from scipy.stats import pearsonr
import numpy as np
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetworkManager')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

__all__ = ["SabiaNetworkManager"]

class SabiaNetworkManager(iPIinterface):

    def __init__(self: T, 
                 output: str = "ED",
                 **kwargs) -> None:

        self.output = output
        if self.output not in ["E", "D", "ED", "EF", "EDF"]:
            raise ValueError("'output' must be 'E', 'D', 'EF', 'ED' or 'EDF'")

        super().__init__(**kwargs)
        
        pass

    def n_parameters(self:T):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def batch(X):
        return len(torch.unique(X.batch))
    
    def message(self):
        if self.use_shift:
                print("\t!! SHIFT:", self.shift.detach().numpy())
                print("\t!! FACTOR:", self.factor.detach().numpy())
        return
    
    def get_pred(self: T, X: Data) -> torch.tensor:
        """return Energy, Polarization and Forces"""

        N = {"E": 1, "D": 3, "ED": 4, "EF": 1+3*X.Natoms[0], "EDF": 1+3+3*X.Natoms[0]}
        N = N[self.output]
        batch_size = len(np.unique(X.batch))
        y = torch.zeros((batch_size, N))

        if self.output in ["E", "ED","D"]:
            y = self(X)

        elif self.output == "EDF":
            EP = self(X)
            y[:, 0] = EP[:, 0]         # 1st column for the energy
            y[:, 1:4] = EP[:, 1:4]     # 2nd to 4th columns for the dipole
            y[:, 4:] = self.forces(X)  # other columns for the forces

        elif self.output == "EF":
            EP = self(X)
            y[:, 0]  = EP[:, 0]         # 1st column for the energy
            y[:, 1:] = self.forces(X)   # other columns for the forces

        return y

    # @staticmethod
    def get_real(self:T,X: Data) -> torch.tensor:
        """return Energy, Polarization and/or Forces"""

        # 'EPF' has to be modified in case we have different molecules in the dataset
        N = {"E": 1, "EF": 1+3*X.Natoms[0], "D": 3, "ED": 4, "EDF": 1+3+3*X.Natoms[0]}
        N = N[self.output]
        batch_size = len(np.unique(X.batch))

        # if batch_size > 1 :

        y = torch.zeros((batch_size, N))

        if self.output in ["E", "EF", "ED", "EDF"]:
            y[:, 0] = X.energy

            if self.output in ["ED", "EDF"]:
                y[:, 1:4] = X.dipole.reshape((batch_size, -1))

            elif self.output == "EDF":
                y[:, 4:] = X.forces.reshape((batch_size, -1))

            if self.output == "EF":
                y[:, 1:] = X.forces.reshape((batch_size, -1))

        elif self.output == "D":
            y[:,0:3] = X.dipole.reshape((batch_size, -1))

        return y

    def loss(self:T,lE:float=None,lF:float=None,lP:float=None,Natoms=None,periodic=False,regularization=0.1)->callable:

        lE = lE if lE is not None else 1.0
        lF = lF if lF is not None else 1.0
        lP = lP if lP is not None else 1.0

        self._mseloss = MSELoss() # Mean Squared Error -> it's not the RMSE!

        if Natoms is None or Natoms <= 0 :
            Natoms = 1

        def add_Natoms(func:callable):
            if Natoms > 1 :
                def divide(x:torch.tensor,y:torch.tensor)->torch.Tensor:
                    return func(x,y) / Natoms
                return divide
            else :
                return func

        @add_Natoms
        def loss_scalar(x:torch.tensor,y:torch.tensor)->torch.Tensor:
            """Loss function for scalar quantity"""
            return self._mseloss(x,y)
        
        @add_Natoms
        def loss_vector(x:torch.tensor,y:torch.tensor)->torch.Tensor:
            return torch.mean(torch.square(x-y).sum(dim=1)) # mean only along the batch size

        if self.output == "E":
            # if Natoms > 1 :
            #     return lambda x,y: loss_scalar(x,y) / Natoms
            # else :
            #     return loss_scalar
            return loss_scalar
        
        elif self.output == "D" :
            if periodic :
                def periodic_loss(x:torch.tensor,y:torch.tensor,\
                                  X:Union[torch_geometric.data.Data, Dict[str, torch.Tensor]])->torch.Tensor:
                    # do things
                    return loss_vector(x,y)
                return periodic_loss
            else :
                return loss_vector
        
        elif self.output == "EF":
            def loss_EF(x,y):
                E = loss_scalar(x[:,0],y[:,0])
                F = loss_vector(x[:,1:],y[:,1:])
                return lE * E + lF * F
            if Natoms > 1 :
                return lambda x,y: loss_EF(x,y) / Natoms
            else :
                return loss_EF 
        
        # elif self.output == "EDF":
        #     def loss_EPF(x,y):
        #         E = self._mseloss(x[:,0],y[:,0])
        #         P = self._mseloss(x[:,1:4],y[:,1:4])
        #         F = self._mseloss(x[:,4:],y[:,4:])
        #         return lE * E + lP * P + lF * F
        #     return loss_EPF
        
        else :
            raise ValueError("error in output mode")
        
    @staticmethod
    def correlation(x:torch.tensor,y:torch.tensor):

        #N = {"E": 1, "P": 3, "EP": 4, "EPF": 1+3+3*X.Natoms[0]}

        x = x.detach().numpy()
        y = y.detach().numpy()

        N = x.shape[1]
        out = np.zeros(N)
        for i in range(N):
            out[i] = pearsonr(x[:,i],y[:,i]).correlation

        return out
        