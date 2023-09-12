import torch
import numpy as np
from torch_geometric.data import Data
from scipy.stats import spearmanr
from torch.nn import MSELoss
from abc import ABC, abstractproperty, abstractmethod
from typing import TypeVar
A = TypeVar('A', bound='Methods4Training')
T = TypeVar('T', bound='EDFMethods4Training')

class Methods4Training(ABC):

    @abstractproperty
    def output(self:A): 
        pass

    @abstractmethod
    def get_pred(self:A,**argv):
        pass

    @abstractmethod
    def get_real(self:A,**argv):
        pass

    @abstractmethod
    def loss(self:A,**argv):
        pass

    @staticmethod
    @abstractmethod
    def correlation(**argv):
        pass

    @abstractmethod
    def forces(self:A,**argv):
        pass

class EDFMethods4Training(Methods4Training):

    # @staticmethod
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

    def loss(self:T,lE:float=None,lF:float=None,lP:float=None)->callable:

        lE = lE if lE is not None else 1.0
        lF = lF if lF is not None else 1.0
        lP = lP if lP is not None else 1.0

        if self.output in ["E","D"]:
            return MSELoss() #MSELoss(reduction='mean') # MSELoss(reduce='sum')
            #return lambda x,y: MSELoss()(x,y)
        
        elif self.output == "ED":
            def loss_EP(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                P = MSELoss()(x[:,1:4],y[:,1:4])
                return lE * E + lP * P
            return loss_EP
        
        elif self.output == "EF":
            def loss_EF(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                F = MSELoss()(x[:,1:],y[:,1:])
                return lE * E + lF * F
            return loss_EF
        
        elif self.output == "EDF":
            def loss_EPF(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                P = MSELoss()(x[:,1:4],y[:,1:4])
                F = MSELoss()(x[:,4:],y[:,4:])
                return lE * E + lP * P + lF * F
            return loss_EPF
        
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
            out[i] = spearmanr(x[:,i],y[:,i]).correlation

        return out
