import torch
from ase.io import read
# from miscellaneous.elia.nn.dataset.make_dataset_delta import make_datapoint_delta
from miscellaneous.elia.nn.dataset import make_datapoint
from .SabiaNetwork import SabiaNetwork
from typing import Dict
from torch_geometric.data import Data
from typing import TypeVar, Tuple
T = TypeVar('T', bound='iPIinterface')

class iPIinterface(SabiaNetwork):

    def __init__(self:T,max_radius:float,**kwargs):

        super().__init__(max_radius=max_radius,**kwargs)

        self._max_radius = max_radius
        self._symbols = None

        if self.output not in ["D","E"] :
            raise ValueError("not implemented yet")
        
        pass

    def make_datapoint(self,lattice, positions,**argv)->Data:

        other = { "lattice":lattice,
                  "positions":positions,
                  "symbols":self._symbols,
                  "max_radius":self._max_radius,
                  "default_dtype": self.default_dtype,
                  "pbc":self.pbc}

        # if self.reference :
        #     y = make_datapoint_delta(dipole=self.ref_dipole,\
        #                                 pos=self.ref_pos,\
        #                                 **other,\
        #                                 **argv)
        # else :
        #     y = make_datapoint(**other,**argv)

        return make_datapoint(**other,**argv) 
        
    def store_chemical_species(self,file,**argv):

        atoms = read(file,**argv)
        self._symbols = atoms.get_chemical_symbols()

        pass
    
    def correct_cell(self,cell=None):
        if not self.pbc and cell is None:
            cell = torch.eye(3).fill_diagonal_(torch.inf)
        return cell
    
    def get(self,pos,cell=None)->Tuple[torch.tensor,Data]: #,what:str="D"): #,detach=True,**argv):

        # 'cell' has to be in i-PI format

        cell = self.correct_cell(cell)

        self.eval()

        requires_grad = {   "pos"        : True,\
                            "lattice"    : True,\
                            "x"          : False,\
                            "edge_vec"   : True,\
                            "edge_index" : False }

        X = self.make_datapoint(lattice=cell.T,positions=pos,requires_grad=requires_grad)
        # del X.edge_vec

        return self(X)[0], X
    
    def get_jac(self,pos,cell=None,y=None,X=None)->Tuple[torch.tensor,torch.tensor]:

        if y is None or X is None:
            y,X = self.get(pos=pos,cell=cell)

        N = len(X.pos.flatten())
        jac = torch.full((N,y.shape[0]),torch.nan)

        for n in range(y.shape[0]):
            y[n].backward(retain_graph=True)
            jac[:,n] = X.pos.grad.flatten().detach()
            X.pos.grad.data.zero_()

        return jac,X

    def get_value_and_jac(self,pos,cell=None)->Tuple[torch.tensor,torch.tensor,Data]:
        y,X = self.get(pos,cell)
        jac,X = self.get_jac(pos,cell,y=y,X=X)
        return y.detach(),jac.detach(),X


