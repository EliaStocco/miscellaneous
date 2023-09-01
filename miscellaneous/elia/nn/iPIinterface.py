import torch
from ase.io import read
from miscellaneous.elia.nn.get_type_onehot_encoding import symbols2x
# from miscellaneous.elia.nn.make_dataset import my_neighbor_list
from miscellaneous.elia.nn.make_dataset_delta import make_datapoint_delta
from miscellaneous.elia.nn.make_dataset import make_datapoint
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
from abc import ABC, abstractproperty, abstractmethod

class iPIinterface(ABC):

    # @abstractproperty
    # def ref_dipole(self): pass

    # @abstractproperty
    # def ref_pos(self): pass

    # @abstractproperty
    # def reference(self): pass

    # @abstractmethod
    # def _get(self,**argv): pass


    def __init__(self,max_radius:float,normalization:dict=None,**kwargs):

        super().__init__(max_radius=max_radius,**kwargs)

        if normalization is None :
            normalization = {
                "energy":{
                    "mean" : 0,
                    "std"  : 1,
                },
                "dipole":{
                    "mean" : torch.tensor([0.,0.,0.]),
                    "std"  : torch.tensor([1.,1.,1.]),
                },
            }
        
        self.normalization = normalization
        for k in self.normalization.keys():
            for j in self.normalization[k].keys():
                x = self.normalization[k][j]
                self.normalization[k][j] = torch.tensor(x)

        #self._X = None
        self._max_radius = max_radius
        self._symbols = None

        pass

    def make_datapoint(self,lattice, positions,**argv):

        other = { "lattice":lattice,\
                  "positions":positions,\
                  "symbols":self._symbols,\
                  "max_radius":self._max_radius}

        if self.reference :
            y = make_datapoint_delta(dipole=self.ref_dipole,\
                                        pos=self.ref_pos,\
                                        **other,\
                                        **argv)
        else :
            y = make_datapoint(**other,**argv)

        # y.batch = torch.full((len(positions),),0)

        return y 
        
    def store_chemical_species(self,file,**argv):

        atoms = read(file,**argv)
        self._symbols = atoms.get_chemical_symbols()
        #self._symbols = symbols2x(symbols)

        # self._X = Data(
        #     x=x,                                   # fixed
        #     pos     = torch.full((),torch.nan),    # updated in 'get'
        #     lattice = torch.full((),torch.nan),    # updated in 'get'              
        #     max_radius = self.max_radius,          # fixed
        #     symbols    = symbols,                  # fixed
        #     edge_index = torch.full((),torch.nan), # computed in 'get'
        #     edge_vec   = torch.full((),torch.nan), # computed in 'get'
        #     edge_shift = torch.full((),torch.nan), # computed in 'get'
        # )

        pass

    # def _get(self,X,what:str,**argv)-> torch.tensor:
    #     """To be overwritten"""

    #     raise ValueError("This method has to be overwritten by the child class.")
        
    def get(self,cell,pos,what:str):

        #cell = torch.from_numpy(cell.T).to(torch.float64)

        requires_grad = {   "pos"        : True,\
                            "lattice"    : True,\
                            "x"          : False,\
                            "edge_vec"   : True,\
                            "edge_index" : False }

        X = self.make_datapoint(lattice=cell.T,positions=pos,requires_grad=requires_grad)

        return self._get(what=what,X=X) # .detach()
        
    # def get(self,cell, pos,what:str):
    #     """Mask to '_get' for i-PI.
    #     You need to have called (once) 'store_chemical_species' before calling this method"""

    #     if self._X is None :
    #         raise ValueError("Chemical species unknown: you need to call store_chemical_species' before calling 'get'")
        
    #     # I should check that the format of 'cell' is the same in i-PI and e3nn
    #     # e3nn uses ase.Atoms.cell.array
    #     lattice = torch.from_numpy(cell.T).to(torch.float64)
    #     pos = torch.from_numpy(pos).to(torch.long)

    #     edge_src, edge_dst, edge_shift = my_neighbor_list(lattice,pos,self.max_radius)
    #     edge_shift = torch.from_numpy(edge_shift).to(torch.float64)

    #     lattice = lattice.unsqueeze(0)

    #     batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
    #     edge_batch = batch[edge_src]
    #     edge_vec = (pos[edge_dst]
    #                 - pos[edge_src]
    #                 + torch.einsum('ni,nij->nj',
    #                             edge_shift,
    #                             lattice[edge_batch]))
        
    #     edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)

    #     self._X.pos     = pos
    #     self._X.lattice = lattice
    #     self._X.edge_index = edge_index
    #     self._X.edge_vec   = edge_vec
    #     self._X.edge_shift = edge_shift

    #     X = DataLoader([self._X],batch_size=1,shuffle=False,drop_last=False)

    #     X = next(iter(X))

    #     y = self._get(what=what,X=X)

    #     return y

