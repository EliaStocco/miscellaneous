import torch
from ase.io import read
from miscellaneous.elia.nn.dataset.make_dataset_delta import make_datapoint_delta
from miscellaneous.elia.nn.dataset import make_datapoint
from miscellaneous.elia.good_coding import froze
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
from abc import ABC, abstractproperty

#@froze
class iPIinterface():

    # __slots__ = ("normalization","_max_radius","_symbols")

    # @abstractproperty
    # def ref_dipole(self): 
    #     pass

    # @abstractproperty
    # def ref_pos(self): 
    #     pass

    # @abstractproperty
    # def reference(self): 
    #     pass

    def __init__(self,max_radius:float,normalization:dict=None,**kwargs):

        super().__init__(max_radius=max_radius,**kwargs)

        if normalization is None :
            normalization = {
                "energy":{
                    "mean" : 0.,
                    "std"  : 1.,
                },
                "dipole":{
                    "mean" : 0.,# torch.tensor([0.,0.,0.]),
                    "std"  : 1.,# torch.tensor([1.,1.,1.]),
                },
            }
        
        self.normalization = normalization
        # for k in self.normalization.keys():
        #     for j in self.normalization[k].keys():
        #         x = self.normalization[k][j]
        #         self.normalization[k][j] = torch.tensor(x)

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

        return y 
        
    def store_chemical_species(self,file,**argv):

        atoms = read(file,**argv)
        self._symbols = atoms.get_chemical_symbols()

        pass
    
    def _get(self,X,what:str,**argv)-> torch.tensor:
        """Get the correct value of the output restoring the original 'mean' and 'std' values.
        This should be used only during MD simulation."""

        # lower case
        what = what.lower()

        # compute output of the model
        if what == "energy" :
            y = self.energy(X,**argv)
        
        elif what == "forces":
            y = self.forces(X,**argv)
        
        elif what == "dipole":
            y = self.dipole(X,**argv)
        
        elif what == "bec":
            y = self.BEC(X,**argv)
        else :
            raise ValueError("quantity '{:s}' is not supported as output of this model".format(what))

        mean = None
        std = None
        if what == "energy" :
            mean, std = self.normalization["energy"]["mean"], self.normalization["energy"]["std"]
        
        elif what == "forces":
            std = self.normalization["energy"]["std"]
        
        elif what == "dipole":
            mean, std = self.normalization["dipole"]["mean"], self.normalization["dipole"]["std"]
        
        elif what == "bec":
            std = self.normalization["dipole"]["std"]

        # resize
        # batch_size = y.shape[0]
        # newdim = [1]*(len(y.shape)-2) # I remove the batch_size axis and the 'actual' value of the output
        # mean = mean.view(batch_size,3,*newdim) if mean is not None else mean
        # std  =  std.view(batch_size,3,*newdim) if  std is not None else std

        if what in ["energy","dipole"] :
            return y * std + mean
        
        elif what in ["forces","bec"]:
            return y * std
        
    def get(self,cell,pos,what:str,detach=True,**argv):

        requires_grad = {   "pos"        : True,\
                            "lattice"    : True,\
                            "x"          : False,\
                            "edge_vec"   : True,\
                            "edge_index" : False }

        X = self.make_datapoint(lattice=cell.T,positions=pos,requires_grad=requires_grad)

        y = self._get(what=what,X=X,**argv)

        if detach :
            y = y.detach()

        return  y
