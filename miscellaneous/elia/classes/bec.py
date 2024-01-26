import xarray as xr
import numpy as np
from .io import pickleIO

class bec(xr.DataArray,pickleIO):
    __slots__ = ('_data', '_dtype', '_file', '_other_attribute')  # Add any additional attributes you may have

    @classmethod
    def from_extxyz(cls,file):
        pass

    @classmethod
    def from_txt(cls,file):
        pass

    @classmethod
    def from_numpy(cls,array:np.ndarray):
        """Create a 'bec' object from a numpy ndarray."""

        array = np.atleast_3d(array)

        if len(array.shape) != 3:
            raise ValueError("only 3D array are supported")
        
        Nstruc = array.shape[0]
        if array.shape[2] == 9:
            Natoms = array.shape[1]
            empty = np.full((Nstruc,3*Natoms,3),np.nan)
            for s in range(empty.shape[0]): # cycle over the atomic structures
                empty[s,:,:] = array[s,:,:].reshape((3*Natoms,3))
            array = empty.copy()
        elif array.shape[2] == 3:
            pass
        else:
            raise ValueError("wrong number of columns")
        
        obj = xr.DataArray(array.copy(), dims=('structure', 'dof', 'dir'))
        return cls(obj)
