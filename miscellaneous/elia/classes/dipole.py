from .io import pickleIO
from dataclasses import dataclass, field
import numpy as np
from ase import Atoms

@dataclass
class dipoleLM(pickleIO):

    ref: Atoms
    bec: np.ndarray
    dipole: np.ndarray
    Natoms: int = field(init=False)  # Natoms is set in __post_init__
    frame: str = field(default="global")

    def __post_init__(self):
        self.Natoms = self.ref.get_global_number_of_atoms()  # Set Natoms based on the shape of self.ref

        if self.bec.shape[0] != 3 * self.Natoms:
            raise ValueError(f"Invalid shape[0] for 'bec'. Expected {3 * self.Natoms}, got {self.bec.shape}")
        if self.bec.shape[1] != 3:
            raise ValueError(f"Invalid shape[1] for 'bec'. Expected 3, got {self.bec.shape}")

        if self.dipole.shape != (3,):
            raise ValueError(f"Invalid shape for 'dipole'. Expected (3,), got {self.dipole.shape}")

        if self.frame not in ["global", "eckart"]:
            raise ValueError(f"Invalid value for 'frame'. Expected 'global' or 'eckart', got {self.frame}")
        
    def get(self,pos:np.ndarray):
        """Compute the dipole according to a linear model in the cartesian displacements."""
        if pos.ndim != 3:
            pos = np.atleast_3d(pos)
            pos = np.moveaxis(pos, -1, 0)
        if pos[0,:,:].shape != self.ref.positions.shape:
            raise ValueError(f"Invalid shape for 'pos[0]'. Expected {self.ref.positions.shape}, got {pos[0,:,:].shape}")
        out, _, _ = self._evaluate(pos)
        return out
        
    def _evaluate(self,pos:np.ndarray):
        match self.frame:
            case "eckart" :
                raise NotImplementedError("'eckart' frame not implemented yet.")
                from copy import copy
                newx, com, rotmat  = self.eckart(index)            
                # save old positions
                oldpos = copy(self.positions)
                # set the rotated positions
                self.positions = copy(newx.reshape((len(newx),-1)))
                # compute the model in the Eckart frame
                model, _, _ = self.dipole_model(index,frame="global")
                # re-set the positions to the original values
                self.positions = oldpos
                # return the model

                # 'rotmat' is supposed to be right-multiplied:
                # vrot = v @ rotmat
                return model, com, rotmat 

            case "global" :
                N = len(pos)
                model  = np.full((N,3),np.nan)
                for n in range(N):
                    R = pos[n]#.reshape((-1,3))
                    dD = self.bec.T @ np.asarray(R - self.ref.positions).reshape(3*self.Natoms)
                    model[n,:] = dD + self.dipole
                return model, None, None
        
            case _ :
                raise ValueError("'frame' can be only 'eckart' or 'global' (dafault).")

