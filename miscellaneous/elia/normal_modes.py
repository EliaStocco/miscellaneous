import numpy as np
from copy import copy
from miscellaneous.elia.functions import get_one_file_in_folder

class NormalModes():

    def __init__(self,Nmodes,Ndof=None):

        # Nmodes
        self.Nmodes = int(Nmodes)
        if Ndof is None:
            Ndof = Nmodes
        self.Ndof = int(Ndof)

        # Natoms
        self.Natoms = int(self.Ndof / 3)

        empty = np.full((self.Ndof,self.Nmodes),np.nan)
        self.ortho_modes = empty.copy()
        self.eigvec = empty.copy()
        self.dynmat = empty.copy()
        self.modes  = empty.copy()
        self.proj   = empty.copy()

        self.eigvals = np.full(self.Nmodes,np.nan)
        # self.freq    = np.full(self.Nmodes,np.nan)
        self.masses  = np.full(self.Ndof,np.nan)

        pass
    
    def __repr__(self) -> str:
        line = "" 
        line += "{:<10s}: {:<10d}\n".format("# modes",self.Nmodes)  
        line += "{:<10s}: {:<10d}\n".format("# dof",self.Ndof)  
        line += "{:<10s}: {:<10d}\n".format("# atoms",self.Natoms)  
        return line
    
    @classmethod
    def load(cls,folder=None):    

        file = get_one_file_in_folder(folder=folder,ext=".mode")
        tmp = np.loadtxt(file)

        self = cls(tmp.shape[0],tmp.shape[1])    

        # masses
        file = get_one_file_in_folder(folder=folder,ext=".masses")
        self.masses[:] = np.loadtxt(file)

        # ortho modes
        file = get_one_file_in_folder(folder=folder,ext=".mode")
        self.ortho_modes[:,:] = np.loadtxt(file)

        # eigvec
        file = get_one_file_in_folder(folder=folder,ext=".eigvec")
        self.eigvec[:,:] = np.loadtxt(file)

        # # hess
        # file = get_one_file_in_folder(folder=folder,ext="_full.hess")
        # self.hess = np.loadtxt(file)

        # eigvals
        file = get_one_file_in_folder(folder=folder,ext=".eigval")
        self.eigvals[:] = np.loadtxt(file)

        # dynmat 
        file = get_one_file_in_folder(folder=folder,ext=".dynmat")
        self.dynmat[:,:] = np.loadtxt(file)

        # modes
        # self.modes[:,:] = diag_matrix(self.masses,"-1/2") @ self.eigvec
        self.eigvec2modes()

        # proj
        # self.proj[:,:] = self.eigvec.T @ diag_matrix(self.masses,"1/2")
        self.eigvec2proj()

        return self   
    
    def set_dynmat(self,dynmat,mode="phonopy"):
        _dynmat = np.asarray(dynmat)
        if mode == "phonopy":
            # https://phonopy.github.io/phonopy/setting-tags.html
            # _dynmat = []
            N = _dynmat.shape[0]
            dynmat = np.full((N,N),np.nan,dtype=np.complex64)
            for n in range(N):
                row = np.reshape(_dynmat[n,:], (-1, 2))
                dynmat[n,:] = row[:, 0] + row[:, 1] * 1j
            self.dynmat = dynmat
        else:
            raise ValueError("not implemented yet")
        pass

    def set_eigvec(self,band,mode="phonopy"):
        if mode == "phonopy":
            N = self.Nmodes
            eigvec = np.full((N,N),np.nan,dtype=np.complex64)
            for n in range(N):
                f = band[n]["eigenvector"]
                f = np.asarray(f)
                f = f[:,:,0] + 1j * f[:,:,1]
                eigvec[:,n] = f.flatten()
            self.eigvec = eigvec
        else:
            raise ValueError("not implemented yet")
        pass

    # def set_eigvals(self,band,mode="phonopy"):
    #     if mode == "phonopy":
    #         N = self.Nmodes
    #         eigvals = np.full(N,np.nan)
    #         for n in range(N):
    #             eigvals[n] = band[n]["frequency"]
    #         self.eigvals = np.square(eigvals)
    #     else:
    #         raise ValueError("not implemented yet")
    #     pass

    @property
    def freq(self):
        return np.sqrt(np.abs(self.eigvals.real)) * np.sign(self.eigvals.real)
        
    def diagonalize(self,**argv):
        M = self.dynmat
        # if np.allclose(M, M.conj().T):
        #     eigvals, eigvecs, = np.linalg.eigh(M,**argv)
        # else:
        eigvals, eigvecs, = np.linalg.eigh(M,**argv)
        frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
        return frequencies, eigvals, eigvecs

    @staticmethod
    def diag_matrix(M,exp):
        out = np.eye(len(M))        
        if exp == "-1":
            np.fill_diagonal(out,1.0/M)
        elif exp == "1/2":
            np.fill_diagonal(out,np.sqrt(M))
        elif exp == "-1/2":
            np.fill_diagonal(out,1.0/np.sqrt(M))
        else :
            raise ValueError("'exp' value not allowed")
        return out           
    
    def eigvec2modes(self):
        self.modes = NormalModes.diag_matrix(self.masses,"-1/2") @ self.eigvec
        # self.ortho_modes[:,:] = self.modes / np.linalg.norm(self.modes,axis=0)

    def eigvec2proj(self):
        self.proj = self.eigvec.T @ NormalModes.diag_matrix(self.masses,"1/2")

    def project_displacement(self,displ):
        return self.proj @ displ

    def project_velocities(self,vel):
        return NormalModes.diag_matrix(self.eigvals,"-1/2") @ self.proj @ vel
    
    def build_supercell_normal_modes(self,size):

        from itertools import product
        import cmath

        values = [None]*len(size)
        for n,a in enumerate(size):
            values[n] = np.arange(a)
        r_point = list(product(*values))
        k_point = r_point.copy()

        size = np.asarray(size)
        N = size.prod()
        supercell = NormalModes(self.Nmodes*N,self.Ndof*N)
        supercell.masses[:] = np.asarray(list(self.masses)*N)
        supercell.eigvec.fill(np.nan)
        for i,r in enumerate(r_point):
            r = np.asarray(r) 
            for j,k in enumerate(k_point):
                kr = np.asarray(k) / size @ r
                phase = np.exp(1.j * 2 * np.pi * kr )
                # phi = int(cmath.phase(phase)*180/np.pi)
                # ic(k,r,phi)
                supercell.eigvec[i*self.Ndof:(i+1)*self.Ndof,j*self.Nmodes:(j+1)*self.Nmodes] = \
                    ( self.eigvec * phase).real
                
        if np.isnan(supercell.eigvec).sum() != 0:
            raise ValueError("error")
        
        supercell.eigvec /= np.linalg.norm(supercell.eigvec,axis=0)
        
        supercell.eigvec2modes()
        supercell.eigvec2proj()

        return supercell
    
    def remove_dof(self,dof):
        if not hasattr(dof,"__len__"):
            return self.remove_dof([dof])
        
        out = copy(self)

        ii = [x for x in np.arange(self.Ndof) if x not in dof]

        # out.ortho_modes = empty.copy()
        out.eigvec = self.eigvec[:,ii]
        out.dynmat = np.nan
        # out.modes = empty.copy()
        # out.proj = empty.copy()
        out.eigvals = self.eigvals[ii]

        out.Nmodes = out.eigvec.shape[1]

        out.eigvec2modes()
        out.eigvec2proj()
        
        return out