import numpy as np
from copy import copy
from itertools import product
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
        self.eigvec = empty.copy()
        self.dynmat = empty.copy()
        self.mode   = empty.copy()
        self.proj   = empty.copy()
        self.non_ortho_modes = empty.copy()

        self.eigval = np.full(self.Nmodes,np.nan)
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

        # ortho mode
        file = get_one_file_in_folder(folder=folder,ext=".mode")
        self.mode[:,:] = np.loadtxt(file)

        # eigvec
        file = get_one_file_in_folder(folder=folder,ext=".eigvec")
        self.eigvec[:,:] = np.loadtxt(file)

        # # hess
        # file = get_one_file_in_folder(folder=folder,ext="_full.hess")
        # self.hess = np.loadtxt(file)

        # eigval
        file = get_one_file_in_folder(folder=folder,ext=".eigval")
        self.eigval[:] = np.loadtxt(file)

        # dynmat 
        file = get_one_file_in_folder(folder=folder,ext=".dynmat")
        self.dynmat[:,:] = np.loadtxt(file)

        # mode
        # self.mode[:,:] = diag_matrix(self.masses,"-1/2") @ self.eigvec
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
    #         eigval = np.full(N,np.nan)
    #         for n in range(N):
    #             eigval[n] = band[n]["frequency"]
    #         self.eigval = np.square(eigval)
    #     else:
    #         raise ValueError("not implemented yet")
    #     pass

    @property
    def freq(self):
        return np.sqrt(np.abs(self.eigval.real)) * np.sign(self.eigval.real)
        
    def diagonalize(self,**argv):
        M = self.dynmat
        # if np.allclose(M, M.conj().T):
        #     eigval, eigvecs, = np.linalg.eigh(M,**argv)
        # else:
        eigval, eigvecs, = np.linalg.eigh(M,**argv)
        frequencies = np.sqrt(np.abs(eigval.real)) * np.sign(eigval.real)
        return frequencies, eigval, eigvecs

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
        self.non_ortho_mode = NormalModes.diag_matrix(self.masses,"-1/2") @ self.eigvec
        self.mode = self.non_ortho_mode / np.linalg.norm(self.non_ortho_mode,axis=0)

    def eigvec2proj(self):
        self.proj = self.eigvec.T @ NormalModes.diag_matrix(self.masses,"1/2")

    def project_displacement(self,displ):
        return self.proj @ displ

    def project_velocities(self,vel):
        return NormalModes.diag_matrix(self.eigval,"-1/2") @ self.proj @ vel
    
    def build_supercell_displacement(self,size,q):

        q = np.asarray(q)

        values = [None]*len(size)
        for n,a in enumerate(size):
            values[n] = np.arange(a)
        r_point = list(product(*values))
        
        size = np.asarray(size)
        N = size.prod()
        supercell = NormalModes(self.Nmodes,self.Ndof*N)
        supercell.masses[:] = np.asarray(list(self.masses)*N)
        supercell.eigvec.fill(np.nan)
        for i,r in enumerate(r_point):
            kr = np.asarray(r) / size @ q
            phase = np.exp(1.j * 2 * np.pi * kr )
            # phi = int(cmath.phase(phase)*180/np.pi)
            # ic(k,r,phi)
            supercell.eigvec[i*self.Ndof:(i+1)*self.Ndof,:] = ( self.eigvec * phase).real
                
        if np.isnan(supercell.eigvec).sum() != 0:
            raise ValueError("error")
        
        supercell.eigvec /= np.linalg.norm(supercell.eigvec,axis=0)
        supercell.eigval = self.eigval.copy()
        
        supercell.eigvec2modes()
        supercell.eigvec2proj()

        return supercell
    
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
        # out.mode = empty.copy()
        # out.proj = empty.copy()
        out.eigval = self.eigval[ii]

        out.Nmodes = out.eigvec.shape[1]

        out.eigvec2modes()
        out.eigvec2proj()
        
        return out