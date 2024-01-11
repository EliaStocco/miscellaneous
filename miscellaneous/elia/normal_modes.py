import numpy as np
from copy import copy
from itertools import product
import xarray as xr
from miscellaneous.elia.functions import get_one_file_in_folder, nparray2list_in_dict
from warnings import warn
from miscellaneous.elia.units import *
import pickle
import warnings
# Disable all UserWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def dot(A:xr.DataArray,B:xr.DataArray,dim:str):
    _A,ua = remove_unit(A)
    _B,ub = remove_unit(B)
    out = _A.dot(_B,dim=dim)
    return set_unit(out,ua*ub)

def norm_by(array,dim):
    tmp = np.linalg.norm(array.data,axis=array.dims.index(dim))
    tmp, _ = remove_unit(tmp)
    return tmp * get_unit(array)

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

class NormalModes():

    def __init__(self,Nmodes,Ndof=None,ref=None):

        # Nmodes
        self.Nmodes = int(Nmodes)
        if Ndof is None:
            Ndof = Nmodes
        self.Ndof = int(Ndof)

        # Natoms
        self.Natoms = int(self.Ndof / 3)

        self.dynmat = xr.DataArray(np.full((self.Ndof,self.Ndof),np.nan), dims=('dof-a', 'dof-b'))

        empty = xr.DataArray(np.full((self.Ndof,self.Nmodes),np.nan,dtype=np.complex128), dims=('dof', 'mode'))
        self.eigvec = empty.copy()
        self.mode   = empty.copy()
        # self.non_ortho_modes = empty.copy()

        self.eigval = xr.DataArray(np.full(self.Nmodes,np.nan), dims=('mode')) 
        self.masses = xr.DataArray(np.full(self.Ndof,np.nan), dims=('dof'))

        self.reference = ref

        pass
    
    def __repr__(self) -> str:
        line = "" 
        line += "{:<10s}: {:<10d}\n".format("# modes",self.Nmodes)  
        line += "{:<10s}: {:<10d}\n".format("# dof",self.Ndof)  
        line += "{:<10s}: {:<10d}\n".format("# atoms",self.Natoms)  
        return line
    
    def to_dict(self)->dict:
        return nparray2list_in_dict(vars(self))

    def write(self,file,module="pickle",mode="w"):
        match module:
            case "pickle":
                import pickle
                with open(file, mode) as f:
                    pickle.dump(self, f)
            case "pprint":
                from pprint import pprint
                with open(file, mode) as f:
                    pprint(self, f)
            case "yaml":
                import yaml
                with open(file, mode) as f:
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
            case "json":
                import json
                with open(file, mode) as f:
                    json.dump(self.to_dict(), f)
            case _:
                raise ValueError("saving with module '{:s}' not implemented yet".format(module))
    
    @classmethod
    def read(cls,file,module="pickle",mode="r"):
        match module:
            case "pickle":
                import pickle
                with open(file,'rb') as f:
                    loaded_data = pickle.load(f)
            case "yaml":
                import yaml
                with open(file,mode) as f:
                    loaded_data = yaml.load(f)
            case "json":
                import json
                with open(file,mode) as f:
                    loaded_data = json.load(f)
        return cls(loaded_data)

    def to_pickle(self,file):
        # Open the file in binary write mode ('wb')
        with open(file, 'wb') as file:
            # Use pickle.dump() to serialize and save the object to the file
            pickle.dump(self, file)

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
        # self.eigvec2proj()

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
            self.dynmat = xr.DataArray(dynmat, dims=('dof-a', 'dof-b'))
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
            self.eigvec[:,:] = xr.DataArray(eigvec, dims=('dof', 'mode'))
        else:
            raise ValueError("not implemented yet")
        pass

    def set_eigval(self,eigval):
        self.eigval[:] = xr.DataArray(eigval, dims=('mode'))


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

    # @property
    # def freq(self):
    #     return np.sqrt(np.abs(self.eigval.real)) * np.sign(self.eigval.real)
        
    # def diagonalize(self,**argv):
    #     eigval, eigvecs, = np.linalg.eigh(self.dynmat,**argv)
    #     frequencies = np.sqrt(np.abs(eigval.real)) * np.sign(eigval.real)
    #     return frequencies, eigval, eigvecs
        
    
    def eigvec2modes(self):
        self.non_ortho_mode = self.eigvec.copy()
        for i in range(self.non_ortho_mode.sizes['dof']):
            index = {'dof': i}
            self.non_ortho_mode[index] = self.eigvec[index] * np.sqrt(self.masses[index])
        self.mode = self.non_ortho_mode / norm_by(self.non_ortho_mode,"dof")
        pass

    # def eigvec2proj(self):
    #     self.proj = self.eigvec.T @ NormalModes.diag_matrix(self.masses,"1/2")

    # def project_displacement(self,displ):
    #     return self.proj @ displ

    # def project_velocities(self,vel):
    #     return NormalModes.diag_matrix(self.eigval,"-1/2") @ self.proj @ vel
    
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
        # supercell.eigvec.fill(np.nan)
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
        # supercell.eigvec2proj()

        return supercell
    
    # def build_supercell_normal_modes(self,size):

    #     from itertools import product
    #     import cmath

    #     values = [None]*len(size)
    #     for n,a in enumerate(size):
    #         values[n] = np.arange(a)
    #     r_point = list(product(*values))
    #     k_point = r_point.copy()

    #     size = np.asarray(size)
    #     N = size.prod()
    #     supercell = NormalModes(self.Nmodes*N,self.Ndof*N)
    #     supercell.masses[:] = np.asarray(list(self.masses)*N)
    #     supercell.eigvec.fill(np.nan)
    #     for i,r in enumerate(r_point):
    #         r = np.asarray(r) 
    #         for j,k in enumerate(k_point):
    #             kr = np.asarray(k) / size @ r
    #             phase = np.exp(1.j * 2 * np.pi * kr )
    #             # phi = int(cmath.phase(phase)*180/np.pi)
    #             # ic(k,r,phi)
    #             supercell.eigvec[i*self.Ndof:(i+1)*self.Ndof,j*self.Nmodes:(j+1)*self.Nmodes] = \
    #                 ( self.eigvec * phase).real
                
    #     if np.isnan(supercell.eigvec).sum() != 0:
    #         raise ValueError("error")
        
    #     supercell.eigvec /= np.linalg.norm(supercell.eigvec,axis=0)
        
    #     supercell.eigvec2modes()
    #     supercell.eigvec2proj()

    #     return supercell
    
    # def remove_dof(self,dof):
    #     if not hasattr(dof,"__len__"):
    #         return self.remove_dof([dof])
        
    #     out = copy(self)

    #     ii = [x for x in np.arange(self.Ndof) if x not in dof]

    #     # out.ortho_modes = empty.copy()
    #     out.eigvec = self.eigvec[:,ii]
    #     out.dynmat = np.nan
    #     # out.mode = empty.copy()
    #     # out.proj = empty.copy()
    #     out.eigval = self.eigval[ii]

    #     out.Nmodes = out.eigvec.shape[1]

    #     out.eigvec2modes()
    #     # out.eigvec2proj()
        
    #     return out
    
    def project(self,trajectory,warning="**Warning**"):       

        #-------------------#
        # reference position
        ref = trajectory[0] if self.reference is None else self.reference

        #-------------------#
        # positions -> displacements
        q = trajectory.positions - ref.positions
        q = q.reshape(len(q),-1)
        q *= atomic_unit["length"]

        #-------------------#
        # velocities
        try :
            v = trajectory.call(lambda e: e.arrays["velocities"])
            v = v.reshape(len(v),-1)
        except:
            warn("velocities not found, setting them to zero.")
            v = np.zeros(q.shape)

        v *= atomic_unit["velocity"]

        #-------------------#
        # building xarrays
        q = xr.DataArray(q, dims=('time','dof')) 
        v = xr.DataArray(v, dims=('time','dof')) 

        #-------------------#
        # eigvec
        # Rename the 'time' dimension to 'new_time' and 'space' dimension to 'new_space'
        eigvec = self.eigvec.copy() * atomic_unit["dimensionless"]
        A = self.eigvec.rename({'mode': 'mode-a', 'dof': 'dof'})
        B = self.eigvec.rename({'mode': 'mode-b', 'dof': 'dof'})
        test = A.dot(B,dim="dof")
        if test.shape != (self.Nmodes,self.Nmodes):
            raise ValueError("wrong shape")
        if np.square(test - np.eye(self.Nmodes)).sum() > 1e-8:
            raise ValueError("eigvec is not orthogonal")
        
        #-------------------#
        # masses
        M = xr.DataArray(self.masses,dims=("dof")) * atomic_unit["mass"]
        Msqrt = np.sqrt(M)

        #-------------------#
        # proj
        proj = Msqrt * eigvec.T
        mode = proj / norm_by(proj,"dof")
        # mode,_ = set_unit(mode,atomic_unit["dimensionless"])
        if not np.allclose(mode.data.magnitude,self.mode.data):
            raise ValueError("conflict between 'eigvec' and 'mode'")
        
        #-------------------#
        # proj should be real
        if np.any(proj.imag != 0.0):
            warn("'proj' matrix should be real --> discarding its imaginary part.")
        proj = proj.real


        # #-------------------#
        # # Normal Modes should be real
        # save = self.mode.copy()
        # if np.any(self.mode.imag != 0.0):
        #     warn("'mode' matrix should be real --> discarding its imaginary part.")
        # # do it anyway
        # self.mode = self.mode.real

        #-------------------#
        # create the projection operator onto the normal modes
        # proj = self.mode.T * atomic_unit["dimensionless"]

        #-------------------#
        # simple test
        if not check_dim(q,'[length]'):
            raise ValueError("displacements have the wrong unit")
        if not check_dim(v,'[length]/[time]'):
            raise ValueError("velocities have the wrong unit")
        if not check_dim(proj,'[mass]**0.5'):
            raise ValueError("projection operator has the wrong unit")

        #-------------------#
        # project positions and velocities
        # pint is not compatible with np.tensordot
        # we need to remove the unit and set them again
        # q,uq    = remove_unit(q)
        # v,uv    = remove_unit(v)
        # proj,up = remove_unit(proj)

        # qn = proj.dot(q,dim="dof")
        # vn = proj.dot(v,dim="dof")

        # qn   = set_unit(qn,uq*up)
        # vn   = set_unit(vn,uv*up)
        # proj = set_unit(vn,up)

        qn = dot(proj,q,"dof")
        vn = dot(proj,v,"dof")

        # #-------------------#
        # # masses
        # m = proj.dot(self.masses,dim="dof").dot(proj.T,dim="dof")
        # m = set_unit(m,atomic_unit["mass"])

        #-------------------#
        # vib. modes eigenvalues
        w2 = xr.DataArray(self.eigval, dims=('mode')) 
        w2 = set_unit(w2,atomic_unit["frequency"]**2)
        
        #-------------------#
        # energy: kinetic, potential and total
        #
        # H = 1/2 M V^2 + 1/2 M W^2 X^2
        #   = 1/2 M V^2 + 1/2 K     X^2
        #   = 1/2 M ( V^2 + W^2 X^2 )
        #
        # K = 0.5 * m * vn*vn.conjugate()      # kinetic
        # U = 0.5 * m * w2 * qn*qn.conjugate() # potential
        K = 0.5 * np.square(vn) # vn*vn.conjugate()      # kinetic
        U = 0.5 * w2 * np.square(qn) # qn*qn.conjugate() # potential
        if not check_dim(K,'[energy]'):
            raise ValueError("the kinetic energy has the wrong unit: ",get_unit(K))
        if not check_dim(U,'[energy]'):
            raise ValueError("the potential energy has the wrong unit: ",get_unit(U))

        if np.any( U < 0 ):
            print("\t{:s}: negative potential energies!".format(warning),end="\n\t")
        if np.any( K < 0 ):
            print("\t*{:s}:negative kinetic energies!".format(warning),end="\n\t")
        
        energy = U + K
        if not check_dim(energy,'[energy]'):
            raise ValueError("'energy' has the wrong unit")
        else:
            energy = set_unit(energy,atomic_unit["energy"])
            if not check_dim(energy,'[energy]'):
                raise ValueError("'energy' has the wrong unit")
            
        # if np.any( energy < 0 ):
        #     raise ValueError("negative energies!")
            
        #-------------------#
        # amplitudes of the vib. modes
        vv = 1/np.sqrt(w2) * vn
        A2 = ( np.square(qn) + np.square(vv) )
        A  = np.sqrt(A2)
        amplitude  = dot(dot(mode,1./np.sqrt(M) ,"mode"),eigvec,"dof")* A
        if not check_dim(amplitude,'[length]'):
            raise ValueError("'amplitude' have the wrong unit")
        
        # amplitude_ = np.sqrt( energy / ( 0.5 * M * w2 ) )
        # if not np.allclose(amplitude,amplitude_):
        #     raise ValueError("inconsistent value")

        #-------------------#
        # check how much the trajectory 'satisfies' the equipartition theorem
        equipartition = energy.shape[energy.dims.index("mode")] * energy / energy.sum("time")
        if not check_dim(equipartition,'[]'):
            raise ValueError("'equipartition' has the wrong unit")
        
        #-------------------#
        # occupations (in units of hbar)
        occupation = energy / np.sqrt(w2)
        occupation /= atomic_unit["action"]
        if not check_dim(occupation,'[]'):
            raise ValueError("'occupation' has the wrong unit")
        
        #-------------------#
        w = np.sqrt(w2)
        if not check_dim(w,"1/[time]"):
            raise ValueError("'w' has the wrong unit")
        
        #-------------------#
        # phases (angles variables of the harmonic oscillators, i.e. the vib. modes)
        phases = np.arctan2(-vn, w*qn)
        if not check_dim(phases,"[]"):
            raise ValueError("'phases' has the wrong unit")

        #-------------------#
        # output
        out = {
            "energy"        : energy,
            "kinetic"       : K,
            "potential"     : U,
            "amplitude"     : amplitude,
            "equipartition" : equipartition,
            "occupation"    : occupation,
            "phases"        : phases
        }

        # self.mode = save

        return out

    # def project(self, trajectory):

    #     import unyt
    #     # Define the Hartree unit
    #     unyt.define_unit('hartree', 4.359744650e-18 * unyt.joule)

    #     # Define the Bohr unit
    #     unyt.define_unit('bohr', 5.291772108e-11 * unyt.meter)

    #     # -------------------#
    #     use_unyt = True

    #     # -------------------#
    #     # Create a UnitRegistry with atomic units
    #     unit_registry = unyt.UnitRegistry()

    #     # Define atomic units based on fundamental constants
    #     unit = {
    #         "energy": unyt.hartree if use_unyt else 1,
    #         "length": unyt.bohr if use_unyt else 1,
    #         "mass": unyt.me if use_unyt else 1,
    #         "action": unyt.hbar if use_unyt else 1,
    #         "dimensionless": unyt.dimensionless if use_unyt else 1
    #     }
    #     unit["time"] = unit["action"] / unit["energy"]
    #     unit["velocity"] = unit["length"] / unit["time"]

    #     # -------------------#
    #     # reference position
    #     ref = trajectory[0] if self.reference is None else self.reference

    #     # -------------------#
    #     # positions -> displacements
    #     q = trajectory.positions - ref.positions
    #     q = q.reshape(len(q), -1)
    #     q *= unit["length"]

    #     # -------------------#
    #     # velocities
    #     try:
    #         v = trajectory.call(lambda e: e.arrays["velocities"])
    #         v = v.reshape(len(v), -1)
    #     except:
    #         warn("velocities not found, setting them to zero.")
    #         v = np.zeros(q.shape)

    #     v *= unit["velocity"]

    #     # -------------------#
    #     # building xarrays
    #     q = xr.DataArray(q, dims=('time', 'dof'))
    #     v = xr.DataArray(v, dims=('time', 'dof'))

    #     # -------------------#
    #     # Normal Modes should be real
    #     save = self.mode.copy()
    #     if np.any(self.mode.imag != 0.0):
    #         warn("'mode' matrix should be real --> discarding its imaginary part.")
    #     # do it anyway
    #     self.mode = self.mode.real

    #     # -------------------#
    #     # create projection operator
    #     proj = self.mode.T * unit["dimensionless"]

    #     # -------------------#
    #     # simple test
    #     if use_unyt:
    #         if not q.units.equal_dimensions(unit_registry.length):
    #             raise ValueError("displacements have the wrong unit")
    #         if not v.units.equal_dimensions(unit_registry.length / unit_registry.time):
    #             raise ValueError("velocities have the wrong unit")
    #         if not proj.units.equal_dimensions(unit_registry.dimensionless):
    #             raise ValueError("projection operator has the wrong unit")

    #     # -------------------#
    #     # project positions and velocities
    #     qn = proj.dot(q.magnitude, dim="dof")
    #     vn = proj.dot(v.magnitude, dim="dof")

    #     # -------------------#
    #     # compute
    #     m = self.mode.T.dot(self.masses, dim="dof").dot(self.mode, dim="dof")
    #     w2 = xr.DataArray(self.eigval, dims=('mode'))

    #     # A2 = qn*qn.conjugate() + vn*vn.conjugate()
    #     # energy = 0.5 * m * (w2 * A2)
    #     energy = 0.5 * m * (vn * vn.conjugate() + w2 * qn * qn.conjugate())
    #     A2 = energy / (0.5 * m * w2)
    #     amplitude = np.sqrt(A2)
    #     equipartition = energy.shape[energy.dims.index("mode")] * energy / energy.sum("time")
    #     occupation = energy / np.sqrt(w2)
    #     phases = np.arctan2(-vn, qn)

    #     out = {
    #         "energy": energy,
    #         "amplitude": amplitude,
    #         "equipartition": equipartition,
    #         "occupation": occupation,
    #         "phases": phases
    #     }

    #     self.mode = save

    #     return out

        
       
        # proj_displ = MicroState.project_displacement(q.T,self.proj).T
        # if not null_vel :
        #     proj_vel   = MicroState.project_velocities  (v.T,   self.proj, self.eigvals).T
        # else :
        #     proj_vel = np.zeros(proj_displ.shape)

        # if skip :
        #     proj_vel   = proj_vel  [:,Ndof:]
        #     proj_displ = proj_displ[:,Ndof:]
        #     w2 = self.eigvals[Ndof:]
        # else :
        #     w2 = self.eigvals
        
        # A2 = ( np.square(proj_displ) + np.square(proj_vel) )
        # energy = ( w2 * A2 / 2.0 ) # w^2 A^2 / 2
        # #energy [ energy == np.inf ] = np.nan
        # normalized_energy = ( ( self.Nmodes - Ndof ) * energy.T / energy.sum(axis=1).T ).T
        # Aamplitudes = np.sqrt(A2)

        # # print(norm(proj_displ-c))
        # # print(norm(proj_vel-s))
        
        # # Vs = MicroState.potential_energy_per_mode(proj_displ,self.eigvals)
        # # Ks = MicroState.kinetic_energy_per_mode  (proj_vel,  self.eigvals)
        # # Es = Vs + Ks        
        # # print(norm(energy-Es.T))

        # # self.energy = self.occupations = self.phases = self.Aamplitudes = self.Bamplitudes = None 
    
        # # energy = Es.T
        # occupations = energy / np.sqrt( w2 ) # - 0.5 # hbar = 1 in a.u.
        # # A  = np.sqrt( 2 * Es.T / self.eigvals  )
        # # print(norm(A-Aamplitudes))
        # if skip :
        #     tmp = np.zeros((Aamplitudes.shape[0],self.Nmodes))
        #     tmp[:,Ndof:] = Aamplitudes
        #     Bamplitudes = self.A2B(A=tmp)
        #     Bamplitudes = Bamplitudes[:,Ndof:]
        # else :
        #     Bamplitudes = self.A2B(A=Aamplitudes)
        
        # if hasattr(self,"properties") and "time" in self.properties:
        #     time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        # else :
        #     time = np.zeros(len(Bamplitudes))
        # phases = np.arctan2(-proj_vel,proj_displ) - np.outer(np.sqrt( w2 ) , time).T
        # # phases = np.unwrap(phases,discont=0.0,period=2*np.pi)

        # out = {"energy": energy,\
        #        "norm-energy": normalized_energy,\
        #        "occupations": occupations,\
        #        "phases": phases,\
        #        "A-amplitudes": Aamplitudes,\
        #        "B-amplitudes": Bamplitudes}
        
        # if inplace:
        #     self.energy = energy
        #     self.occupations = occupations
        #     self.phases = phases
        #     self.Aamplitudes = Aamplitudes
        #     self.Bamplitudes = Bamplitudes
        #     self.normalized_energy = normalized_energy

        # if MicroStatePrivate.debug :
        #     test = self.project_on_cartesian_coordinates(Aamplitudes,phases,inplace=False)
        #     #print(norm(test["positions"] - self.positions))
        #     print(norm(test["displacements"] - q))
        #     print(norm(test["velocities"] -v))            

        # return out