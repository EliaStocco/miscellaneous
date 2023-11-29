# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import os
import argparse
import numpy as np
from ase.io import read,write
from icecream import ic
from copy import copy

DEBUG= True


def get_one_file_in_folder(folder, ext, pattern=None):
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            if pattern is None:
                files.append(os.path.join(folder, file))
            elif pattern in file:
                files.append(os.path.join(folder, file))

    if len(files) == 0:
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1:
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

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

def project_displacement(displ,proj):
    return proj @ displ

def project_velocities(vel,proj,eigvals):
    return diag_matrix(eigvals,"-1/2") @ proj @ vel


def potential_energy_per_mode(proj_displ,eigvals): #,hess=None,check=False):
    """return an array with the potential energy of each vibrational mode"""        
    return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ


def kinetic_energy_per_mode(proj_vel,eigvals): #,check=False):
    """return an array with the kinetic energy of each vibrational mode"""        
    return 0.5 * ( np.square(proj_vel).T * eigvals ).T #, 0.5 * ( proj_vel * eigvals ) * identity @ ( eigvals * proj_vel )


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

def A2B(self,A,N=None,M=None,E=None):
    """
    purpose:
        convert the A-amplitude [length x mass^{-1/2}] into B-amplitudes [length]

    input :
        A : A-amplitudes
        N : normal modes (normalized)
        M : masses
        E : eigevectors (of the dynamical matrix)

    output:
        B : B-amplitudes
    """
    if N is None : N = self.ortho_modes
    if M is None : M = self.masses
    if E is None : E = self.eigvec

    if DEBUG: 
        print("A shape : ",A.shape)
        print("N shape : ",N.shape)
        print("M shape : ",M.shape) 
        print("E shape : ",E.shape)

    B = (np.linalg.inv(N) @ diag_matrix(M,"-1/2") @ E @ A.T).T
    if DEBUG: 
        print("B shape : ",B.shape)

    return B

def B2A(self,B,N=None,M=None,E=None):
    """
    purpose:
        convert the B-amplitude [length] into A-amplitudes [length x mass^{-1/2}]

    input :
        B : B-amplitudes
        N : normal modes (normalized)
        M : masses
        E : eigevectors (of the dynamical matrix)

    output:
        A : A-amplitudes
    """
    
    if N is None : N = self.ortho_modes
    if M is None : M = self.masses
    if E is None : E = self.eigvec
    
    if DEBUG: 
        print("B shape : ",B.shape)
        print("N shape : ",N.shape)
        print("M shape : ",M.shape)
        print("E shape : ",E.shape)

    A = (E.T @ diag_matrix(M,"1/2") @ N @ B.T).T
    if DEBUG: 
        print("A shape : ",A.shape)
    
    return A

def project_on_vibrational_modes(self,deltaR=None,v=None,inplace=True,Ndof=3,skip=True):

    if deltaR is None :
        deltaR = self.displacements
    elif len(deltaR.shape) == 1 :
        deltaR = deltaR.reshape(1,-1) 

    null_vel = False
    if v is None :
        v = self.velocities
    if np.isscalar(v):
        null_vel = True
        v = np.zeros(deltaR.shape)
    elif len(v.shape) == 1 :
        v = v.reshape(1,-1)


    
    # arrays = [  self.displacements,\
    #             self.velocities,\
    #             #self.modes, \
    #             #self.hess, \
    #             self.eigvals, \
    #             #self.Nmodes, \
    #             #self.dynmat, \
    #             #self.eigvec, \
    #             #self.Nconf,\
    #             #self.masses,\
    #             self.ortho_modes,\
    #             self.proj,\
    #             self.time ]
    
    # if np.any( arrays is None ) :
    #     raise ValueError("'compute': some arrays are missing")

    # c = ( self.proj @ deltaR.T )
    # s = ( diag_matrix(self.eigvals,"-1/2") @ self.proj @ v.T )
    # A = np.sqrt(np.square(c) + np.square(s))
    
    proj_displ = project_displacement(deltaR.T,self.proj).T
    if not null_vel :
        proj_vel   = project_velocities  (v.T,   self.proj, self.eigvals).T
    else :
        proj_vel = np.zeros(proj_displ.shape)

    if skip :
        proj_vel   = proj_vel  [:,Ndof:]
        proj_displ = proj_displ[:,Ndof:]
        w2 = self.eigvals[Ndof:]
    else :
        w2 = self.eigvals
    
    A2 = ( np.square(proj_displ) + np.square(proj_vel) )
    energy = ( w2 * A2 / 2.0 ) # w^2 A^2 / 2
    #energy [ energy == np.inf ] = np.nan
    normalized_energy = ( ( self.Nmodes - Ndof ) * energy.T / energy.sum(axis=1).T ).T
    Aamplitudes = np.sqrt(A2)

    # print(norm(proj_displ-c))
    # print(norm(proj_vel-s))
    
    # Vs = MicroState.potential_energy_per_mode(proj_displ,self.eigvals)
    # Ks = MicroState.kinetic_energy_per_mode  (proj_vel,  self.eigvals)
    # Es = Vs + Ks        
    # print(norm(energy-Es.T))

    # self.energy = self.occupations = self.phases = self.Aamplitudes = self.Bamplitudes = None 

    # energy = Es.T
    occupations = energy / np.sqrt( w2 ) # - 0.5 # hbar = 1 in a.u.
    # A  = np.sqrt( 2 * Es.T / self.eigvals  )
    # print(norm(A-Aamplitudes))
    if skip :
        tmp = np.zeros((Aamplitudes.shape[0],self.Nmodes))
        tmp[:,Ndof:] = Aamplitudes
        Bamplitudes = self.A2B(A=tmp)
        Bamplitudes = Bamplitudes[:,Ndof:]
    else :
        Bamplitudes = self.A2B(A=Aamplitudes)
    
    if hasattr(self,"properties") and "time" in self.properties:
        time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
    else :
        time = np.zeros(len(Bamplitudes))
    phases = np.arctan2(-proj_vel,proj_displ) - np.outer(np.sqrt( w2 ) , time).T
    # phases = np.unwrap(phases,discont=0.0,period=2*np.pi)

    out = {"energy": energy,\
            "norm-energy": normalized_energy,\
            "occupations": occupations,\
            "phases": phases,\
            "A-amplitudes": Aamplitudes,\
            "B-amplitudes": Bamplitudes}
    
    if inplace:
        self.energy = energy
        self.occupations = occupations
        self.phases = phases
        self.Aamplitudes = Aamplitudes
        self.Bamplitudes = Bamplitudes
        self.normalized_energy = normalized_energy

    # if DEBUG :
    #     test = self.project_on_cartesian_coordinates(Aamplitudes,phases,inplace=False)
    #     #print(norm(test["positions"] - self.positions))
    #     print(norm(test["displacements"] - deltaR))
    #     print(norm(test["velocities"] -v))            

    return out

# This script project a MD trajectoies onto the vibrational modes.

# Input:
# - MD trajectories (positions and velocities)
# - some output files produced by the vibrational analysis
# - the equilibrium nuclear configuration w.r.t. the vibrational analysis has been performed
# - the time (in a.u.) at which the nuclear configuration are evaluated
# The latter is necessary in order to compute the displacements w.r.t. this configuration.

# Some comments:
# The projection on the vibrational modes is just a change of variables.
# Actually it means to change from cartesian coordinates to angle-action variables (more or less).

# Output:
#  - the amplitudes of the modes in two different formats:
#  - A-amplitudes.txt (dimension of lenght x mass^{1/2})
#  - B-amplitudes (dimensionless)
#         (The A-amplitudes have to be considered in case the vibrational modes are written in terms of the dynamical matrix eigenvector and the nuclear masses.
#         The B-amplitudes have to be considered in case the vibrational modes are written in terms of the (dimensionless and orthonormalized) normal modes.
#         Ask to the author for more details.)
#  - the "initial phases" of the modes (the angle variable are actually wt+\phi and not just \phi).
#  - the energy of each vibrational modes
#  - the (classical) occupation number of each vibrational modes (i.e. the zero-point-energy is neglected)

# The script can produce as additional output:
#  - some plots of the vibrationalmodes energies
#         (The energies of the vibrational modes can be used to check the equipartition theorem is valid, i.e. whether the system has thermalized.)

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

# //"args":["-c","true","-q","i-pi.positions_0.xyz","-v","i-pi.velocities_0.xyz","-r","start.xyz","-o","output","-m","vib","-p","output/energy.pdf","-pr","i-pi.properties.out"],//"-t","2500000"],

description = ""
# description =  \
# " \n\
# \tThis script project a MD trajectoies onto the vibrational modes.\n\
# \n\tInput:\n\
# \t- MD trajectories (positions and velocities)\n\
# \t- some output files produced by the vibrational analysis\n\
# \t- the equilibrium nuclear configuration w.r.t. the vibrational analysis has been performed\n\
# \t- the time (in a.u.) at which the nuclear configuration are evaluated\n\
# \tThe latter is necessary in order to compute the displacements w.r.t. this configuration.\n\
# \n\tSome comments:\n\
# \tThe projection on the vibrational modes is just a change of variables.\n\
# \tActually it means to change from cartesian coordinates to angle-action variables (more or less).\n\
# \n\tOutput:\n\
# \t - the amplitudes of the modes in two different formats:\n\
# \t - A-amplitudes.txt (dimension of lenght x mass^{1/2})\n\
# \t - B-amplitudes (dimensionless)\n\
# \t\t(The A-amplitudes have to be considered in case the vibrational modes are written in terms of the dynamical matrix eigenvector and the nuclear masses.\n\
# \t\tThe B-amplitudes have to be considered in case the vibrational modes are written in terms of the (dimensionless and orthonormalized) normal modes.\n\
# \t\tAsk to the author for more details.)\n\
# \t - the \"initial phases\" of the modes (the angle variable are actually wt+\phi and not just \phi).\n\
# \t - the energy of each vibrational modes\n\
# \t - the (classical) occupation number of each vibrational modes (i.e. the zero-point-energy is neglected)\n\
# \n\tThe script can produce as additional output:\n\
# \t - some plots of the vibrationalmodes energies\n\
# \t\t(The energies of the vibrational modes can be used to check the equipartition theorem is valid, i.e. whether the system has thermalized.)\n\
# \n\tauthor: Elia Stocco\
# \n\temail : stocco@fhi-berlin.mpg.de"

# - the characteristic mass of the vibrationa modes\n\
# - the characteristic lenght of the vibrationa modes\n\

class NormalModes():

    def __init__(self,Nmodes,Ndof):

        # Nmodes
        self.Nmodes = int(Nmodes)
        self.Ndof = int(Ndof)

        # Natoms
        self.Natoms = int(self.Ndof / 3)

        empty = np.full((self.Ndof,self.Nmodes),np.nan)
        self.ortho_modes = empty.copy()
        self.eigvec = empty.copy()
        self.dynmat = empty.copy()
        self.modes = empty.copy()
        self.proj = empty.copy()

        self.eigvals = np.full(self.Nmodes,np.nan)
        self.masses = np.full(self.Ndof,np.nan)

        pass
    
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
    
    def eigvec2modes(self):
        self.modes = diag_matrix(self.masses,"-1/2") @ self.eigvec
        # self.ortho_modes[:,:] = self.modes / np.linalg.norm(self.modes,axis=0)

    def eigvec2proj(self):
        self.proj = self.eigvec.T @ diag_matrix(self.masses,"1/2")

    def project_displacement(self,displ):
        return self.proj @ displ

    def project_velocities(self,vel):
        return diag_matrix(self.eigvals,"-1/2") @ self.proj @ vel
    
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
                phi = int(cmath.phase(phase)*180/np.pi)
                ic(k,r,phi)
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



def prepare_parser():

    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-t", "--trajectory",  type=str,**argv,
        help="input file with the trajectory [a.u., extxyz]"#, default=None
    )
    parser.add_argument(
        "-r", "--relaxed",  type=str,**argv,
        help="input file with the relaxed atomic structure (a.u.)"#, default=None
    )
    parser.add_argument(
        "-c", "--cell",  type=str,**argv,
        help="file with the primitive unit cell [a.u., extxyz]"#, default=None
    )
    parser.add_argument(
        "-sc", "--super_cell",  type=str,**argv,
        help="file with the super cell [a.u., extxyz]"#, default=None
    )
    parser.add_argument(
        "-cm", "--cell_modes",  type=str,**argv,
        help="prefix of the i-PI output files of the vibrational analysis"#, default=None
    )
    parser.add_argument(
        "-scm", "--super_cell_modes",  type=str,**argv,
        help="prefix of the i-PI output files of the vibrational analysis"#, default=None
    )
    parser.add_argument(
        "-o", "--output",  type=str,**argv,
        help="output file", default="output.extxyz"
    )  
    options = parser.parse_args()

    return options

def main():
    ###
    # prepare/read input arguments
    # print("\n\tReding input arguments")
    args = prepare_parser()

    print(description)

    trajectory = read(args.trajectory,format="extxyz",index=":")
    relaxed = read(args.relaxed)
    cell = read(args.cell).cell
    supercell = read(args.super_cell).cell
    cnm  = NormalModes.load(args.cell_modes)
    scnm = NormalModes.load(args.super_cell_modes)


    print("\t      cell # atoms: ",cnm.Natoms)
    print("\t supercell # atoms: ",scnm.Natoms)

    # np.asarray(supercell).T = M @ np.asarray(cell).T
    M = np.asarray(supercell).T @ np.linalg.inv(np.asarray(cell).T)

    size = M.round(0).diagonal().astype(int)

    print("\tsupercell size: ",size)

    factor_atoms = int(M.round(0).diagonal().astype(int).prod())

    if scnm.Natoms != cnm.Natoms * factor_atoms:
        raise ValueError("error")
    
    cnm = cnm.remove_dof([0,1,2])
    # cnm.eigvec = cnm.eigvec[:,3:]
    bsnm = cnm.build_supercell_normal_modes(size)

    
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()