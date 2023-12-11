#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
import os
import argparse
import numpy as np
from ase.io import read, write
from ase import Atoms
from icecream import ic
from copy import copy
from miscellaneous.elia.formatting import matrix2str
import pandas as pd
from ase.geometry import get_distances
from itertools import product

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


def prepare_parser():

    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument(
        "-t", "--trajectory",  type=str,**argv,
        help="input file with the trajectory [a.u., extxyz]"#, default=None
    )
    # parser.add_argument(
    #     "-cr", "--cell_relaxed",  type=str,**argv,
    #     help="cell relaxed atomic structure (a.u.)"#, default=None
    # )
    # parser.add_argument(
    #     "-scr", "--super_cell_relaxed",  type=str,**argv,
    #     help="supercell relaxed atomic structure (a.u.)"#, default=None
    # )
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

def compute_relative_distances(config1:Atoms, config2:Atoms):
    """
    Compute relative distances between all pairs of atoms in two configurations.

    Parameters:
    - config1: ase.Atoms object representing the first atomic configuration.
    - config2: ase.Atoms object representing the second atomic configuration.

    Returns:
    - distances_df: Pandas DataFrame containing the upper triangular part of the distance matrix.
    """
    positions1 = config1.get_positions()
    positions2 = config2.get_positions()

    N = len(positions1)
    if N != len(positions2):
        raise ValueError("error")

    distances = np.full((N,N),np.nan)
    for r in range(N):
        for c in range(r,N):
            distances[r,c] = np.linalg.norm(positions1[r,:] - positions2[c,:])
            distances[c,r] = distances[r,c]
    return distances

def find_trasformation(c:Atoms,sc:Atoms):
    M = np.asarray(sc.cell).T @ np.linalg.inv(np.asarray(c.cell).T)
    size = M.round(0).diagonal().astype(int)
    return size, M

def find_replica(supercell:Atoms,cell:Atoms):

    size, M = find_trasformation(c=cell,sc=supercell)

    # distances = get_distances(cell.repeat(size),supercell) 

    # this could be wrong
    distances = compute_relative_distances(cell.repeat(size),supercell)
    a = np.min(distances,axis=1)
    b = np.min(distances,axis=0)
    if not np.all(a == b):
        raise ValueError("error")
    index = np.argmin(distances,axis=0)
    N = cell.get_global_number_of_atoms()
    return index % N

def plot_matrix(M,Natoms=None,file=None):
    import matplotlib.pyplot as plt  
    # from matplotlib.colors import ListedColormap
    # Create a figure and axis
    fig, ax = plt.subplots()  
    argv = {
        "alpha":0.5
    }
    ax.matshow(M, origin='upper',extent=[0, M.shape[1], M.shape[0], 0],**argv)
    if Natoms is not None:
        argv = {
            "linewidth":0.8,
            "linestyle":'--',
            "color":"white",
            "alpha":1
        }
        xx = np.arange(0,M.shape[0],Natoms*3)
        yy = np.arange(0,M.shape[1],Natoms*3)
        for x in xx:
            ax.axhline(x, **argv) # horizontal lines
        for y in yy:
            ax.axvline(y, **argv) # horizontal lines
        
        

        xx = xx + np.unique(np.diff(xx)/2)
        N = int(np.power(len(xx),1/3)) # int(np.log2(len(xx)))
        ticks = list(product(*([np.arange(N).tolist()]*3)))
        ax.set_xticks(xx)
        ax.set_xticklabels([str(i) for i in ticks])
        # ax.xaxis.set(ticks=xx, ticklabels=[str(i) for i in ticks])
        
        yy = yy + np.unique(np.diff(yy)/2)
        N = int(np.power(len(yy),1/3))
        ticks = list(product(*([np.arange(N).tolist()]*3)))
        # ax.yaxis.set(ticks=yy, ticklabels=ticks)
        ax.set_yticks(yy)
        ax.set_yticklabels([str(i) for i in ticks])

    plt.tight_layout()
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    return

def sc2c(M,Natoms):

    xx = np.arange(0,M.shape[0],Natoms*3)
    N = int(np.power(len(xx),1/3))
    cols = list(product(*([np.arange(N).tolist()]*3)))
    # cols = [np.asarray(i) for i in cols ]
    df = pd.DataFrame(columns=cols,index=cols)
    xx = np.arange(0,M.shape[0]+1,Natoms*3)
    for i,r in enumerate(cols):
        for j,c in enumerate(cols):
            df.at[r,c] = M[xx[i]:xx[i+1],xx[j]:xx[j+1]]
    return df



def main():
    ###
    # prepare/read input arguments
    # print("\n\tReding input arguments")
    args = prepare_parser()

    print(description)

    trajectory = read(args.trajectory,format="extxyz",index=":")
    # Crelaxed = read(args.cell_relaxed)
    # SCrelaxed = read(args.super_cell_relaxed)
    cell = read(args.cell)# .cell
    supercell = read(args.super_cell)#.cell
    cnm  = NormalModes.load(args.cell_modes)
    scnm = NormalModes.load(args.super_cell_modes)

    print("\tcell:\n\t# atoms: ",cnm.Natoms)
    tmp = np.asarray(cell.cell).T
    print(matrix2str(tmp.round(4),col_names=["1","2","3"],cols_align="^",width=6))

    print("\tsupercell:\n\t# atoms: ",scnm.Natoms)
    tmp = np.asarray(supercell.cell).T
    print(matrix2str(tmp.round(4),col_names=["1","2","3"],cols_align="^",width=6))

    # np.asarray(supercell).T = M @ np.asarray(cell).T
    print("\ttrasformation matrix:")
    size, M = find_trasformation(c=cell,sc=supercell)
    print(matrix2str(M.round(2),col_names=["1","2","3"],cols_align="^",width=6))
    print("\tsupercell size: ",size)

    plot_matrix(scnm.dynmat,cnm.Natoms,"supercell.dynmat.pdf")

    dynmat = sc2c(scnm.dynmat,cnm.Natoms)

    dynmatK = pd.DataFrame(index=dynmat.index,columns=["dynmat","eigvec","eigvals"],dtype=object)
    for i in dynmat.index:

        # Fourier transform
        dynmatK.at[i,"dynmat"] = np.zeros(dynmat.at[i,i].shape)
        for c in dynmat.columns:
            kr = np.asarray(c) / size @ np.asarray(i)
            phase = np.exp(1.j * 2 * np.pi * kr )
            dynmatK.at[i,"dynmat"] = dynmatK.at[i,"dynmat"] + phase * dynmat.at[i,c]
        imag = dynmatK.at[i,"dynmat"].imag
        real = dynmatK.at[i,"dynmat"].real
        ic(np.linalg.norm(imag))
        ic(np.linalg.norm(real))
        dynmatK.at[i,"dynmat"] = real

        # Diagonalization
        if not np.allclose(real, real.T):
            raise ValueError("not symmetric")
        
        eigenvalues, eigenvectors = np.linalg.eigh(real)
        dynmatK.at[i,"eigvec"] = eigenvectors
        dynmatK.at[i,"eigvals"] = eigenvalues

        # fuck, their negative





    factor_atoms = int(M.round(0).diagonal().astype(int).prod())

    if scnm.Natoms != cnm.Natoms * factor_atoms:
        raise ValueError("error")
    
    cnm = cnm.remove_dof([0,1,2])
    # cnm.eigvec = cnm.eigvec[:,3:]
    bsnm = cnm.build_supercell_normal_modes(size)

    # distances = compute_relative_distances(cell.repeat(size),supercell)
    index = find_replica(supercell,cell)
    print("\treplica: ",index)


    plot_matrix(scnm.dynmat,None)

    
    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()