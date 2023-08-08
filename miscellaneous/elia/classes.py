from miscellaneous.elia.functions import get_one_file_in_folder,getproperty,output_file,get_property_header
from miscellaneous.elia.functions import convert, Dict2Obj, get_attributes, merge_attributes, read_comments_xyz
import os
from ase import io
from ase import Atoms
import numpy as np
import numpy.linalg as linalg
norm = linalg.norm
import matplotlib.pyplot as plt
from ipi.utils.units import unit_to_internal, unit_to_user
import pickle
import numpy.random as rand
import pandas as pd
#from reloading import reloading
import re
import ipi.utils.mathtools as mt
from copy import deepcopy

__all__ = ["MicroState"]

# To Do:
# - create 'trajectory' attribute, that containes positions, velocities, forces, etc as ase.Atoms object

deg2rad = np.pi / 180.0
abcABC = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")

# reloading: https://towardsdatascience.com/introducing-reloading-never-re-run-your-python-code-again-to-print-more-details-374bee33473d

class MicroStatePrivate:

    debug = False
    tab = "\t\t"
    check = True
    thr = 0.1
    check_orth = True
    fmt = "%20.12e"
    # output files
    ofile = {"energy":"energy.txt",\
             "phases":"phases.txt",\
             "occupations":"occupations.txt",\
             "A-amplitudes":"A-amplitudes.txt",\
             "B-amplitudes":"B-amplitudes.txt",\
             "violin":"violin.csv"}
    
    smallest_float = np.nextafter(0,1)
    

class MicroState:

    #@reloading
    def add(self,*argv,**argc):
        try :
            print("Adding new attributes ...\n")
            # # empty object
            # temp = MicroState()
            # # give it the attributes
            # temp = merge_attributes(temp,self)
            # # read new attrbitues
            temp = deepcopy(self)
            MicroState.__init__(temp,*argv,**argc)
        except:
            raise ValueError("Error in __int__")
        
        print("Merging new attributes ...")
        temp = merge_attributes(self,temp)
        print("Merging completed :)\n")
        #return temp
        
    #@reloading
    def __init__(self,instructions=None,todo=None,toread=None,options=None):

        print("Initializing object of type 'MicroState' ...")

        if instructions is None :
            instructions= dict()

        if options is None :
            options= dict()

        if type(instructions) == dict :
            instructions = Dict2Obj(instructions)
        
        attribute_names  = [ "relaxed", "positions", "displacement", "velocities", "cells", "types" ]
        attribute_names += [ "eigvals", "dynmat", "eigvec", "modes", "ortho_modes", "masses" ]
        attribute_names += [ "Nmodes", "Nconf" ]
        attribute_names += [ "energy", "Aamplitudes", "Bamplitudes", "properties" ]

        for name in attribute_names:
            if not hasattr(self,name):
                setattr(self, name, None)

        if todo is None :
            todo = ""
        if toread is None :
            toread = list()

        for k in get_attributes(instructions):
            toread.append(k)

        if todo == "vib":
            toread += [ "masses",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals"]            

        if todo == "proj-on-vib-modes" :
            toread += [ "relaxed",\
                        "masses",\
                        "positions",\
                        "displacements",\
                        "velocities",\
                        #"modes",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals",\
                        #"dynmat",\
                        "properties"]
            
        if todo == "plot-vib-modes-energy" :
            toread += [ "eigvals",\
                        "eigvec",\
                        "energy",\
                        "A-amplitudes",\
                        "properties"]
            
        if todo == "generate-thermal-state":
            toread += [ "relaxed",\
                        "masses",\
                        "ortho_modes",\
                        "proj",\
                        "eigvec",\
                        "hess",\
                        "eigvals",\
                        "atoms"]
            
        print("\nProperties to be read:")
        for k in toread:
            print("{:s}".format(MicroStatePrivate.tab),k)

        ###################
        #                 # 
        #  start reading  #
        #                 #
        ###################

        if "relaxed" in toread:
            ###
            # reading original position
            print("{:s}reading original/relaxed position from file '{:s}'".format(MicroStatePrivate.tab,instructions.relaxed))
            tmp = io.read(instructions.relaxed)
            atoms = tmp.get_chemical_symbols()
            relaxed = tmp.positions
            if relaxed.shape[1] != 3 :
                raise ValueError("the relaxed configurations do not have 3 components")
            self.Nmodes = relaxed.shape[0] * 3
            self.relaxed = relaxed.flatten()


        if "masses" in toread:
    
            if not hasattr(instructions, 'masses') or instructions.masses is None :
                if not os.path.isdir(instructions.modes):
                    raise ValueError("'--modes' should be a folder")            
                file = get_one_file_in_folder(folder=instructions.modes,ext=".masses")
                
            else :
                file = instructions.masses
            print("{:s}reading masses from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.masses = np.loadtxt(file)

            # no longer supported
            # if len(masses) == len(relaxed.positions) :
            #     # set masses
            #     M = np.zeros((3 * len(masses)), float)
            #     M[ 0 : 3 * len(masses) : 3] = masses
            #     M[ 1 : 3 * len(masses) : 3] = masses
            #     M[ 2 : 3 * len(masses) : 3] = masses
            #     masses = M

            # elif len(masses) != 3 * len(relaxed.positions):            
            #     raise ValueError("wrong number of nuclear masses")
                        
            # positions
            # relaxed = relaxed.positions
            # Nmodes = relaxed.shape[0] * 3

        if "positions" in toread:

            print("{:s}reading positions from file '{:s}'".format(MicroStatePrivate.tab,instructions.positions))
            positions0 = io.read(instructions.positions,index=":")
            
            tmp = positions0[0]
            atoms = tmp.get_chemical_symbols()
            Nconf = len(positions0) 

            a = tmp.positions.shape[0] * tmp.positions.shape[1]
            positions = np.zeros((len(positions0),a))
            for n in range(Nconf):
                positions[n] = positions0[n].positions.flatten()

            if self.Nmodes is None :
                self.Nmodes = len(positions[0])
            for i in range(Nconf):
                if np.asarray( len(positions[i]) != self.Nmodes) :
                    raise ValueError("some configurations do not have the correct shape")

            self.positions = np.asarray(positions)
            self.Nconf = Nconf

            if "types" in toread:
                print("{:s}reading atomic types from file '{:s}'".format(MicroStatePrivate.tab,instructions.types))
                self.types = [ system.get_chemical_symbols() for system in positions0 ]
                self.numbers = [ system.numbers for system in positions0 ]

            del positions
            del positions0

        # if "types" in toread:

        #     print("{:s}reading atomic types from file '{:s}'".format(MicroStatePrivate.tab,instructions.types))
        #     if not hasattr(instructions,"types"):
        #         instructions.types = instructions.positions
        #     positions = io.read(instructions.types,index=":")
        #     self.types = [ system.get_chemical_symbols() for system in positions ]
        #     self.numbers = [ system.numbers for system in positions ]


        if "cells" in toread :
            print("{:s}reading cells (for each configuration) from file '{:s}'".format(MicroStatePrivate.tab,instructions.cells))

            comments = read_comments_xyz(instructions.cells)
            cells = [ abcABC.search(comment) for comment in comments ]
            self.cell = np.zeros((len(cells),3,3))
            for n,cell in enumerate(cells):
                a, b, c = [float(x) for x in cell.group(1).split()[:3]]
                alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
                self.cell[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

            
        if "atoms" in toread:
            self.atoms = atoms


        if "displacements" in toread:

            if self.positions is None :
                raise ValueError("'positions' not defined")
            if self.relaxed is None:
                raise ValueError("'relaxed' not defined")            
            self.displacements = np.asarray(self.positions) - np.asarray(self.relaxed)


        if "velocities" in toread:

            if instructions.velocities is None:
                print("{:s}setting velocities to zero".format(MicroStatePrivate.tab))
                if self.positions is None :
                    raise ValueError("'positions' not defined")
                self.velocities = np.zeros(self.positions.shape)

            else :
                print("{:s}reading velocities from file '{:s}'".format(MicroStatePrivate.tab,instructions.velocities))
                velocities = io.read(instructions.velocities,index=":")
                Nvel = len(velocities)
                print("{:s}read {:d} velocities".format(MicroStatePrivate.tab,Nvel))
                if self.Nconf is not None :
                    if Nvel != self.Nconf :
                        raise ValueError("number of velocities and positions configuration are different")
                for n in range(Nvel):
                    velocities[n] = velocities[n].positions.flatten()
                self.velocities = np.asarray(velocities)

        if "forces" in toread:

            print("{:s}reading forces from file '{:s}'".format(MicroStatePrivate.tab,instructions.forces))
            forces = io.read(instructions.forces,index=":")
            Nforces = len(forces)
            print("{:s}read {:d} forces".format(MicroStatePrivate.tab,Nforces))
            if self.Nconf is not None :
                if Nforces != self.Nconf :
                    raise ValueError("number of forces and positions configuration are different")
            for n in range(Nforces):
                forces[n] = forces[n].positions.flatten()
            self.forces = np.asarray(forces)


        if "ortho_modes" in toread:   

            if not os.path.isdir(instructions.modes):
                raise ValueError("'--modes' should be a folder")

            print("{:s}searching for '*.mode' file in folder '{:s}'".format(MicroStatePrivate.tab,instructions.modes))            
            file = get_one_file_in_folder(folder=instructions.modes,ext=".mode")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            modes = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = modes.shape[0]
            if modes.shape[0] != self.Nmodes or modes.shape[1] != self.Nmodes :
                raise ValueError("vibrational modes matrix with wrong size")
            if modes.shape[0] != modes.shape[1]:
                raise ValueError("vibrational modes matrix is not square")
            self.ortho_modes = modes


        if "eigvec" in toread:

            if not os.path.isdir(instructions.modes):
                raise ValueError("'--modes' should be a folder")
            
            file = get_one_file_in_folder(folder=instructions.modes,ext=".eigvec")
            print("{:s}reading eigenvectors from file '{:s}'".format(MicroStatePrivate.tab,file))
            eigvec = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = eigvec.shape[0]
            if eigvec.shape[0] != self.Nmodes or eigvec.shape[1] != self.Nmodes:
                raise ValueError("eigenvectors matrix with wrong size")
                
            # check that the eigenvectors are orthogonal (they could not be so)
            if MicroStatePrivate.check_orth :                
                print("{:s}checking that the eigenvectors are orthonormal, i.e. M @ M^t = Id".format(MicroStatePrivate.tab))
                res = np.linalg.norm(eigvec @ eigvec.T - np.eye(self.Nmodes))
                print("{:s} | M @ M^t - Id | = {:>20.12e}".format(MicroStatePrivate.tab,res))
                if res > MicroStatePrivate.thr :
                    raise ValueError("the eigenvectors are not orthonormal")            
            self.eigvec = eigvec


        if "modes" in toread:   
            if self.eigvec is None :
                raise ValueError("'eigvec' not defined")
            if self.masses is None:
                raise ValueError("'masses' not defined")   
            self.modes = MicroState.diag_matrix(self.masses,"-1/2") @ self.eigvec


        if "proj" in toread:   
            if self.eigvec is None :
                raise ValueError("'eigvec' not defined")
            if self.masses is None:
                raise ValueError("'masses' not defined")   
            self.proj = self.eigvec.T @ MicroState.diag_matrix(self.masses,"1/2")


        # I read the full hessian
        if "hess" in toread:
        
            if not os.path.isdir(instructions.modes):
                    raise ValueError("'--modes' should be a folder")   
                
            file = get_one_file_in_folder(folder=instructions.modes,ext="_full.hess")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            hess = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = hess.shape[0]
            if hess.shape[0] != self.Nmodes or hess.shape[1] != self.Nmodes:
                raise ValueError("hessian matrix with wrong size")           
            self.hess = hess


        # # pay attention: I never use it, so it has still to be debugged
        # if "full_hess" in toread:

        #     if not os.path.isdir(instructions.modes):
        #             raise ValueError("'--modes' should be a folder")   
            
        #     file = get_one_file_in_folder(folder=instructions.modes,ext="_full.hess")
        #     print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
        #     full_hess = np.loadtxt(file)
        #     if self.Nmodes is not None :
        #         if hess.shape[0] != self.Nmodes or hess.shape[1] != self.Nmodes:
        #             raise ValueError("full hessian matrix with wrong size")       
        #     else :
        #         self.Nmodes = len       
        #     self.full_hess = full_hess
            

        if "eigvals" in toread:

            if not os.path.isdir(instructions.modes):
                    raise ValueError("'--modes' should be a folder")   
            
            file = get_one_file_in_folder(folder=instructions.modes,ext=".eigval")
            print("{:s}reading vibrational modes from file '{:s}'".format(MicroStatePrivate.tab,file))
            eigvals = np.loadtxt(file)
            if self.Nmodes is not None :
                if len(eigvals) != self.Nmodes:
                    raise ValueError("eigenvalues array with wrong size")
            else :
                self.Nmodes = len(eigvals)
            self.eigvals = eigvals
            if np.any( self.eigvals < 0.0 ):
                print("{:s}!**Warning**: some eigenvalues are negative, setting them to (nearly) zero".format(MicroStatePrivate.tab))
                self.eigvals = np.asarray( [ MicroStatePrivate.smallest_float if i < 0.0 else i for i in self.eigvals ] )
            
        if "dynmat" in toread:

            if not os.path.isdir(instructions.modes):
                    raise ValueError("'--modes' should be a folder")   
            
            file = get_one_file_in_folder(folder=instructions.modes,ext=".dynmat")
            print("{:s}reading the dynamical matrix from file '{:s}'".format(MicroStatePrivate.tab,file))
            dynmat = np.loadtxt(file)
            if self.Nmodes is None :
                self.Nmodes = dynmat.shape[0]
            if dynmat.shape[0] != self.Nmodes or dynmat.shape[1] != self.Nmodes:
                raise ValueError("dynamical matrix with wrong size")
            self.dynmat = dynmat
               
        # if MicroStatePrivate.check :
        #     print("\n{:s}Let's do a little test".format(MicroStatePrivate.tab))
        #     # mode      = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext=".mode"))
        #     # dynmat    = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext=".dynmat"))
        #     # full_hess = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext="_full.hess"))
        #     # eigvals    = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext=".eigvals"))
        #     # eigvec    = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext=".eigvec"))
        #     # hess      = np.loadtxt(get_one_file_in_folder(folder=instructions.modes,ext=".hess"))
            
        #     if np.all(a is not None for a in [self.dynmat,self.eigvec,self.eigvals]):
        #         print("{:s}checking that D@V = E@V".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(self.dynmat @ self.eigvec - self.eigvals @ self.eigvec).sum())
        #         print("{:s} | D@V - E@V | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #     if np.all(a is not None for a in [self.modes,self.eigvals]):
        #         eigsys = np.linalg.eigh(self.modes)

        #         print("{:s}checking that eigvec(M) = M".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(eigsys[1] - self.modes).flatten().sum())
        #         print("{:s} | eigvec(H) - M | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #         print("{:s}checking that eigvals(H) = E".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square( np.sort(eigsys[0]) - np.sort(self.eigvals)).sum())
        #         print("{:s} | eigvec(H) - E | = {:>20.12e}".format(MicroStatePrivate.tab,res))

        #         print("{:s}checking that H@eigvec(H) = eigvals(H)@eigvec(H)".format(MicroStatePrivate.tab))
        #         res = np.sqrt(np.square(eigsys[0] - self.eigvals).sum())
        #         print("{:s} | eigvec(H) - E | = {:>20.12e}".format(MicroStatePrivate.tab,res))
            
        if "energy" in toread:
    
            file = output_file(instructions.output,MicroStatePrivate.ofile["energy"])
            print("{:s}reading energy from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.energy = np.loadtxt(file)


        if "A-amplitudes" in toread:

            file = output_file(instructions.output,MicroStatePrivate.ofile["A-amplitudes"])
            print("{:s}reading A-amplitudes from file '{:s}'".format(MicroStatePrivate.tab,file))
            self.Aamplitudes = np.loadtxt(file)
            
            if self.energy is not None:

                if np.any(self.Aamplitudes.shape != self.energy.shape):
                    raise ValueError("energy and A-amplitudes matrix size do not match")
            
        # if "time" in toread:
        #     if instructions.properties is None:
        #         raise ValueError("The file with the system (time-dependent) properties is not defined")
            
        #     t,u = getproperty(instructions.properties,["time"])
        #     self.time  = t["time"]
        #     self.units = u["time"]

        #     if u["time"] not in ["a.u.","atomic_unit"]:
        #         print("{:s}'time' is not in 'atomic units' but in '{:s}'".format(MicroStatePrivate.tab,u["time"]))
        #         factor = unit_to_internal("time","femtosecond",1)
        #         print("{:s}converting 'time' to 'atomic units' by multiplication for {:>14.10e}".format(MicroStatePrivate.tab,factor))
        #         self.time *= factor
        #         self.units = "a.u."


        if "properties" in toread:
            if instructions.properties is None:
                raise ValueError("The file with the system (time-dependent) properties is not defined")
            
            header = get_property_header(instructions.properties,search=True)
            p,u = getproperty(instructions.properties,header)
            self.header  = get_property_header(instructions.properties,search=False)
            self.properties  = p
            self.units = u
            

        for name in attribute_names:
            if getattr(self, name) is None:
                delattr(self, name)

        print("\nInitialization completed :)") 
        pass

    @staticmethod
    def project_displacement(displ,proj):
        return proj @ displ
    
    @staticmethod
    def project_velocities(vel,proj,eigvals):
        # N = len(eigvals)
        # omega_inv = np.zeros((N,N))
        # np.fill_diagonal(omega_inv,1.0/np.sqrt(eigvals))
        # return np.nan_to_num(MicroState.diag_matrix(eigvals,"-1/2") @ proj @ vel,0.0)
        return MicroState.diag_matrix(eigvals,"-1/2") @ proj @ vel
    
    @staticmethod
    def potential_energy_per_mode(proj_displ,eigvals): #,hess=None,check=False):
        """return an array with the potential energy of each vibrational mode"""        
        return 0.5 * ( np.square(proj_displ).T * eigvals ).T #, 0.5 * proj_displ * omega_sqr @ proj_displ
    
    @staticmethod
    def kinetic_energy_per_mode(proj_vel,eigvals): #,check=False):
        """return an array with the kinetic energy of each vibrational mode"""        
        return 0.5 * ( np.square(proj_vel).T * eigvals ).T #, 0.5 * ( proj_vel * eigvals ) * identity @ ( eigvals * proj_vel )
 
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

        if MicroStatePrivate.debug: 
            print("A shape : ",A.shape)
            print("N shape : ",N.shape)
            print("M shape : ",M.shape) 
            print("E shape : ",E.shape)

        B = (np.linalg.inv(N) @ MicroState.diag_matrix(M,"-1/2") @ E @ A.T).T
        if MicroStatePrivate.debug: 
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
        
        if MicroStatePrivate.debug: 
            print("B shape : ",B.shape)
            print("N shape : ",N.shape)
            print("M shape : ",M.shape)
            print("E shape : ",E.shape)

        A = (E.T @ MicroState.diag_matrix(M,"1/2") @ N @ B.T).T
        if MicroStatePrivate.debug: 
            print("A shape : ",A.shape)
        
        return A

    def project_on_cartesian_coordinates(self,Aamp=None,phases=None,inplace=True):
        
        if Aamp is None :
            Aamp = self.Aamplitudes
        if phases is None :
            phases = self.phases

        if len(Aamp.shape) == 1 :
            Aamp = Aamp.reshape(1,-1)
        if len(phases.shape) == 1 :
            phases = phases.reshape(1,-1)

        if hasattr(self,"properties") and "time" in self.properties :
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(Aamp))
            
        phi = np.outer(np.sqrt( self.eigvals) , time).T
        c = + Aamp * np.cos( phi + phases )
        s = - Aamp * np.sin( phi + phases)
        
        #np.linalg.inv(self.proj)
        invproj = MicroState.diag_matrix(self.masses,"-1/2") @ self.eigvec 

        deltaR = ( invproj @ c.T ).T
        v = ( invproj @ MicroState.diag_matrix(self.eigvals,"1/2") @ s.T ).T

        N = len(deltaR)
        positions = np.full((N,len(self.relaxed)),np.nan)
        for i in range(N):
            positions[i] = deltaR[i] + self.relaxed

        if inplace :
            self.displacements = deltaR
            self.velocities = v
            self.positions = positions

        return { "displacements":deltaR,\
                 "velocities":v,\
                 "positions":positions }

    # @reloading
    def project_on_vibrational_modes(self,deltaR=None,v=None,inplace=True):

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
        # s = ( MicroState.diag_matrix(self.eigvals,"-1/2") @ self.proj @ v.T )
        # A = np.sqrt(np.square(c) + np.square(s))
        
        proj_displ = MicroState.project_displacement(deltaR.T,self.proj).T
        if not null_vel :
            proj_vel   = MicroState.project_velocities  (v.T,   self.proj, self.eigvals).T
        else :
            proj_vel = np.zeros(proj_displ.shape)

        A2 = ( np.square(proj_displ) + np.square(proj_vel) )
        energy = ( self.eigvals * A2 / 2.0 ) # w^2 A^2 / 2
        #energy [ energy == np.inf ] = np.nan
        normalized_energy = ( ( self.Nmodes - 3 ) * energy.T / energy.sum(axis=1).T ).T
        Aamplitudes = np.sqrt(A2)

        # print(norm(proj_displ-c))
        # print(norm(proj_vel-s))
        
        # Vs = MicroState.potential_energy_per_mode(proj_displ,self.eigvals)
        # Ks = MicroState.kinetic_energy_per_mode  (proj_vel,  self.eigvals)
        # Es = Vs + Ks        
        # print(norm(energy-Es.T))

        # self.energy = self.occupations = self.phases = self.Aamplitudes = self.Bamplitudes = None 
    
        # energy = Es.T
        occupations = energy / np.sqrt( self.eigvals) # - 0.5 # hbar = 1 in a.u.
        # A  = np.sqrt( 2 * Es.T / self.eigvals  )
        # print(norm(A-Aamplitudes))

        Bamplitudes = self.A2B(A=Aamplitudes)
        
        if hasattr(self,"properties") and "time" in self.properties:
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(Bamplitudes))
        phases = np.arctan2(-proj_vel,proj_displ) - np.outer(np.sqrt( self.eigvals) , time).T
        phases = np.unwrap(phases,discont=0.0,period=2*np.pi)

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

        if MicroStatePrivate.debug :
            test = self.project_on_cartesian_coordinates(Aamplitudes,phases,inplace=False)
            #print(norm(test["positions"] - self.positions))
            print(norm(test["displacements"] - deltaR))
            print(norm(test["velocities"] -v))            

        return out
    
    def save2xyz(self,what,file=None,name=None,folder=None,atoms=None):

        if file is None:
            file = output_file(folder,MicroStatePrivate.ofile[name])
        if atoms is None :
            atoms = self.atoms

        if len(what.shape) == 1 : # just one configuration, NOT correctly formatted

            what = what.reshape((-1,3))
            return self.save2xyz(what,file,name,folder,atoms)
        
        elif len(what.shape) == 2 : 

            if what.shape[1] != 3 : # many configurations
                what = what.reshape((len(what),-1,3))
                return self.save2xyz(what,file,name,folder,atoms)
            
            else : # just one configurations, correctly formatted
                return self.save2xyz(np.asarray([what]),file,name,folder,atoms)

        elif len(what.shape) == 3 :

            Na = what.shape[1]
            if what.shape[2] != 3 :
                raise ValueError("wrong shape")
            
            with open(file,"w") as f :
                
                for i in range(what.shape[0]):
                    pos = what[i,:,:]
                    _atoms = atoms[i] if type(atoms[0]) != str else atoms
                    f.write(str(Na)+"\n")
                    f.write("# configuration {:d}\n".format(i))
                    for ii in range(Na):
                        f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(_atoms[ii],*pos[ii,:]))
            return

    @staticmethod
    def save2txt(what,file=None,name=None,folder=None):
        if file is None:
            file = output_file(folder,MicroStatePrivate.ofile[name])
        print("{:s}saving {:s} to file '{:s}'".format(MicroStatePrivate.tab,name,file))
        np.savetxt(file,what, fmt=MicroStatePrivate.fmt)
        pass

    def savefiles(self,folder,what):

        if what == "proj-on-vib-modes":
            MicroState.save2txt(what=self.energy,name="energy",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["energy"])
            # print("{:s}saving energy to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.energy.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.occupations,name="occupations",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["occupations"])
            # print("{:s}saving occupations to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.occupations.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.phases,name="phases",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["phases"])
            # print("{:s}saving phases to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.phases.T, fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.Aamplitudes,name="A-amplitudes",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["Aamp"])
            # print("{:s}saving A-amplitudes to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.Aamplitudes.T,fmt=MicroStatePrivate.fmt)

            MicroState.save2txt(what=self.Bamplitudes,name="B-amplitudes",folder=folder)
            # file = output_file(folder,MicroStatePrivate.ofile["Bamp"])
            # print("{:s}saving B-amplitudes to file '{:s}'".format(MicroStatePrivate.tab,file))
            # np.savetxt(file,self.Bamplitudes.T,fmt=MicroStatePrivate.fmt)

        pass

    def plot(self,instructions):

        if type(instructions) == dict:
            instructions = Dict2Obj(instructions)

        if "time" in self.properties and self.properties is not None:
            time = convert(self.properties["time"],"time",_from=self.units["time"],_to="atomic_unit")
        else :
            time = np.zeros(len(self.Aamplitudes))

        if instructions.t_min > 0 :            
            print("\tSkipping the {:d} {:s}".format(instructions.t_min,instructions.unit))
            i = np.where( self.properties["time"] >= instructions.t_min )[0][0]
            print("\tthen skipping the first {:d} MD steps".format(i))
            self.Aamplitudes = self.Aamplitudes[i:,:]
            self.energy = self.energy[i:,:] 
            time   = time[i:]

        Ndof = self.Aamplitudes.shape[1]
        normalization = self.energy.sum(axis=1) / ( Ndof - 3 )

        normalized_occupations = np.zeros(self.Aamplitudes.shape)
        for i in range(Ndof):
            normalized_occupations[:,i] = np.square(self.Aamplitudes[:,i])  * self.eigvals[i] / ( 2*normalization[i] )

        fig, ax = plt.subplots(figsize=(10,6))

        factor = unit_to_user("time","picosecond",1)
        time = time*factor
        ax.plot(time,normalized_occupations)

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("time (ps)")
        ax.set_xlim(min(time),max(time))
        ylim = ax.get_ylim()
        #ax.set_ylim(0,ylim[1])
        ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        plt.savefig(instructions.plot)

        ###
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        mean = np.mean(normalized_occupations,axis=0)
        std = np.std(normalized_occupations,axis=0)
        if len(mean) != Ndof or len(std) != Ndof:
            raise ValueError("wrong array size for barplot")

        fig, ax = plt.subplots(figsize=(10,6))
        w = np.sqrt(self.eigvals) * unit_to_user("frequency","thz",1)
        # ax.scatter(x=w,y=mean,color="navy")
        ax.errorbar(x=w,y=mean,yerr=std,color="red",ecolor="navy",fmt="o")

        # plt.title('LiNbO$_3$ (NVT@$20K$,$\\Delta t = 1fs$,T$=20-50ps$,$\\tau=10fs$)')
        ax.set_ylabel("$A^2_s\\omega^2_s / \\left( 2 N \\right)$ with $N=E_{harm}\\left(t\\right)$")
        ax.set_xlabel("$\\omega$ (THz)")
        
        #ax.set_xlim(min(self.time),max(self.time))
        xlim = ax.get_xlim()
        ax.hlines(1.0,xlim[0],xlim[1],linestyle="dashed",color="black",alpha=0.5)
        ax.set_xlim(*xlim)
        # ax.set_yscale("log")

        plt.grid()
        plt.tight_layout()
        tmp = os.path.splitext(instructions.plot)
        file = "{:s}.{:s}{:s}".format(tmp[0],"mean-std",tmp[1])
        # plt.show()
        plt.savefig(file)
        ###
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        df = pd.DataFrame()
        df["w [THz]"] = w
        df["mean"] = mean
        df["std"] = std
        file = file = output_file(instructions.output,MicroStatePrivate.ofile["violin"])
        df.to_csv(file,index=False,float_format="%22.12f")

        pass

    def get_position_template(self,flattened=True):
        if flattened:
            return np.full(self.relaxed.shape,np.nan)
        else :
            return self.get_position_template(flattened=False).reshape((-1,3))
        
    def get_position_template_shape(self,flattened=True):
        if flattened:
            return list(self.relaxed.shape)
        else :
            return list(len(self.relaxed)/3,3)
        
    #@reloading
    def displace_along_normal_mode(self,amp:dict,amptype="B",unit="atomic_unit"):

        
        # convert to amplitudes into atomic_unit
        for k in amp.keys():
            amp[k] *= unit_to_internal("length",unit,1)

        # convert from dict to np.ndarray
        temp = np.zeros(self.Nmodes)
        for k in amp.keys():
            temp[k] = amp[k] 
        amp = temp
        
        phases = np.zeros(amp.shape)

        if amptype == "B":
            Aamp = self.B2A(amp)
        elif amptype == "A":
            Aamp = amp
        else :
            raise ValueError("wrong amplitude type")

        out = self.project_on_cartesian_coordinates(Aamp=Aamp,\
                                                    phases=phases,\
                                                    inplace=False)
        
        return out["positions"].reshape((-1,3))
    
    def generate_thermal_state(self,T,randomfile,N=1,save=True,read=True,unit="kelvin"):

        if unit == "kelvin":
            factor = unit_to_internal("temperature","kelvin",1)
            print("{:s}converting temperature from 'kelvin' to 'atomic units' by multiplication for {:>14.10e}".format(MicroStatePrivate.tab,factor))
            T *= factor

        if N <= 0 :
            raise ValueError("'N' has to be greater than 0")
        
        # set the random number generator state
        if read :
            if not randomfile.endswith(".pickle") :
                randomfile += ".pickle"
        
            if os.path.exists(randomfile):
                try :
                    with open(randomfile, 'rb') as f:
                        state = pickle.load(f)
                    rand.set_state(state)
                except:
                    print("{:s}file '{:s}' supposed to contain the (pseudo-)random number generator state is empty".format(MicroStatePrivate.tab,randomfile))

        # call generate_thermal_state N times
        if N > 1 :
            tmp = self.get_position_template_shape()
            r = np.full(tmp*N,np.nan,dtype=object)
            v = r.copy()

            if hasattr(T,"__len__"):
                if len(T) != N :
                    raise ValueError("'N' and 'T' must have the same length")
            else :
                T = np.full(N,T)

            for i in range(N):
                r[i],v[i] = self.generate_thermal_state(T=T[i],randomfile=None,N=1,save=False,read=False,unit="a.u.")

        else:
            # r = self.get_position_template()
            # v = self.get_position_template()

            # generate random phases
            phases = rand.rand(len(self.relaxed))*2*np.pi

            # generate A-amplitudues according to the Equipartition Theorem
            Amplitudes = np.full(self.Nmodes,0.0) # Nmodes = number degrees of freedom
            #total_energy = self.Nmodes * T # since k_B = 1

            # T = E_tot / N = A^2 * w^2 / 2
            # sqrt( 2 T / w^2 ) = A 
            Amplitudes[3:] = np.sqrt( 2 * T / self.eigvals[3:] )

            out = self.project_on_cartesian_coordinates(Aamp=Amplitudes,\
                                                        phases=phases,\
                                                        inplace=False)
            
            r = out["positions"]
            v = out["velocities"]

            # test = self.project_on_vibrational_modes(deltaR=out["displacements"],v=v,inplace=False)
                

        if save:
            with open(randomfile, 'wb') as f:
                print("{:s}Saving the (pseudo-)random number generator state to file '{:s}'".format(MicroStatePrivate.tab,randomfile))
                s = rand.get_state()
                pickle.dump(s, f)

        return r,v
    
    def get_properties_as_dataframe(self):

        import pandas as pd
        p = pd.DataFrame(data=self.properties,columns=self.header)
        return p
    
    def show(self):
        '''show the attributes of the class'''
        
        print("Attributes of the object:")
        attribs = get_attributes(self)
        for a in attribs:
            print("{:s}".format(MicroStatePrivate.tab),a)

    def show_properties(self):
        '''show the properties of the class'''

        print("Properties of the object:")
        keys = list(self.properties.keys())
        size = [None]*len(keys)
        for n,k in enumerate(keys):
            tmp = list(self.properties[k].shape[1:])
            if len(tmp) == 0 :
                size[n] = 1
            elif len(tmp) == 1:
                size[n] = tmp[0]
            else :
                size[n] = tmp
        df = pd.DataFrame(columns=["name","unit","shape"])
        df["name"] = keys
        df["unit"] = [ self.units[k] for k in keys ]
        df["shape"] = size
        return df
    
    families = {    "energy"        :["conserved","kinetic_md","potential"],
                    "polarization"  :["totalpol"],
                    "dipole"        :["electric-dipole"],
                    "time"          :["time"]
                }
    
    @staticmethod
    def search_family(what):
        for k in MicroState.families:
            if what in MicroState.families[k]:
                return k
        else :
            raise ValueError('family {:s} not found. \
                             But you can add it to the "MicroState.families" dict :) \
                             to improve the code '.format(what))

    def convert_property(self,what,unit,family=None,inplace=True):
        if family is None:
            family = self.search_family(what)
        factor = convert(1,family,_from=self.units[what],_to=unit)
        if inplace :
            self.properties[what] = self.properties[what] * factor
            self.units[what] = unit
            return self.properties[what]
        else :
            return self.properties[what] * factor
        
    def remove_data(self,time_end,time_start=0,unit="atomic_unit"):

        # convert time into internal units
        factor = convert(1,"time",_from=unit,_to=self.units["time"])
        time_end   *= factor
        time_start *= factor

        time = self.properties["time"]
        index = np.logical_and(time >= time_start , time < time_end)
        new = deepcopy(self)

        for k in self.properties.keys():
            new.properties[k] = self.properties[k][index]

        other = ["positions","velocities","displacements"]
        for k in other:
            if hasattr(self,k):
                new_attr = getattr(self,k)[index]
                setattr(new,k,None)
                setattr(new,k,new_attr)

        new.Nconf = len(new)
        
        return new 

    def derivative(self,what,wrt="time",saved_as=None,*argc,**argv):

        df = np.diff(self.properties[what],axis=0)
        dt = np.diff(self.properties[wrt],axis=0)

        dt = dt.append(dt[-1])
        dt = dt.append(dt[-1])

        der = np.outer(df,1.0/dt) 

        if saved_as is not None:
            self.add_property(name=saved_as,array=der,*argc,**argv)
        
        return der

    def add_property(self,name,array,overwrite=False):
        if len(self) != 0:
            if len(self) != len(array):
                raise ValueError("array with wrong length")
        if name in self.properties:
            print("Warning: '{:s}' in properties".format(name))
            if overwrite :
                print("Warning: '{:s}' will be overwritten".format(name))
            else :
                print("Warning: to overwrite '{:s}' set 'overwrite' = True ".format(name))
        self.properties[name] = array
        pass

    def __len__(self):
        if hasattr(self,"properties"):
            k = next(iter(self.properties.keys()))
            return len(self.properties[k])
        else :
            return 0

    # def __repr__(self)->None:
    #     self.show()

    # @reloading
    def vibrational_analysis_summary(self)->pd.DataFrame:
        """ summary of the vibrational analysis"""
        print("Summary of the vibrational analysis:")
        #cols = [ "eigvals [a.u.]" , "w [a.u.]", "w [THz]", "w [cm^-1]", "T [a.u.]", "T [ps]","E [a.u.]", "n [a.u.]"]
        df = pd.DataFrame()
        eigvals = self.eigvals.copy()
        eigvals [ eigvals == MicroStatePrivate.smallest_float ] = 0.0
        df["eigvals [a.u.]"] = eigvals
        df["w [a.u.]"]  = [ np.sqrt(i) if i > 0. else None for i in eigvals ]
        df["w [THz]"]   = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="thz")
        df["w [cm^-1]"] = convert(df["w [a.u.]"],"frequency",_from="atomic_unit",_to="inversecm")
        df["T [a.u.]"]  = 2*np.pi / df["w [a.u.]"]
        df["T [ps]"]    = convert(df["T [a.u.]"],"time",_from="atomic_unit",_to="picosecond")

        if hasattr(self,"energy"):
            df["E [a.u.]"]  = self.energy.mean(axis=0)

        if hasattr(self,"occupations"):
            df["n [a.u.]"]  = self.occupations.mean(axis=0)

        return df

    # @reloading
    def to_ase(self,inplace=False,recompute=False)->Atoms:

        out = None
        if recompute or not hasattr(self,"ase"):
            out = [None]*self.Nconf
            N = np.arange(len(out))
            for n,t,p,c in zip(N,self.types,self.positions,self.cell):
                out[n] = Atoms(symbols=t, positions=p.reshape(-1,3), cell=c.T, pbc=True)

        if inplace and out is not None:
            self.ase = out
        elif hasattr(self,"ase"):
            out = self.ase

        return out

    # @reloading
    @staticmethod
    def save(obj,file)->None:
        print("Saving object to file '{:s}'".format(file))
        with open(file, 'wb') as f:
            pickle.dump(obj,f)
        pass

    # @reloading
    @staticmethod
    def load(file):
        print("Loading object from file '{:s}'".format(file))
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    @staticmethod
    def _cart2lattice(array,lattice,matrixT=None,get_matrix=False): #,*argc,**argv):
        """ Cartesian to lattice coordinates
        
        Input:
            array: array in cartesian to be converted, shape (N,3)
            lattice: lattice parameters, 
                where the i^th basis vector is stored in the i^th columns
                (it's the opposite of ASE, QE, FHI-aims)
                lattice : 
                    | a_1x a_2x a_3x |
                    | a_1y a_2y a_3y |
                    | a_1z a_2z a_3z |
        Output:
            array in fractional coordinates, shape (N,3)

        Notes:
            R = C @ X   -->   X = C^{-1} @ R
            R are the cartesian coordinates of 1 configuration, shape (3,1)
            X are the fractional coordinates of 1 configuration, shape (3,1)
            C are the lattice parameters, shape (3,3)
        """
        # if array.shape[1] != 3 :
        #     raise ValueError("array with wrong shape:",array.shape)
        if lattice.shape != (3,3):
            raise ValueError("lattice with wrong shape:",lattice.shape)
        
        # speed-up if I have many coordinates but the same lattice
        if matrixT is None:
            # I have to divide normalize the lattice parameters
            length = np.linalg.norm(lattice,axis=1)
            matrixT = deepcopy(lattice)
            # normalize the columns
            for i in range(3):
                matrixT[:,i] /= length[i]
            matrixT = np.linalg.inv( matrixT ).T

        # I should do 
        #   return ( inv @ array.T ).T
        # but it is equal to array @ matrix.T
        out = array @ matrixT
        if get_matrix:
            return out, matrixT
        else :
            return out
    
    @staticmethod
    def _lattice2cart(array,lattice,matrixT=None): #,*argc,**argv):
        """ Lattice to Cartesian coordinates
        
        Input:
            array: array in fractional to be converted, shape (N,3)
            lattice: lattice parameters, 
                where the i^th basis vector is stored in the i^th columns
                (it's the opposite of ASE, QE, FHI-aims)
                lattice : 
                    | a_1x a_2x a_3x |
                    | a_1y a_2y a_3y |
                    | a_1z a_2z a_3z |
        Output:
            array in cartesian coordinates, shape (N,3)

        Noted:
            R = C @ X
            R are the cartesian coordinates of 1 configuration, shape (3,1)
            X are the fractional coordinates of 1 configuration, shape (3,1)
            C are the lattice parameters, shape (3,3)
        """
        # if array.shape[1] != 3 :
        #     raise ValueError("array with wrong shape:",array.shape)
        if lattice.shape != (3,3):
            raise ValueError("lattice with wrong shape:",lattice.shape)
        
        # I should do 
        #   return ( inv @ array.T ).T
        # but it is equal to

        if matrixT is None:
            length = np.linalg.norm(lattice,axis=1)
            matrixT = deepcopy(lattice)
            # normalize the columns
            for i in range(3):
                matrixT[:,i] /= length[i]
            matrixT = matrixT.T

        return array @ matrixT
    
    def cart2lattice(self,*argc,**argv):
        return self.trasform_basis(func=MicroState._cart2lattice,*argc,**argv)
    
    def lattice2cart(self,*argc,**argv):
        return self.trasform_basis(func=MicroState._lattice2cart,*argc,**argv)
    
    def trasform_basis(self,func,what=None,array=None,unit=None,family=None,same_lattice=True,reshape=None):

        if same_lattice:

            from .functions import print_cell
            
            lattice = deepcopy(self.cell[0])
            print_cell(lattice)

            if array is None :
                array = self.properties[what]

            if unit is None :
                unit = self.units[what]

            factor = convert(1,_from=unit,_to="atomic_unit",family=family)
            array *= factor 
            
            # print(" array shape:",array.shape)

            out = np.zeros(array.shape)
            matrixT = None
            for n,pol in enumerate(array):
                if reshape is not None:
                    pol = pol.reshape(reshape)
                p = func(pol,lattice,matrixT,get_matrix=n==0)
                if type(p) == tuple and n == 0 :
                    matrixT = p[1]
                    p = p[0]
                out[n,:] = p.reshape(pol.shape)

            factor = convert(1,family=family,_from="atomic_unit",_to=unit)
            return out * factor
    
        else :
            raise ValueError("not implemented yet")
    
    def get_volume(self,same_lattice,only_first):
        
        if only_first and not same_lattice:
            raise ValueError("'only_first' == True can be used only if 'same_lattice' == True too")
        
        if same_lattice:

            lattice = self.cell[0]
            if only_first :
                volume = np.linalg.det(lattice)
            else:
                volume = np.full(len(self),np.linalg.det(lattice))
                #length = np.full(len(polarization),np.linalg.norm(lattice,axis=0))

        else:
            
            volume = np.zeros(len(self))
            #length = np.zeros((len(polarization),3))

            for n in range(len(self)):
                lattice = self.cell[n]
                volume[n] = np.linalg.det(lattice)
                #length[n] = np.linalg.norm(lattice,axis=0)

        return volume

    def get_basisvectors_length(self,same_lattice,only_first):

        if only_first and not same_lattice:
            raise ValueError("'only_first' == True can be used only if 'same_lattice' == True too")
            
        if same_lattice:

            lattice = self.cell[0]
            if only_first :
                length = np.linalg.norm(lattice,axis=0)
            else :
                #volume = np.full(len(self),np.linalg.det(lattice))
                length = np.full((len(self),3),np.linalg.norm(lattice,axis=0))

        else:
            
            length = np.zeros((len(self),3))

            for n in range(len(self)):
                lattice = self.cell[n]
                #volume[n] = np.linalg.det(lattice)
                length[n] = np.linalg.norm(lattice,axis=0)

        return length

    def get_phases(self,array=None,unit="atomic_unit",same_lattice=True):
        """Compute the phases of the polarization vectors"""

        # convert the polarization from cartesian di lattice coordinates
        if array is None:
            polarization = self.cart2lattice(what="totalpol",\
                                            family="polarization",\
                                            same_lattice=same_lattice,\
                                            reshape=None)
        else :
            polarization = self.cart2lattice(array=array,\
                                            unit=unit,\
                                            family="polarization",\
                                            same_lattice=same_lattice,\
                                            reshape=None)
            
        # compute the volume and the norm of the lattice vectors
        volume = self.get_volume(same_lattice=same_lattice,only_first=False)
        length = self.get_basisvectors_length(same_lattice=same_lattice,only_first=False)
        
        # compute the phases with periodicity 1
        phases = np.zeros(polarization.shape)
        for xyz in range(3):
            phases[:,xyz] = polarization[:,xyz]*volume[:]/(length[:,xyz]) # e == 1 in a.u
            phases[:,xyz] = np.unwrap(phases[:,xyz],discont=0.0,period=1.0)
        
        return phases
    
    def get_dipole(self,same_lattice=True,inplace=True):

        volume = self.get_volume(same_lattice=same_lattice,only_first=False)
        polarization = self.properties["totalpol"]

        dipole = (polarization.T * volume).T
        if inplace :
            self.properties["electric-dipole"] = dipole

        return dipole

    def get_polarization_quantum(self,same_lattice=True,only_first=True):

        if only_first:

            volume = self.get_volume(same_lattice=True,only_first=only_first)
            length = self.get_basisvectors_length(same_lattice=True,only_first=only_first)

            quantum = np.zeros(3)
            quantum[:] = length[:]/volume          
            quantum = self.lattice2cart(array=quantum.reshape((1,3)),\
                                            family="polarization",\
                                            unit=self.units["totalpol"],\
                                            same_lattice=True,\
                                            reshape=None).flatten()
            
        else :
            volume = self.get_volume(same_lattice=same_lattice)
            length = self.get_basisvectors_length(same_lattice=same_lattice)

            quantum = np.zeros((len(volume),3))
            for xyz in range(3):            
                quantum[:,xyz] = length[:,xyz]/volume


            quantum = self.lattice2cart(array=quantum,\
                                            family="polarization",\
                                            unit=self.units["totalpol"],\
                                            same_lattice=same_lattice,\
                                            reshape=None)

        return quantum

    def fix_polarization(self,\
                         array=None,\
                         unit=None,\
                         same_lattice=True,
                         inplace=False):
        """ Fix polarization jumps
        
            Attention:
                - the lattice parameters have to be in atomic_unit
                - the polarization is automatically converted into atomic_unit,
                    so you can not to worry about its measure unit
        """
        if array is None:
            array = self.properties["totalpol"]
            if unit is None :
                unit = self.units["totalpol"]

        phases = self.get_phases(array=array,unit=unit,same_lattice=same_lattice)

        #
        # the phases are now fixed in 'get_phases'
        #
        # # cycle over the 3 directions
        # for xyz in range(3): 
        #     # unwrap since the phases are defined modulo 1
        #     # i.e. phi = phi + n 
        #     # with n integer
        #     phases[:,xyz] = np.unwrap(phases[:,xyz],discont=0.0,period=1.0)
            
        #
        volume = self.get_volume(same_lattice=same_lattice,only_first=False)
        length = self.get_basisvectors_length(same_lattice=same_lattice,only_first=False)
        polarization = np.zeros((len(phases),3))
        for xyz in range(3):            
            polarization[:,xyz] = phases[:,xyz]*length[:,xyz]/volume[:]
        
        polarization = self.lattice2cart(array=polarization,\
                                         family="polarization",\
                                         unit=unit,\
                                         same_lattice=same_lattice,\
                                         reshape=None)

        if inplace :
            self.properties["totalpol"] = polarization

        return polarization
    
    def all_types(self):
        """Return a list of all types of atoms"""
        return np.unique(self.types)
    
    def atomic_numbers(self):
        """Return a list of all types of atoms"""
        return self.numbers


def main():

    from miscellaneous.elia.classes import MicroState
    from miscellaneous.elia.functions import segment
    import matplotlib.pyplot as plt
    from itertools import product    
    import pandas as pd
    from ase.io import read

    inverted = read("inverted.xyz").positions
    standard = read("standard.xyz").positions
    symbols = read("standard.xyz").get_chemical_symbols()
    print("inverted:\n",inverted,"\n")
    print("standard:\n",standard)

    N = 50
    sequence = segment(standard,inverted,N)

    #sequence = np.asarray([ i.positions for i in read("sequence.xyz",":")])
    # standard = sequence[0]
    # inverted = sequence[-1]


    data = MicroState()
    data.save2xyz(what=sequence,file="test.xyz",atoms=symbols)


def _main():


    xyz = 0

    fig,ax = plt.subplots()

    instructions = {"properties" : "not-gauge-fixed/i-pi.properties.out",\
               "velocities":"not-gauge-fixed/i-pi.velocities_0.xyz",\
               "cells" : "not-gauge-fixed/i-pi.positions_0.xyz"}
    self = MicroState(instructions)
    self.show()
    self.show_properties()
    #pol = self.fix_polarization()
    phases = self.get_phases()

    y = phases[:,xyz]
    ax.scatter(np.arange(len(y)),y,color="blue",label="not fixed gauge",s=0.1)

    xlims = ax.get_xlim()


    # ax.scatter(np.diff(np.linalg.norm(self.velocities,axis=1)),np.diff(np.linalg.norm(phases,axis=1)),color="red",label="new",s=0.1)

    # fig,ax = plt.subplots()

    instructions = {"properties" : "gauge-fixed/i-pi.properties.out",\
               "velocities":"gauge-fixed/i-pi.velocities_0.xyz",\
               "cells" : "gauge-fixed/i-pi.positions_0.xyz"}
    self = MicroState(instructions)
    self.show()
    self.show_properties()
    #pol, quantum, phases, old_phases = self.fix_polarization()
    phases = self.get_phases()

    
    #x = np.arange(len(pol)-1)

    y = phases[:,xyz]
    ax.scatter(np.arange(len(y)),y,color="red",label="fixed gauge",s=0.1)


    ax.set_xlim(*xlims)



    # y = old_phases[:,xyz]
    # ax.plot(np.arange(len(y)),y,color="blue",label="old")
    m = phases[0,xyz]
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    for i in range(-10,10):
        plt.hlines(m+i,xlims[0],xlims[1],color="black",linestyle="dashed",alpha=0.5)
    ax.set_ylim(ylims)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("\n\tJob done :)\n")

# if __name__ == "__main__":
#     main()


# def _main():

#     instructions = {"modes":"vib","relaxed":"start.xyz","types":"start.xyz"}
#     what = "vib"
#     toread = ["relaxed","types"]
#     data = MicroState(instructions=instructions,toread=toread,what=what)

#     df = data.vibrational_analysis_summary()
#     print(df)

#     x = 2.0
#     displ_P  = np.linspace(-x,x,11)
#     displ_IR = np.linspace(-x,x,11)

#     from itertools import product    
#     import pandas as pd
#     displ = np.asarray(list(product(displ_P, displ_IR)))
#     df = pd.DataFrame(columns=["P","IR"],index=np.arange(len(displ)))
  
#     positions = np.zeros((len(displ),*data.relaxed.reshape(-1,3).shape))
#     for n,(a,b) in enumerate(displ):
#         amplitudes = {8:a,26:b}    
#         df.at[n,"P"] = a
#         df.at[n,"IR"] = b
#         positions[n] = data.displace_along_normal_mode(amplitudes)

#     data.save2xyz(positions,file="sequence.xyz",atoms=data.types[0])
#     df.to_csv("info.csv",index=False)


#     print("\n\tJob done :)\n")