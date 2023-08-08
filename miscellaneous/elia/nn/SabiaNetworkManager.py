from torch_geometric.data import Data
import torch
from copy import copy
import numpy as np
from scipy.stats import spearmanr
from torch.nn import MSELoss
import time
#import jax
#import jax.numpy as jnp
from .SabiaNetwork import SabiaNetwork
from miscellaneous.elia.nn.water.make_dataset import make_datapoint
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetworkManager')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.

__all__ = ["SabiaNetworkManager"]

class SabiaNetworkManager(SabiaNetwork):

    output:str
    lattice: torch.Tensor
    radial_cutoff: float
    _radial_cutoff: float
    symbols: list
    x: Data
    R:torch.Tensor

    def __init__(self: T, radial_cutoff: float = 0.0, output: str = "ED", **kwargs) -> None:
        super(SabiaNetworkManager, self).__init__(**kwargs)

        self.output = output
        if self.output not in ["E", "D", "ED", "EF", "EDF"]:
            raise ValueError("'output' must be 'E', 'D', 'EF', 'ED' or 'EDF'")

        #self.grad = None
        #self.bec = None
        self.lattice = torch.Tensor()
        self.radial_cutoff = 0.0
        self._radial_cutoff = radial_cutoff
        self.symbols = list()
        self.X = Data()
        self.R = torch.Tensor()
        pass

    # def train(self: T, mode: bool) -> T:
    #     if self.grad is not None:
    #         del self.grad     # delete the gradient
    #         self.grad = None  # build an empty gradient
    #     return super(SabiaNetworkManager, self).train(mode)

    # @torch.jit.script
    def PES(self: T, R) -> torch.tensor:
        """Potential Energy Surface
        Input:
            - R: (N,3) tensor of positions
        Output:
            - potential energy
        Comments:
            - this methods will be automatically differentiated to get the forces"""
        # x = make_datapoint( lattice=self.lattice,\
        #                     radial_cutoff=self.radial_cutoff,\
        #                     symbols=self.symbols,\
        #                     positions=R)#,
        #                     #fake = R.detach() if hasattr(R,"detach")  else R)
        self.X.pos = R
        y = self(self.X)
        y = y if self.output == "E" else y[:,0]
        return y#.reshape(())
    
    def pol(self: T, R) -> torch.tensor:
        self.X.pos = R.reshape((-1,3))
        y = self(self.X)
        if self.output == "D":
            return y
        elif self.output in ["ED","EDF"]:
            return y[:,1:4]
        
    def dipole(self: T, X: Data,requires_grad:bool=True) -> torch.tensor:
        """Electric dipole
        Input:
            - R: (N,3) tensor of positions
        Output:
            - dipole
        """

        if self.output not in ["D","ED","EDF"]:
            raise ValueError("'dipole' not present in the output of this torch.nn.Module")
        
        batch_size = len(np.unique(X.batch))
        y = torch.zeros((batch_size,3), requires_grad=requires_grad)

        for n in range(batch_size):

            # prepare data
            x,R = self._prepare(X,n)
        
            tmp = self(self.X)

            if self.output == "D":
                y.data[n,:] = tmp
            elif self.output in ["ED","EDF"]:
                y.data[n,:] = tmp[:,1:4]

        return y

    # def _prepare(self: T, X: Data, n:int,rotate=None,replace=None,**argv)->(Data,torch.tensor):
    #     """prepare data """

    #     index = X.batch == n
    #     self.lattice = X.lattice[n]
    #     self.symbols = X.symbols[n]
    #     self.radial_cutoff = float( X.radial_cutoff[n] if hasattr(X, "radial_cutoff") else self._radial_cutoff)
    #     self.R = X.pos[index]

    #     # replace values
    #     if replace is not None:
    #         if "lattice" in replace :
    #             self.lattice = replace["lattice"]
    #         if "symbols" in replace :
    #             self.symbols = replace["symbols"]
    #         if "radial_cutoff" in replace :
    #             self.radial_cutoff = replace["radial_cutoff"]
    #         if "R" in replace :
    #             self.R = replace["R"]

    #     # rotate, useful for 'check_equivariance'
    #     if rotate is not None :          
    #         self.R        = torch.einsum("ij,zj->zi",rotate,self.R.reshape((1,-1))).reshape((-1,3))
    #         self.lattice  = torch.einsum("ij,zj->zi",rotate,self.lattice.reshape((1,-1))).reshape((-1,3))

    #     # create the Data object
    #     self.X = make_datapoint(lattice=self.lattice,
    #                     radial_cutoff=self.radial_cutoff,
    #                     symbols=self.symbols,
    #                     positions=self.R,
    #                     default_dtype=self.default_dtype,
    #                     **argv)
        
    #     return self.X, self.R

    def _prepare(self: T, X: Data, n:int,rotate=None,replace=None,**argv)->(Data,torch.tensor):
        """prepare data """

        index = X.batch == n
        lattice = X.lattice[n]
        symbols = X.symbols[n]
        radial_cutoff = float( X.radial_cutoff[n] if hasattr(X, "radial_cutoff") else self._radial_cutoff)
        R = X.pos[index]

        # replace values
        if replace is not None:
            if "lattice" in replace :
                lattice = replace["lattice"]
            if "symbols" in replace :
                symbols = replace["symbols"]
            if "radial_cutoff" in replace :
                radial_cutoff = replace["radial_cutoff"]
            if "R" in replace :
                R = replace["R"]

        # rotate, useful for 'check_equivariance'
        if rotate is not None :          
            R        = torch.einsum("ij,zj->zi",rotate,R.reshape((1,-1))).reshape((-1,3))
            lattice  = torch.einsum("ij,zj->zi",rotate,lattice.reshape((1,-1))).reshape((-1,3))

        # create the Data object
        X = make_datapoint(lattice=lattice,
                        radial_cutoff=radial_cutoff,
                        symbols=symbols,
                        positions=R,
                        default_dtype=self.default_dtype,
                        **argv)
        
        return X, R

    def energy(self: T, X: Data,requires_grad:bool=True) -> torch.tensor:
        batch_size = len(np.unique(X.batch))
        y = torch.zeros(batch_size, requires_grad=requires_grad)#.reshape((batch_size, -1))

        for n in range(batch_size):

            # index = X.batch == n
            # self.lattice = X.lattice[n]
            # self.symbols = X.symbols[n]
            # self.radial_cutoff = X.radial_cutoff if hasattr(
            #     X, "radial_cutoff") else self._radial_cutoff
            # if batch_size == 1 :
            #     self.radial_cutoff = float(self.radial_cutoff)
            # self.R = X.pos[index]#.flatten()
            # 
            # self.X = make_datapoint(lattice=self.lattice,
            #                         radial_cutoff=self.radial_cutoff,
            #                         symbols=self.symbols,
            #                         positions=self.R,
            #                         default_dtype=self.default_dtype)

            # prepare data
            x,R = self._prepare(X,n)
            
            tmp = self.PES(R)
            y.data[n] = tmp

        return y

    def forces(self: T, X: Data) -> torch.tensor:

        batch_size = len(np.unique(X.batch))
        y = torch.zeros(X.pos.shape, requires_grad=True).reshape((batch_size, -1))

        # X.pos.requires_grad_(True)
        # X.lattice.requires_grad_(False)
        # del X.edge_vec

        # del X.edge_vec
        # x,R = self._prepare(X)
        # forces = torch.autograd.functional.jacobian(self.PES,X.pos)

        # energy = self.forward(X)
        # energy.backward(retain_graph=True)
        # forces = X.pos.grad
        # return forces

        for n in range(batch_size):

            # index = X.batch == n
            # self.lattice = X.lattice[n]
            # self.symbols = X.symbols[n]
            # self.radial_cutoff = X.radial_cutoff if hasattr(
            #     X, "radial_cutoff") else self._radial_cutoff
            # R = X.pos[index]#.flatten()
            # 
            # self.X = make_datapoint(lattice=self.lattice,
            #                         radial_cutoff=self.radial_cutoff,
            #                         symbols=self.symbols,
            #                         positions=R,
            #                         default_dtype=self.default_dtype)

            # prepare data
            
            x,R = self._prepare(X,n)
            self.X = x
            
            # 
            # 'self.forward' does not depend on the positions 'X.pos'
            # but only on the reltive positions 'self.X.edge_vec'
            # we need to be sure that 'X.pos' and 'self.X.edge_vec' are "related"
            # this means that we need to set 'R.requires_grad_(True)'
            # and then recompute 'self.X.edge_vec'
            # This will be done in 'self.preprocess'
            # 
            R.requires_grad_(True)
            if hasattr(self.X,"edge_vec"):
                del self.X.edge_vec

            # the mapping between x.pos and R has to be done inside self.PES
            # energy = self(x)
            # energy.backward()
            # forces = x.pos.grad
            # print(x)
            # print(R)
            # print(forces)
            # R.requires_grad_(True)

            # ATTEMPT 1)
            # https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
            #
            # Function that computes the Jacobian of a given function.
            #
            # strategy (str, optional) – Set to "forward-mode" or "reverse-mode" to determine whether
            # the Jacobian will be computed with forward or reverse mode AD.
            # Currently, "forward-mode" requires vectorized=True. Defaults to "reverse-mode".
            # If func has more outputs than inputs, "forward-mode" tends to be more performant.
            # Otherwise, prefer to use "reverse-mode".
            # The default is "reverse-mode"
            #
            # Comments:
            # The jacobian function seems to be computed every time, even though 'create_graph=True'
            # should make the computation faster.
            #
            #---- code starts here ----#
            if True:
                
                # print("Ciao")
                #start = time.process_time()
                #R = X.pos[index].requires_grad_(True)#.flatten()
                tmp = torch.autograd.functional.jacobian(self.PES, R,create_graph=True)
                #print("TIME:",time.process_time() - start)

                # dR = torch.zeros((3,3))
                # dR[0,0] = 0.05 
                # check = ( self.PES(R+dR) - self.PES(R-dR) ) / 0.1
                y.data[n, :] = tmp.flatten()
                # create_graph=self.training
            # ATTEMPT 2)
            # https://pytorch.org/docs/stable/generated/torch.func.grad.html
            #
            # grad operator helps computing gradients of func with respect to the input(s) specified by argnums.
            # This operator can be nested to compute higher-order gradients.
            #
            # Comments:
            # In this way we should get a callable function ... but it's not working since I got the following error:
            # 'Cannot access data pointer of Tensor that doesn't have storage'
            #
            #---- code starts here ----#
            if False:
                # R = X.pos[index]#.requires_grad_(True)#.flatten()
                if self.grad is None:
                    self.grad = torch.func.grad(self.PES)
                tmp = self.PES(R)
                tmp.backward()
                print(R.grad)
                y.data[n, :] = self.grad(R).flatten()

            # ATTEMPT 3)
            # https://www.kaggle.com/code/goktugguvercin/gradients-and-jacobians-in-jax
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html#jax.jacfwd
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html#jax.jacrev
            #
            # At this point, JAX provides us with two different transformers 'jax.jacfwd()' and 'jax.jacrev()'
            # to be able to compute jacobian matrices of vector-valued functions:
            # As it is understood from their names, the former uses forward-mode automatic differentation
            # to construct entire jacobian, the latter does same thing with reverse-mode.
            # Naturally, we ask this question to ourselves:
            # If we have two different functions to compute same jacobian, which one do we opt for?
            # Their computational efficiency tends to change depending on the dimension of domain and codomain.
            # In the case of m>>n, we end up with a tall jacobian, which is computed by 'jacfwd()'
            # in much more efficient way than 'jacfwd()'.
            # In opposite case, it is obvious that reverse accumulation does a better job than forward mode.
            # For the square jacobians, 'jacfwd()' seems like it has an advantage over 'jacrev()'.
            # Their signature are almost same as 'grad()', which means that they take a function to be
            # totally differentiated as an argument and works as a transformer to return a new function
            # in responsible for computing jacobian for any input.
            # In other words, they don't directly compute jacobian matrices, instead they return
            # the function that can do it for any point where function 'g' will be differentiated.
            # Also note that, as in 'grad()', both of these jacobian transformers enable us to specify which
            # one of input variables the jacobian will be calculated with respect to; the parameter  argnums is also usable for them.
            #
            # Comments:
            #
            #
            #---- code starts here ----#
            if False:
                R = jnp.asarray(X.pos[index])#.flatten()
                if self.grad is None:
                    self.grad = jax.grad(self.PES, argnums=0)
                y.data[n, :] = self.grad(R)

        return y
    
    def BEC(self: T, X: Data,requires_grad=False,*argc,**argv) -> torch.tensor:

        batch_size = len(np.unique(X.batch))
        # y = torch.zeros((batch_size, 3,X.pos.shape[0]*X.pos.shape[1]/batch_size), requires_grad=requires_grad)
        y = torch.zeros((3,X.pos.shape[0]*X.pos.shape[1]), requires_grad=requires_grad)
        y = y.reshape((batch_size,3,-1))

        for n in range(batch_size):

            # prepare data
            x,R = self._prepare(X,n)
                   
            
            tmp = torch.autograd.functional.jacobian(self.pol, self.R.flatten(),*argc,**argv)
            #self.bec = torch.func.grad(self.pol)
            #tmp = self.bec(self.R)
            y.data[n, :] = tmp

        return y

    @staticmethod
    def get_pred(model: T, X: Data) -> torch.tensor:
        """return Energy, Polarization and Forces"""

        N = {"E": 1, "D": 3, "ED": 4, "EF": 1+3*X.Natoms[0], "EDF": 1+3+3*X.Natoms[0]}
        N = N[model.output]
        batch_size = len(np.unique(X.batch))
        y = torch.zeros((batch_size, N))

        if model.output in ["E", "ED","D"]:
            y = model(X)

        elif model.output == "EDF":
            EP = model(X)
            y[:, 0] = EP[:, 0]         # 1st column for the energy
            y[:, 1:4] = EP[:, 1:4]     # 2nd to 4th columns for the dipole
            y[:, 4:] = model.forces(X) # other columns for the forces

        elif model.output == "EF":
            EP = model(X)
            y[:, 0]  = EP[:, 0]         # 1st column for the energy
            y[:, 1:] = model.forces(X)  # other columns for the forces

        return y

    @staticmethod
    def get_real(X: Data, output: str = "E") -> torch.tensor:
        """return Energy, Polarization and Forces"""

        # 'EPF' has to be modified in case we have different molecules in the dataset
        N = {"E": 1, "EF": 1+3*X.Natoms[0], "D": 3, "ED": 4, "EDF": 1+3+3*X.Natoms[0]}
        N = N[output]
        batch_size = len(np.unique(X.batch))

        # if batch_size > 1 :

        y = torch.zeros((batch_size, N))

        if output in ["E", "EF", "ED", "EDF"]:
            y[:, 0] = X.energy

            if output in ["ED", "EDF"]:
                y[:, 1:4] = X.dipole.reshape((batch_size, -1))

            elif output == "EDF":
                y[:, 4:] = X.forces.reshape((batch_size, -1))

            if output == "EF":
                y[:, 1:] = X.forces.reshape((batch_size, -1))

        elif output == "D":
            y[:,0:3] = X.dipole.reshape((batch_size, -1))

        return y

    def loss(self:T,lE:float=None,lF:float=None,lP:float=None)->callable:

        lE = lE if lE is not None else 1.0
        lF = lF if lF is not None else 1.0
        lP = lP if lP is not None else 1.0

        if self.output in ["E","D"]:
            return MSELoss() #MSELoss(reduction='mean') # MSELoss(reduce='sum')
            #return lambda x,y: MSELoss()(x,y)
        
        elif self.output == "ED":
            def loss_EP(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                P = MSELoss()(x[:,1:4],y[:,1:4])
                return lE * E + lP * P
            return loss_EP
        
        elif self.output == "EF":
            def loss_EF(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                F = MSELoss()(x[:,1:],y[:,1:])
                return lE * E + lF * F
            return loss_EF
        
        elif self.output == "EDF":
            def loss_EPF(x,y):
                E = MSELoss()(x[:,0],y[:,0])
                P = MSELoss()(x[:,1:4],y[:,1:4])
                F = MSELoss()(x[:,4:],y[:,4:])
                return lE * E + lP * P + lF * F
            return loss_EPF
        
        else :
            raise ValueError("error in output mode")
        
    @staticmethod
    def correlation(x:torch.tensor,y:torch.tensor):

        #N = {"E": 1, "P": 3, "EP": 4, "EPF": 1+3+3*X.Natoms[0]}

        x = x.detach().numpy()
        y = y.detach().numpy()

        N = x.shape[1]
        out = np.zeros(N)
        for i in range(N):
            out[i] = spearmanr(x[:,i],y[:,i]).correlation

        return out

    def check_equivariance(self:T,X: Data,angles=None)->np.ndarray:

        from e3nn import o3

        with torch.no_grad():

            batch_size = len(np.unique(X.batch))
            y = np.zeros(batch_size)

            irreps_out = o3.Irreps(self.irreps_out)

            if angles is None :
                angles = o3.rand_angles()

            for n in range(batch_size):

                irreps_in = "{:d}x1o".format(int(X.Natoms[n]))
                irreps_in = o3.Irreps(irreps_in)

                # prepare data -> prepare self.R and self.X

                # rotate output
                x,R = self._prepare(X,n)
                R_out = irreps_out.D_from_angles(*angles)
                y0 = self(x)
                rot_out = torch.einsum("ij,zj->zi",R_out,y0)

                # rotate input
                R_in  = irreps_in.D_from_angles(*angles)
                x,R = self._prepare(X,n,R_in)
                #R = R.reshape((1,-1))

                
                
                # R = x.pos.reshape((1,-1))                
                # rot_in  = torch.einsum("ij,zj->zi",R_in,R)

                # l = x.lattice.reshape((1,-1))
                # l_in  = torch.einsum("ij,zj->zi",R_in,l)
                
                # x.pos     = rot_in.reshape((-1,3))
                # x.lattice = l_in.reshape((-1,3))

                #x,R_ = self._prepare(X,n,replace_pos=rot_in.reshape((-1,3)))
                out_rot = self(x)

                thr = torch.norm(rot_out - out_rot)
                
                y[n] = thr
        
        return y

def main():

    import os
    from miscellaneous.elia.classes import MicroState
    from miscellaneous.elia.nn.water.make_dataset import make_dataset
    from miscellaneous.elia.nn import _make_dataloader

    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)

    # Changing the current working directory
    os.chdir('./water/')

    radial_cutoff = 6.0

    ##########################################

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/microstate.pickle"

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "data/i-pi.positions_0.xyz"
        instructions = {"properties" : "data/i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"data/i-pi.forces_0.xyz"}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ########################################## 

    RESTART = False 
    READ = True
    SAVE = False
    savefile = "data/dataset"

    if not READ or not os.path.exists(savefile+".train.torch") or RESTART :
        print("building dataset")

        if os.path.exists(savefile+".torch"):
            dataset = torch.load(savefile+".torch")
        else :
            dataset = make_dataset( data=data,radial_cutoff=radial_cutoff)

        # train, test, validation
        #p_test = 20/100 # percentage of data in test dataset
        #p_val  = 20/100 # percentage of data in validation dataset
        n = 2000
        i = 500#int(p_test*len(dataset))
        j = 500#int(p_val*len(dataset))

        train_dataset = dataset[:n]
        val_dataset   = dataset[n:n+j]
        test_dataset  = dataset[n+j:n+j+i]

        del dataset

    else :
        print("reading datasets from file {:s}".format(savefile))
        train_dataset = torch.load(savefile+".train.torch")
        val_dataset   = torch.load(savefile+".val.torch")
        test_dataset  = torch.load(savefile+".test.torch")
        SAVE = False
            
    if SAVE :
        print("saving dataset to file {:s}".format(savefile))
        torch.save(train_dataset,savefile+".train.torch")
        torch.save(val_dataset,  savefile+".val.torch")
        torch.save(test_dataset, savefile+".test.torch")

    print(" test:",len(test_dataset))
    print("  val:",len(val_dataset))
    print("train:",len(train_dataset))

    ##########################################

    for folder in ["results/","results/networks/","results/dataframes","results/images"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    ##########################################

    OUTPUT = "E"

    irreps_in = "{:d}x0e".format(len(data.all_types()))
    if OUTPUT == "E":
        irreps_out = "1x0e"
    elif OUTPUT in ["ED","EDF"]:
        irreps_out = "1x0e + 1x1o"
    elif OUTPUT == "D":
        irreps_out = "1x1o"

    print("irreps_out:",irreps_out)

    radial_cutoff = 6.0
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":irreps_out,    # vector (L=1 and odd parity) to output the dipole
        "max_radius":radial_cutoff, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":10,
        "layers":2,
        "lmax":1,
        "default_dtype" : default_dtype,
    }
    net = SabiaNetworkManager(output=OUTPUT,radial_cutoff=radial_cutoff,**model_kwargs)
    print(net)
    print("net.irreps_in:",net.irreps_in)   #>> net.irreps_in: 2x0e
    print("net.irreps_out:",net.irreps_out) #>> net.irreps_out: 1x0e+1x1o

    # this line does not affect the results (at least the code does not complain)
    # net.eval()

    # it works for any batch_size
    train_dataloader = _make_dataloader(train_dataset,batch_size=50)
    for X in train_dataloader:

        # the water molecule has 3 atoms
        # this means that all the coordinates have lenght 9
        # if we predict energy and dipole the output will have lenght 4

        y = net(X)
        print("y.shape:",y.shape)           #>> y.shape: torch.Size([50, 4])

        E = net.energy(X)
        print("E.shape:",E.shape)           #>> E.shape: torch.Size([50])

        F = net.forces(X)
        print("F.shape:",F.shape)           #>> F.shape: torch.Size([50, 9])
        
        if OUTPUT in ["ED","D"] :
            P = net.dipole(X)
            print("P.shape:",P.shape)           #>> P.shape: torch.Size([50, 3])
            
            BEC = net.BEC(X)
            print("BEC.shape:",BEC.shape)       #>> BEC.shape: torch.Size([50, 3, 9])

        diff = net.check_equivariance(X)
        print("diff:",diff.mean())          #>> diff: 6.481285499819072e-15

        break

    print("\n Job done :)")

if __name__ == "__main__":
    main()