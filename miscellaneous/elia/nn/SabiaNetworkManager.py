from torch_geometric.data import Data
import torch
from copy import copy
import numpy as np
import jax
import jax.numpy as jnp
from .SabiaNetwork import SabiaNetwork
from miscellaneous.elia.nn.water.make_dataset import make_datapoint
from typing import TypeVar
T = TypeVar('T', bound='SabiaNetworkManager')
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.


class SabiaNetworkManager(SabiaNetwork):

    lattice: torch.Tensor
    radial_cutoff: float
    _radial_cutoff: float
    symbols: list
    x: Data

    def __init__(self: T, radial_cutoff: float = 0.0, output: str = "EP", **kwargs) -> None:
        super(SabiaNetworkManager, self).__init__(**kwargs)

        self.output = output
        if self.output not in ["E", "EP", "EPF"]:
            raise ValueError("'output' must be 'E', 'EP' or 'EPF'")

        self.grad = None
        self.lattice = None
        self.radial_cutoff = None
        self._radial_cutoff = radial_cutoff
        self.symbols = None
        self.x = None
        pass

    def train(self: T, mode: bool) -> T:
        if self.grad is not None:
            del self.grad     # delete the gradient
            self.grad = None  # build an empty gradient
        return super(SabiaNetworkManager, self).train(mode)

    # def evaluate(self:T,positions,lattice,radial_cutoff,symbols)->torch.tensor:
    #     x = make_datapoint( lattice=lattice,\
    #                         radial_cutoff=radial_cutoff,\
    #                         symbols=symbols,\
    #                         positions=positions)
    #     y = self(x)
    #     return y if self.output == "E" else y[:,0]

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
        self.x.pos = R
        y = self(self.x)
        y = y if self.output == "E" else y[:,0]
        return y.reshape(())

    def forces(self: T, X: Data) -> torch.tensor:

        batch_size = len(np.unique(X.batch))
        y = torch.zeros(X.pos.shape, requires_grad=True).reshape(
            (batch_size, -1))

        for n in range(batch_size):

            index = X.batch == n
            self.lattice = X.lattice[n]
            self.symbols = X.symbols[n]
            self.radial_cutoff = X.radial_cutoff if hasattr(
                X, "radial_cutoff") else self._radial_cutoff
            R = X.pos[index]#.flatten()
            self.x = make_datapoint(lattice=self.lattice,
                                    radial_cutoff=self.radial_cutoff,
                                    symbols=self.symbols,
                                    positions=R,
                                    default_dtype=self.default_dtype)

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
            if False:
                R = X.pos[index].requires_grad_(True)#.flatten()
                y.data[n, :] = torch.autograd.functional.jacobian(self.PES, R, create_graph=self.training).flatten()

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
            if True:
                R = X.pos[index]#.requires_grad_(True)#.flatten()
                if self.grad is None:
                    self.grad = torch.func.grad(self.PES)
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

    @staticmethod
    def get_pred(model: T, X: Data) -> torch.Tensor:
        """return Energy, Polarization and Forces"""

        N = {"E": 1, "EP": 4, "EPF": 1+3+3*X.Natoms[0]}
        N = N[model.output]
        batch_size = len(np.unique(X.batch))

        if model.output in ["E", "EP"]:
            y = torch.zeros((batch_size, N))
            y = model(X)

        elif model.output == "EPF":
            y = torch.zeros((batch_size, N))
            EP = model(X)
            y[:, 0] = EP[:, 0]         # 1st column  for the energy
            y[:, 1:4] = EP[:, 1:4]       # 3rd columns for the polarization
            y[:, 4:] = model.forces(X)  # 3rd columns for the forces

        return y

    @staticmethod
    def get_real(X: Data, output: str = "E") -> torch.Tensor:
        """return Energy, Polarization and Forces"""

        # 'EPF' has to be modified in case we have different molecules in the dataset
        N = {"E": 1, "EP": 4, "EPF": 1+3+3*X.Natoms[0]}
        N = N[output]
        batch_size = len(np.unique(X.batch))

        # if batch_size > 1 :

        y = torch.zeros((batch_size, N))

        y[:, 0] = X.energy

        if output in ["EP", "EPF"]:
            y[:, 1:4] = X.polarization.reshape((batch_size, -1))

        elif output == "EPF":
            y[:, 4:] = X.forces.reshape((batch_size, -1))

        # else:
        #     y = torch.zeros(N)
        #     y[0]   = X.energy

        #     if output in ["EP","EPF"]:
        #         y[1:4] = X.polarization.flatten()

        #     elif output == "EPF":
        #         y[4:]  = X.forces.flatten()

        return y
