import numpy as np
from ase import Atoms

#---------------------------------------#
def find_transformation(A: Atoms, B: Atoms):
    """
    Compute the transformation matrix between the cells/lattice vectors of two atomic structures.

    Parameters:
        A (ase.Atoms): The first atomic structure (primitive cell).
        B (ase.Atoms): The second atomic structure (supercell).

    Returns:
        numpy.ndarray: The transformation matrix from A to B.
    """
    # Compute the transformation matrix
    M = B.get_cell().T @ np.linalg.inv(A.get_cell().T)

    if not np.allclose(B.get_cell().T,M @ A.get_cell().T):
        raise ValueError("error in the code implementation")

    return M

#---------------------------------------#
def segment(A:np.ndarray, B:np.ndarray, N:int, start:int=0, end:int=1):
    """This function generates a segment
    given the initial (A) and final (B) points
    and put N points in the middle.

    A and B can be any kind of np.ndarray
    """
    assert A.shape == B.shape

    sequence = np.zeros((N + 2, *A.shape))
    T = np.linspace(start, end, N + 2)
    # N = 0 -> t=0,1
    # N = 1 -> t=0,0.5,1
    for n, t in enumerate(T):
        # t = float(n)/(N+1)
        sequence[n] = A * (1 - t) + t * B
    return sequence