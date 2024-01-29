import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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

#---------------------------------------#
def get_sorted_atoms_indices(reference:Atoms,structure:Atoms):
    """Calculate pairwise distances and obtain optimal sorting indices."""
    # Calculate the pairwise distances between atoms in the two structures
    distances = cdist(reference.get_positions(), structure.get_positions())
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distances)
    return col_ind

#---------------------------------------#
def sort_atoms(reference:Atoms,structure:Atoms):
    """Sort atoms in the second structure by minimizing the distances w.r.t. the atoms in the first structure."""
    indices = get_sorted_atoms_indices(reference, structure)
    sorted = structure[indices]
    return sorted, indices

#---------------------------------------#
def find_transformation(A:Atoms,B:Atoms):
    """Compute the transformation matrix between the lattice vectors of two atomic structures."""
    M = np.asarray(B.cell).T @ np.linalg.inv(np.asarray(A.cell).T)
    size = M.round(0).diagonal().astype(int)
    return size, M