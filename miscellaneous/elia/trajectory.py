from ase.io import read
from ase import Atoms
from miscellaneous.elia.vectorize import easyvectorize
from miscellaneous.elia.functions import read_comments_xyz
import re
import ipi.utils.mathtools as mt
import numpy as np

deg2rad = np.pi / 180.0
abcABC = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
abcABCunits = r'\{([^}]+)\}'
dtype = type(easyvectorize(Atoms))

# class trajectory(dtype):

#     def __init__(self,*argc,**kwargs):
#         super().__init__(*argc,**kwargs)
#         pass

#     @classmethod
#     def load(cls,file):

#         atoms = read(file,index=":")
#         for n in range(len(atoms)):
#             atoms[n].info = None
#             atoms[n].set_calculator(None)

#         try : 
#             comments = read_comments_xyz(file)
#             cells = [ abcABC.search(comment) for comment in comments ]
#             cells = np.zeros((len(cells),3,3))
#             for n,cell in enumerate(cells):
#                 a, b, c = [float(x) for x in cell.group(1).split()[:3]]
#                 alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
#                 cells[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

#             # matches = re.findall(abcABCunits,comments[0])
#             # if len(matches) != 2 :
#             #     raise ValueError("wrong number of matches")
#             # else :
#             #     units = {
#             #         "positions" : matches[0],
#             #         "cell" : matches[1]
#             #     }

#             for n in range(len(atoms)):
#                 atoms[n].set_cell(cells[n].T)

#         except:
#             pass
        
#         return cls(atoms)
    
def trajectory(file):

    atoms = read(file,index=":")
    for n in range(len(atoms)):
        atoms[n].info = dict()
        atoms[n].set_calculator(None)

    try : 
        comments = read_comments_xyz(file)
        cells = [ abcABC.search(comment) for comment in comments ]
        cells = np.zeros((len(cells),3,3))
        for n,cell in enumerate(cells):
            a, b, c = [float(x) for x in cell.group(1).split()[:3]]
            alpha, beta, gamma = [float(x) * deg2rad for x in cell.group(1).split()[3:6]]
            cells[n] = mt.abc2h(a, b, c, alpha, beta, gamma)

        # matches = re.findall(abcABCunits,comments[0])
        # if len(matches) != 2 :
        #     raise ValueError("wrong number of matches")
        # else :
        #     units = {
        #         "positions" : matches[0],
        #         "cell" : matches[1]
        #     }

        for n in range(len(atoms)):
            atoms[n].set_cell(cells[n].T)

    except:
        pass
    
    return easyvectorize(Atoms)(atoms)





