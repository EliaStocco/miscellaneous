import numpy as np
import argparse
#---------------------------------------#
def size_type(s,dtype=int,N=None):
    s = s.split("[")[1].split("]")[0].split(",")
    if N is not None and len(s) != N :
        raise ValueError("You should provide {:d} values".format(N)) 
    else:
        return np.asarray([ dtype(k) for k in s ])
#---------------------------------------#
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")