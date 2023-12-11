import numpy as np
import argparse
#---------------------------------------#
def size_type(s):
    s = s.split("[")[1].split("]")[0].split(",")
    match len(s):
        case 3:
            return np.asarray([ float(k) for k in s ])
        case _:
            raise ValueError("You should provide 3 integers") 
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