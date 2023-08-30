# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

# this fiel contains some useful functions

import argparse
import os
import itertools
import numpy as np
import re
#from ipi.engine.properties import Properties
from ipi.utils.units import unit_to_internal, unit_to_user

__all__ = ['flatten_list', 'get_all_system_permutations', 'get_all_permutations',
            'str2bool','get_one_file_in_folder','get_property_header','getproperty',
            'vector_type', 'output_folder', 'save2xyz', 'print_cell', 'convert',
            'Dict2Obj', 'get_attributes', 'merge_attributes', 'read_comments_xyz', 'segment',
            'recursive_copy', 'add_default']

def recursive_copy(source_dict:dict, target_dict:dict)->dict:
    """
    Recursively copy keys and values from a source dictionary to a target dictionary, if they are not present in the target.

    This function takes two dictionaries, 'source_dict' and 'target_dict', and copies keys and values from 'source_dict' to 'target_dict'. If a key exists in both dictionaries and both values are dictionaries, the function recursively calls itself to copy nested keys and values. If a key does not exist in 'target_dict', it is added along with its corresponding value from 'source_dict'.

    Args:
        source_dict (dict): The source dictionary containing keys and values to be copied.
        target_dict (dict): The target dictionary to which keys and values are copied if missing.

    Returns:
        dict: The modified 'target_dict' with keys and values copied from 'source_dict'.

    Example:
        >>> dict_A = {"a": 1, "b": {"b1": 2, "b2": {"b2_1": 3}}, "c": 4}
        >>> dict_B = {"a": 10, "b": {"b1": 20, "b2": {"b2_2": 30}}, "d": 40}
        >>> result = recursive_copy(dict_A, dict_B)
        >>> print(result)
        {'a': 10, 'b': {'b1': 20, 'b2': {'b2_1': 3, 'b2_2': 30}}, 'd': 40}
    """
    for key, value in source_dict.items():
        if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
            recursive_copy(value, target_dict[key])
        else:
            if key not in target_dict:
                target_dict[key] = value
    return target_dict


def add_default(dictionary: dict = None, default: dict = None) -> dict:
    """
    Add default key-value pairs to a dictionary if they are not present.

    This function takes two dictionaries: 'dictionary' and 'default'. It checks each key in the 'default' dictionary, and if the key is not already present in the 'dictionary', it is added along with its corresponding value from the 'default' dictionary. If 'dictionary' is not provided, an empty dictionary is used as the base.

    Args:
        dictionary (dict, optional): The input dictionary to which default values are added. If None, an empty dictionary is used. Default is None.
        default (dict): A dictionary containing the default key-value pairs to be added to 'dictionary'.

    Returns:
        dict: The modified 'dictionary' with default values added.

    Raises:
        ValueError: If 'dictionary' is not of type 'dict'.

    Example:
        >>> existing_dict = {'a': 1, 'b': 2}
        >>> default_values = {'b': 0, 'c': 3}
        >>> result = add_default(existing_dict, default_values)
        >>> print(result)
        {'a': 1, 'b': 2, 'c': 3}
    """
    if dictionary is None:
        dictionary = {}

    if not isinstance(dictionary, dict):
        raise ValueError("'dictionary' has to be of 'dict' type")

    return recursive_copy(source_dict=default, target_dict=dictionary)



# https://stackabuse.com/python-how-to-flatten-list-of-lists/
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def get_all_system_permutations(atoms):
    species   = np.unique(atoms)
    index     = {key: list(np.where(atoms == key)[0]) for key in species}
    # permutations = {key: get_all_permutations(i) for i,key in zip(index.values(),species)}
    permutations = [get_all_permutations(i) for i in index.values()]
    return list(itertools.product(*permutations))

def get_all_permutations(v):
    tmp = itertools.permutations(list(v))
    return [ list(i) for i in tmp ]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_one_file_in_folder(folder,ext):
    files = list()
    for file in os.listdir(folder):
        if file.endswith(ext):
            files.append(os.path.join(folder, file))
    if len(files) == 0 :
        raise ValueError("no '*{:s}' files found".format(ext))
    elif len(files) > 1 :
        raise ValueError("more than one '*{:s}' file found".format(ext))
    return files[0]

def get_property_header(inputfile,N=1000,search=True):

    names = [None]*N
    restart = False

    with open(inputfile, "r") as ifile:
        icol = 0        
        while True:
            line = ifile.readline()
            nline = line
            if not line:
                break
            elif "#" in line:
                line = line.split("-->")[1]
                line = line.split(":")[0]
                line = line.split(" ")[1]

                nline = nline.split("-->")[0]
                if "column" in nline:
                    lenght = 1
                else :
                    nline = nline.split("cols.")[1]
                    nline = nline.split("-")
                    a,b = int(nline[0]),int(nline[1])
                    lenght = b - a  + 1 

                if icol < N :
                    if not search:
                        if lenght == 1 :
                            names[icol] = line
                            icol += 1
                        else :
                            for i in range(lenght):
                                names[icol] = line + "-" + str(i)
                                icol += 1
                    else :
                        names[icol] = line
                        icol += 1
                else :
                    restart = True
                    icol += 1
                
            
    if restart :
        return get_property_header(inputfile,N=icol)
    else :
        return names[:icol]

def getproperty(inputfile, propertyname,data=None,skip="0",show=False):

    def check(p,l):
        if not l.find(p) :
            return False # not found
        elif l[l.find(p)-1] != " ":
            return False # composite word
        elif l[l.find(p)+len(p)] == "{":
            return True
        elif l[l.find(p)+len(p)] != " " :
            return False # composite word
        else :
            return True

    if type(propertyname) in [list,np.ndarray]: 
        out   = dict()
        units = dict()
        data = np.loadtxt(inputfile)
        for p in propertyname:
            out[p],units[p] = getproperty(inputfile,p,data,skip=skip)
        return out,units
    
    if show : print("\tsearching for '{:s}'".format(propertyname))

    skip = int(skip)

    # propertyname = " " + propertyname + " "

    # opens & parses the input file
    with open(inputfile, "r") as ifile:
        # ifile = open(inputfile, "r")

        # now reads the file one frame at a time, and outputs only the required column(s)
        icol = 0
        while True:
            try:
                line = ifile.readline()
                if len(line) == 0:
                    raise EOFError
                while "#" in line :  # fast forward if line is a comment
                    line = line.split(":")[0]
                    if check(propertyname,line):
                        cols = [ int(i)-1 for i in re.findall(r"\d+", line) ]                    
                        if len(cols) == 1 :
                            icol += 1
                            output = data[:,cols[0]]
                        elif len(cols) == 2 :
                            icol += 1
                            output = data[:,cols[0]:cols[1]+1]
                        elif len(cols) != 0 :
                            raise ValueError("wrong string")
                        if icol > 1 :
                            raise ValueError("Multiple instances for '{:s}' have been found".format(propertyname))

                        l = line
                        p = propertyname
                        if l[l.find(p)+len(p)] == "{":
                            unit = l.split("{")[1].split("}")[0]
                        else :
                            unit = "atomic_unit"

                    # get new line
                    line = ifile.readline()
                    if len(line) == 0:
                        raise EOFError
                if icol <= 0:
                    print("Could not find " + propertyname + " in file " + inputfile)
                    raise EOFError
                else :
                    if show : print("\tfound '{:s}'".format(propertyname))
                    return np.asarray(output),unit

            except EOFError:
                break

def vector_type(arg_value):
    try:
        # Split the input string by commas and convert each element to an integer
        values = [int(x) for x in arg_value.split(',')]
        return values
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid vector: {arg_value}") from e
    
def output_folder(folder):
    if folder in ["",".","./"] :
        folder = "."
    elif not os.path.exists(folder) :
        print("\n\tCreating directory '{:s}'".format(folder))
        os.mkdir(folder)
    return folder

def output_file(folder,what):
    folder = output_folder(folder)
    return "{:s}/{:s}".format(folder,what)

def save2xyz(what,file,atoms,comment=""):

    if len(what.shape) == 1 : # just one configuration, NOT correctly formatted

        what = what.reshape((-1,3))
        return save2xyz(what,file,atoms)
    
    elif len(what.shape) == 2 : 

        if what.shape[1] != 3 : # many configurations
            what = what.reshape((len(what),-1,3))
            return save2xyz(what,file,atoms)
        
        else : # just one configurations, correctly formatted
            return save2xyz(np.asarray([what]),file,atoms)

    elif len(what.shape) == 3 :

        Na = what.shape[1]
        if what.shape[2] != 3 :
            raise ValueError("wrong shape")
        
        with open(file,"w") as f :
            
            for i in range(what.shape[0]):
                pos = what[i,:,:]
                f.write(str(Na)+"\n")
                f.write("# {:s}\n".format(comment))
                for ii in range(Na):
                    f.write("{:>2s} {:>20.12e} {:>20.12e} {:>20.12e}\n".format(atoms[ii],*pos[ii,:]))
        return
    
def print_cell(cell,tab="\t\t"):
    cell = cell.T
    string = tab+"{:14s} {:1s} {:^10s} {:^10s} {:^10s}".format('','','x','y','z')
    for i in range(3):
        string += "\n"+tab+"{:14s} {:1d} : {:>10.6f} {:>10.6f} {:>10.6f}".format('lattice vector',i+1,cell[i,0],cell[i,1],cell[i,2])
    return string

def convert(what,family,_from,_to):
    factor  = unit_to_internal(family,_from,1)
    factor *= unit_to_user(family,_to,1)
    return what * factor

# def get_family(name):
#     return Properties.property_dict[name]["dimension"]

# https://www.blog.pythonlibrary.org/2014/02/14/python-101-how-to-change-a-dict-into-a-class/
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

def get_attributes(obj):
    return [i for i in obj.__dict__.keys() if i[:1] != '_']

def merge_attributes(A,B):
    attribs = get_attributes(B)
    for a in attribs:
        setattr(A, a, getattr(B, a))
    return A

def read_comments_xyz(file,Nmax=1000000):

    from ase import io
    first = io.read(file)
    Natoms = len(first)

    okay = 1
    result = [None]*Nmax
    restart = False
    i = 0
    k = 0 

    with open(file, "r+") as fp:
        # access each line
        line = fp.readline()

        # # skip lines
        # for n in range(skip):
        #     line = fp.readline()
        #     i += 1
        #     if i == okay:
        #         okay += Natoms+2

        while line:
            if i == okay:
                result[k] = line
                okay += Natoms+2
                k += 1
            
            if k >= Nmax:
                restart = True
                break
                
            line = fp.readline()
            i += 1
    
    if restart :
        return read_comments_xyz(file,Nmax*2)

    return result[:k]

def segment(A,B,N,start=0,end=1):
    """This function generates a segment
       given the initial (A) and final (B) points
       and put N points in the middle.
       
       A and B can be any kind of np.ndarray
    """
    assert A.shape == B.shape
    
    sequence = np.zeros((N+2,*A.shape))
    T = np.linspace(start,end,N+2)
    # N = 0 -> t=0,1
    # N = 1 -> t=0,0.5,1
    for n,t in enumerate(T):
        #t = float(n)/(N+1)
        sequence[n] = A*(1-t) + t*B
    return sequence

import os

def remove_empty_folder(folder_path):
    if is_folder_empty(folder_path):
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' has been removed.")
    else:
        print(f"Folder '{folder_path}' is not empty and cannot be removed.")

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

