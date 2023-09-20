import torch
import json5 as json
import importlib
from ase.io import read 
import numpy as np

def get_data_from_dataset(dataset,variable):
    # Extract data for the specified variable from the dataset
    v = getattr(dataset[0],variable)
    data = torch.full((len(dataset),*v.shape),np.nan)
    for n,x in enumerate(dataset):
        data[n,:] = getattr(x,variable)
    return data

def get_class(module_name, class_name):
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Get the class from the module
        class_obj = getattr(module, class_name)
        
        # Create an instance of the class
        #instance = class_obj()
        
        return class_obj
    
    except ImportError:
        raise ValueError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
    
def get_model(instructions,parameters:str):

    if type(instructions) == str :

        with open(instructions, "r") as json_file:
            _instructions = json.load(json_file)
        instructions = _instructions

    # instructions['kwargs']["normalization"] = None

    # wxtract values for the instructions
    kwargs = instructions['kwargs']
    cls    = instructions['class']
    mod    = instructions['module']

    # get the class to be instantiated
    class_obj = get_class(mod,cls)
    
    # instantiate class
    #try :
    model = class_obj(**kwargs)

    # Load the parameters from the saved file
    checkpoint = torch.load(parameters)

    # Update the model's state dictionary with the loaded parameters
    model.load_state_dict(checkpoint)
    model.eval()

    # Store the chemical species that will be used during the simulation.
    model._symbols = instructions["chemical-symbols"]

    return model