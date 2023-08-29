import numpy as np
from copy import copy

__all__ = ["normalize","compute_normalization_factors"]

def normalize(dataset,mu,sigma,variable):
    """
    Normalize the specified variable in the dataset.

    Args:
        dataset (list): List of data points.
        mu (float): Mean value for normalization.
        sigma (float): Standard deviation value for normalization.
        variable (str): Name of the variable to be normalized.

    Returns:
        list: Normalized dataset.
    """
    new_dataset = copy(dataset)
    for n in range(len(dataset)):
        x = getattr(dataset[n],variable)
        x = (x-mu)/sigma
        setattr(new_dataset[n],variable,x)

    return new_dataset

def compute_normalization_factors(dataset,variable):
    """
    Compute mean and standard deviation normalization factors for the given variable.

    Args:
        dataset (list): List of data points.
        variable (str): Name of the variable for which to compute factors.

    Returns:
        tuple: Tuple containing mean and standard deviation factors.
    """
    # Extract data for the specified variable from the dataset
    v = getattr(dataset[0],variable)
    data = np.full((len(dataset),*v.shape),np.nan)
    for n,x in enumerate(dataset):
        data[n] = getattr(x,variable)

    # Calculate mean and standard deviation along axis 0
    mu    = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    
    return mu, sigma
