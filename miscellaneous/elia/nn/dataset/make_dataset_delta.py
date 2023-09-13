# Import necessary modules and classes
from miscellaneous.elia.nn.dataset import make_dataset, make_datapoint
import torch
from torch_geometric.data import Data

#----------------------------------------------------------------#
# Function to add a reference to a datapoint
def add_reference(datapoint: Data,
                  dipole: torch.tensor,
                  pos: torch.tensor):
    """
    Adds a reference to a datapoint by subtracting dipole and adding deltaR.

    Parameters:
    datapoint (Data): The original datapoint.
    dipole (torch.tensor): The dipole to be subtracted.
    pos (torch.tensor): The position tensor for computing deltaR.

    Returns:
    Data: The modified datapoint with the reference added.
    """
    # Subtract the dipole
    # Actually, I could avoid subtraxting the dipole ...
    # Let's remove it for semplicity
    # datapoint.dipole -= dipole

    # Compute distance to the reference configuration and add it to the input features
    deltaR = (datapoint.pos - pos).detach()
    datapoint.x = torch.cat((datapoint.x, deltaR), dim=1)
    
    return datapoint

#----------------------------------------------------------------#

# Function to create a dataset with a reference added
def make_dataset_delta(ref_index: int = 0, **argv):
    """
    Creates a dataset with references added to each datapoint.

    Parameters:
    ref_index (int): Index of the reference datapoint.
    **argv: Additional arguments for make_dataset function.

    Returns:
    list of Data: The dataset with references added.
    torch.tensor: The dipole of the reference datapoint.
    torch.tensor: The position tensor of the reference datapoint.
    """
    # Create the original dataset
    dataset = make_dataset(**argv)

    # Clone the dipole and position of the reference datapoint
    dipole = dataset[ref_index].dipole.clone()
    pos = dataset[ref_index].pos.clone()

    # Add references to each datapoint in the dataset
    for n in range(len(dataset)):
        dataset[n] = add_reference(dataset[n], dipole, pos)

    return dataset, dipole, pos

#----------------------------------------------------------------#

# Function to create a datapoint with a reference added
def make_datapoint_delta(dipole: torch.tensor, pos: torch.tensor, **argv):
    """
    Creates a datapoint with a reference added.

    Parameters:
    dipole (torch.tensor): The dipole to be subtracted.
    pos (torch.tensor): The position tensor for computing deltaR.
    **argv: Additional arguments for make_datapoint function.

    Returns:
    Data: The datapoint with the reference added.
    """
    # Create the original datapoint
    data = make_datapoint(**argv)
    
    # Add a reference to the datapoint
    data = add_reference(data, dipole, pos)    
    return data
