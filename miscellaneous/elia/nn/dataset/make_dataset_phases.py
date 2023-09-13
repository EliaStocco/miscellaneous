from miscellaneous.elia.nn.dataset import make_dataset
from miscellaneous.elia.classes import MicroState

def make_dataset_phases(data:MicroState,**argv):

    # same_lattice should be false
    func = lambda : data.get_phases(array=None,unit="atomic_unit",same_lattice=True,inplace=False,fix=True)

    dataset = make_dataset(output_method=func,data=data,**argv)

    return dataset