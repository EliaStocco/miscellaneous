# __all__ = [ "nn" ]
def import_submodules(package_name):
    import importlib
    import pkgutil

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__):
        if module_name != f"{package_name}.__init__":
            print(f"importing '{module_name}' ... ", end="")
            try:
                module = importlib.import_module(f"{package_name}.{module_name}")
                globals().update(module.__dict__)
                print("done")
            except ImportError:
                print(f"error while importing '{module_name}'")

# Call the function to import and print messages for all submodules
#import_submodules(__name__)



#print("\t 'miscellaneous.elia'")

#print("importing 'miscellaneous.elia.functions' ... ",end="")
from .functions import *
#print("done")

#print("importing 'miscellaneous.elia.classes' ... ",end="")
from .classes import *
#print("done")

#print("\t\timporting 'miscellaneous.elia.functions'")
from .nn import *
#print("\t\timported 'miscellaneous.elia.functions'")

#print("imported 'miscellaneous.elia'")

