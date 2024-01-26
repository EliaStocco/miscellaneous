import pickle

class pickleIO:
    def to_pickle(self,file):
        """Save the object to a *.pickle file."""
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file):
        """Load an object from a *.pickle file."""
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return cls(obj)