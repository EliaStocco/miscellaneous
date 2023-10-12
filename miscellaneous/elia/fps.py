#!/usr/bin/env python
# coding: utf-8

# Loading the datset # need to be parsed by ase
# 

# In[1]:


from ase.io import read, write
import chemiscope
import numpy as np
import os

from rascal.representations import SphericalInvariants as SOAP
from skmatter.preprocessing import StandardFlexibleScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


# In[3]:


structure_file = '../dataset/full/full_dataset.extxyz'
frames = read(structure_file, index=':', format='extxyz')  #eV

available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

print('Number of frames: ', len(frames))
print('Number of atoms/frame: ', len(frames[0]))
print('Available structure properties: ', available_structure_properties)
print('Available atom-level properties: ', available_atom_level_properties)


# Creating the atomic enviroments (SOAP descriptors)

# In[7]:


SOAP_HYPERS = {
    "interaction_cutoff": 3.5,
    "max_radial": 6,
    "max_angular": 6,
    "gaussian_sigma_constant": 0.4,
    "cutoff_smooth_width": 0.5,
    "gaussian_sigma_type": "Constant",
}


# In[8]:


numbers = list(sorted(set([int(n) for frame in frames for n in frame.numbers])))

# initialize SOAP
soap = SOAP(
      global_species=numbers, 
      expansion_by_species_method='user defined',
      **SOAP_HYPERS            
  )

X = None 
print("computing SOAP features...")
for i, frame in enumerate(tqdm(frames)):
    # normalize cell for librascal input
    if np.linalg.norm(frame.cell) < 1e-16:
        extend = 1.5 * (np.max(frame.positions.flatten()) - np.min(frame.positions.flatten()))
        frame.cell = [extend, extend, extend]
        frame.pbc = True
    frame.wrap(eps=1e-16)

    x = soap.transform(frame).get_features(soap).mean(axis=0) # here it takes mean over atoms in the frame
    if X is None:
        X = np.zeros((len(frames), x.shape[-1]))
    X[i] = x

print(f"SOAP features shape: {X.shape}")
np.save('full-featurization.npy', X)


# Farthest Point Sampling for the selection of diverse structure

# In[215]:


n_FPS = 100 # number of structures to select 
struct_idx = FPS(n_to_select=n_FPS, progress_bar = True, initialize = 'random').fit(X.T).selected_idx_
X_fps = X[struct_idx]

print(f"FPS selected indices: {struct_idx.shape}")
print(f"Original: {X.shape} ---> FPS: {X_fps.shape}")


# Saving the fps selected structure

# In[216]:


frames_fps = [frames[i] for i in struct_idx]
write('fps_100_selected_900K.xyz', frames_fps, format='extxyz')


# Visualizing using the principal componenets of the selected dataset and the original dataset using the chemiscope

# In[9]:


X = StandardFlexibleScaler(column_wise=False).fit_transform(X)
T = PCA(n_components=2).fit_transform(X)


# In[12]:


np.savetxt('PES_PCA.txt', np.concatenate([T, np.array([frame.info['energy'] for frame in frames]).reshape(-1, 1)], axis=1))


# In[10]:


available_structure_properties = list(set([k for frame in frames for k in frame.info.keys()]))
available_atom_level_properties = list(set([k for frame in frames for k in frame.arrays.keys()]))

print("Available structure-level properties", available_structure_properties)
print("Available atom-level properties", available_atom_level_properties)


# In[11]:




properties = {
    "PCA": {
        # change the following line if your map is per-atom
        "target": "structure",
        "values": T,

        # change the following line to describe your map
        "description": "PCA of structure-averaged representation",
    },

    # this is an example of how to add structure-level properties
    "energy": {
        "target": "structure",
        "values": [frame.info['energy'] for frame in frames],

        # change the following line to correspond to the units of your property
        "units": "eV",
    },

    # this is an example of how to add atom-level properties
    "numbers": {
        "target": "atom",
        "values": np.concatenate([frame.arrays['numbers'] for frame in frames]),
    },
}

chemiscope.write_input(
    path=f"PES_PCA-chemiscope.json.gz",
    frames=frames,
    properties=properties,

    # # This is required to display properties with `target: "atom"`
    # # Without this, the chemiscope will show only structure-level properties
    # environments=chemiscope.all_atomic_environments(frames),
)


# In[ ]:




