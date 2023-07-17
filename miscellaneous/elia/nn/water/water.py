import torch
from torch.autograd.functional import jacobian
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.nn import SabiaNetwork
from miscellaneous.elia.nn import train
from miscellaneous.elia.nn import get_type_onehot_encoding
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import copy
import pandas as pd
import numpy as np
import random
from ase.io import read
from ase import Atoms
from torch_geometric.data import Data
from tqdm import tqdm
from ase.neighborlist import neighbor_list
import jax

# Documentation
# - https://pytorch.org/docs/stable/autograd.html
# - https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e

#----------------------------------------------------------------#

def make_dataset(data:MicroState,\
                 radial_cutoff:float):
    
    species = data.all_types()
    type_onehot, type_encoding = get_type_onehot_encoding(species)    

    systems = data.to_ase()

    energy       = torch.tensor(data.properties["potential"])
    polarization = torch.tensor(data.properties["totalpol"])
    forces       = torch.tensor(data.forces)

    dataset = [None] * len(systems)
    n = 0 
    for crystal, e, p, f in tqdm(zip(systems,energy,polarization,forces),
                                 total=len(systems), 
                                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        edge_src, edge_dst, edge_shift = \
            neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
        
        pos     = torch.tensor(crystal.get_positions())
        lattice = torch.tensor(crystal.cell.array).unsqueeze(0) # We add a dimension for batching
        x       = type_onehot[[type_encoding[atom] for atom in crystal.get_chemical_symbols()]]

        edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)

        data = Data(
            pos=pos,
            lattice=lattice,  
            x=x,
            symbols = crystal.get_chemical_symbols(),
            edge_index=edge_index,
            edge_shift=torch.tensor(edge_shift),
            energy=e, # energy
            polarization=p, # polarization
            forces=f, # forces
            Natoms=crystal.get_global_number_of_atoms(), # valid only if all the structures have the same number of atoms
        )

        dataset[n] = data
        n += 1
    return dataset

#----------------------------------------------------------------#

def main():

    radial_cutoff = 6.0

    ##########################################

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/microstate.pickle"

    if not READ or not os.path.exists(savefile) or RESTART :
        infile = "data/i-pi.positions_0.xyz"
        instructions = {"properties" : "data/i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "forces":"i-pi.forces_0.xyz"}
        data = MicroState(instructions)
    else :
        data = MicroState.load(savefile)
        SAVE = False

    if SAVE :
        MicroState.save(data,savefile)

    ########################################## 

    RESTART = False 
    READ = True
    SAVE = True
    savefile = "data/dataset.torch"

    if not READ or not os.path.exists(savefile) or RESTART :
        print("building dataset")
        dataset = make_dataset( data=data,radial_cutoff=radial_cutoff)
    else :
        print("reading dataset from file {:s}".format(savefile))
        dataset = torch.load(savefile)
        SAVE = False
            
    if SAVE :
        print("saving dataset to file {:s}".format(savefile))
        torch.save(dataset,savefile)

    # shuffle
    random.shuffle(dataset)

    ##########################################

    # train, test, validation
    #p_test = 20/100 # percentage of data in test dataset
    #p_val  = 20/100 # percentage of data in validation dataset
    n = 2000
    i = 500#int(p_test*len(dataset))
    j = 500#int(p_val*len(dataset))

    train_dataset = dataset[:n]
    val_dataset   = dataset[n:n+j]
    test_dataset  = dataset[n+j:n+j+i]

    print(" test:",len(test_dataset))
    print("  val:",len(val_dataset))
    print("train:",len(train_dataset))

    ##########################################

    for folder in ["results/","results/networks/","results/dataframes","results/images"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    ##########################################

    irreps_in = "{:d}x0e".format(len(data.all_types()))
    radial_cutoff = 6.0
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":"1x0e + 1x1o", # vector (L=1 and odd parity) to output the polarization
        "max_radius":radial_cutoff, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":4,
        "layers":3,
        "lmax":1,
        "p":[1],
        "default_dtype" : default_dtype,
    }
    net = SabiaNetwork(**model_kwargs)
    print(net)
    del net

    #----------------------------------------------------------------#

    def make_datapoint(lattice, positions, radial_cutoff, symbols):

        with torch.no_grad():
            lattice = lattice#.unsqueeze(0) # We add a dimension for batching
            crystal = Atoms(cell=lattice,positions=positions,symbols=symbols)

        species = np.unique(symbols)
        type_onehot, type_encoding = get_type_onehot_encoding(species)

        edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
        
        return Data(
                pos=positions.reshape((-1,3)),
                lattice=lattice.unsqueeze(0),  # We add a dimension for batching
                x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
                edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
                edge_shift=torch.tensor(edge_shift),
            )

    ##########################################
    m  = None # model 
    l  = None # lattice
    rc = None # radial_cutoff
    s  = None # symbols

    def pes(R):
        # this ambda should capute the above variable by reference
        global m  # model 
        global l  # lattice
        global rc # radial_cutoff
        global s  # symbols

        print(l)
        print(rc)
        print(s)
        return m(make_datapoint(lattice=l,\
                                    radial_cutoff=rc,\
                                    symbols=s,\
                                    positions=R))#[:,0].reshape(())
    
    jac = torch.func.vjp(pes)
    forces_grad = torch.func.grad(pes)
    der2 = torch.func.grad(forces_grad)
    
    def forces(model,X):
        """This function compute the forces as the gradient of the model w.r.t. the atomic position.
        The function takes as input a Data object"""

        global m  # model 
        global l  # lattice
        global rc # radial_cutoff
        global s  # symbols

        batch_size = len(np.unique(X.batch))

        out = torch.zeros(X.pos.shape,requires_grad=True).reshape((batch_size,-1))

        for n in range(batch_size):

            index = X.batch == n
            l = X.lattice[n]
            s = X.symbols[n]
            rc = radial_cutoff
            m = model            
            R = X.pos[index].requires_grad_(True)

            # return only the energy
            # pes = lambda R: model(make_datapoint(lattice=lattice,\
            #                                      radial_cutoff=radial_cutoff,\
            #                                      symbols=symbols,\
            #                                      positions=R))[:,0].reshape(())
            
            #temp = jacobian(func=pes,inputs=R)
            temp = forces_grad(R)
            out.data[n,:] = temp.flatten()

        return out
    
    ##########################################

    def EPFpred(model,X)->torch.Tensor:
        """return Energy, Polarization and Forces"""
        p = X.polarization.reshape(-1,3)
        lenX = len(p)
        a = 3
        if hasattr(X.Natoms,"__len__"):
            b = int(X.Natoms[0])*3
        else :
            b = int(X.Natoms)*3
        y = torch.zeros((lenX,1+a+b)) 
        EP = model(X)
        y[:,0]         = EP[:,0]         # 1st column  for the energy
        y[:,1:a+1]     = EP[:,1:4]       # 3rd columns for the polarization
        y[:,a+1:a+b+1] = forces(model,X) # 3rd columns for the forces
        return y
        
    ##########################################

    def EPFreal(X)->torch.Tensor:
        """return Energy, Polarization and Forces"""

        batch_size = len(np.unique(X.batch))

        if batch_size > 1 :
            y = torch.zeros((batch_size,1+3+3*X.Natoms[0]))
            y[:,0]   = X.energy#.reshape((batch_size,-1))
            y[:,1:4] = X.polarization.reshape((batch_size,-1))
            y[:,4:]  = X.forces.reshape((batch_size,-1))

        else:
            y = torch.zeros((1+3+X.Natoms*3))
            y[0]   = X.energy#.reshape((batch_size,-1))
            y[1:4] = X.polarization.reshape((batch_size,-1))
            y[4:]  = X.forces.reshape((batch_size,-1))

        return y

    ##########################################

    n = 0
    all_bs = np.arange(30,101,10)
    all_lr = np.logspace(-1, -4.0, num=8)
    Ntot = len(all_bs)*len(all_lr)
    print("\n")
    print("all batch_size:",all_bs)
    print("all lr:",all_lr)
    print("\n")
    df = pd.DataFrame(columns=["bs","lr","file"],index=np.arange(len(all_bs)*len(all_lr)))

    for batch_size in all_bs :

        for lr in all_lr:

            df.at[n,"bs"] = batch_size
            df.at[n,"lr"] = lr
            df.at[n,"file"] = "bs={:d}.lr={:.1e}.pdf".format(batch_size,lr)
            
            print("\n#########################\n")
            print("\tbatch_size={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(batch_size,lr,n+1,Ntot))

            print("\n\trebuilding network...\n")
            net = SabiaNetwork(**model_kwargs)#.to(device)

            hyperparameters = {
                'batch_size': batch_size,
                'n_epochs'  : 200,
                'optimizer' : "Adam",
                'lr'        : lr,
                'loss'      : "MSE"
            }

            print("\n\ttraining network...\n")
            out_model, arrays = train(  model=net,\
                                        train_dataset=train_dataset,\
                                        val_dataset=val_dataset,\
                                        hyperparameters=hyperparameters,\
                                        get_pred=EPFpred,\
                                        get_real=EPFreal,\
                                        make_dataloader=None)
            
            savefile = "./results/networks/{:s}.torch".format(df.at[n,"file"])                
            print("saving network to file {:s}".format(savefile))
            torch.save(out_model, savefile)

            savefile = "results/dataframes/{:s}.csv".format(df.at[n,"file"])  
            print("saving arrays to file {:s}".format(savefile))
            arrays.to_csv(savefile,index=False)

            try :
                train_loss = arrays["train_loss"]
                val_loss = arrays["val_loss"]

                print("\n\tplotting losses...\n")
                fig,ax = plt.subplots(figsize=(10,4))
                x = np.arange(len(train_loss))

                ax.plot(x,train_loss,color="blue",label="train",marker=".",linewidth=0.7,markersize=2)
                ax.plot(val_loss,color="red",label="val",marker="x",linewidth=0.7,markersize=2)

                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both",ls="-")
                plt.xlim(0,hyperparameters["n_epochs"])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.title("batch_size={:d}, lr={:.1e}.pdf".format(batch_size,lr))

                plt.tight_layout()
                savefile = "results/images/{:s}.pdf".format(df.at[n,"file"])
                plt.savefig(savefile)
                plt.close(fig)

                plt.figure().clear()
                plt.cla()
                plt.clf()


            except:
                print("Some error during plotting")

            n += 1

    df.to_csv("info.csv",index=False)

    print("\nJob done :)")

if __name__ == "__main__":
    main()
