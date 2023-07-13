import os
from miscellaneous.elia.classes import MicroState
from miscellaneous.elia.nn import SabiaNetwork
from miscellaneous.elia.nn import train
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import copy

import numpy as np
import torch_geometric
from tqdm import tqdm
import random
import ase
import torch
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def make_dataset(systems,\
                 type_onehot,\
                 type_encoding,\
                 output,\
                 radial_cutoff):
    
    # better to do this
    output = torch.tensor(output)

    #print("output:",output)
    #print("output value shape:",output[0].shape)

    dataset = [None] * len(systems)
    n = 0 
    bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
    for crystal, out in tqdm(zip(systems, output),total=len(systems), bar_format=bar_format):
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        edge_src, edge_dst, edge_shift = \
            ase.neighborlist.neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)

        data = torch_geometric.data.Data(
            pos=torch.tensor(crystal.get_positions()),
            lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
            x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            # edge_src=torch.tensor(edge_src,dtype=int),
            # edge_dst=torch.tensor(edge_dst,dtype=int),
            edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
            yreal=out  # polarization (??assumed to be normalized "per atom" ??)
        )

        dataset[n] = data
        n += 1
    return dataset#,dtype=torch_geometric.data.Data)

def main():

    radial_cutoff = 6.0

    ##########################################

    READ = True
    SAVE = False
    savefile = "microstate.pickle"

    if not READ or not os.path.exists(savefile) :
        #print("reading properties from raw files")
        infile = "i-pi.positions_0.xyz"
        options = {"properties" : "i-pi.properties.out",\
                "positions":infile,\
                "cells":infile,\
                "types":infile,\
                "velocities":"i-pi.velocities_0.xyz"}
        data = MicroState(options)
        _ = data.to_ase(inplace=True)
        # data.convert_property(what="time",unit="picosecond",family="time",inplace=True)
    else :
        #print("reading data from pickle file")
        data = MicroState.load(savefile)

    if "potential" not in data.properties:
        temp = ase.io.read("H2O.scf.in")
        masses = temp.get_masses() * 1822.8885
        data.masses = np.zeros((len(masses),3))
        data.masses[:,0] = masses
        data.masses[:,1] = masses
        data.masses[:,2] = masses
        data.masses = data.masses.flatten()
        kinetic = 0.5 * np.sum(data.masses * np.square(data.velocities),axis=1)
        potential = data.properties["conserved"] - kinetic
        data.add_property(array=kinetic,name="kinetic")
        data.add_property(array=potential,name="potential")

    if SAVE :
        MicroState.save(data,savefile)

    ##########################################

    # species
    types = data.types
    species = np.unique(types) 
    type_encoding = {}
    for n,s in enumerate(species):
        type_encoding[s] = n
    type_onehot = torch.eye(len(type_encoding))
    # for n in range(len(species)):
    #     print(species[n]," : ",type_onehot[n])
    #type_onehot

    ########################################## 

    READ = True
    SAVE = False
    savefile = "dataset.torch"

    if not READ or not os.path.exists(savefile) :
        print("building dataset")
        dataset = make_dataset( systems=data.ase, # already computes 
                                type_onehot=type_onehot,
                                type_encoding=type_encoding,
                                output=data.properties["potential"],
                                radial_cutoff=radial_cutoff)
    else :
        print("reading dataset from file {:s}".format(savefile))
        dataset = torch.load(savefile)
            
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

    irreps_in = "{:d}x0e".format(type_onehot.shape[1])
    radial_cutoff = 6.0
    model_kwargs = {
        "irreps_in":irreps_in,      # One hot scalars (L=0 and even parity) on each atom to represent atom type
        "irreps_out":"1x0e",        # vector (L=1 and odd parity) to output the polarization
        #"irreps_node_attr": None,#"1x0e + 1x1o",
        "max_radius":radial_cutoff, # Cutoff radius for convolution
        "num_neighbors":2,          # scaling factor based on the typical number of neighbors
        "pool_nodes":True,          # We pool nodes to predict total energy
        "num_nodes":2,
        "mul":4,
        "layers":3,
        "lmax":1,
        "p":[1]
    }
    net = SabiaNetwork(**model_kwargs)

    ##########################################

    n = 1 
    all_bs = np.arange(30,101,10)
    all_lr = np.logspace(-1, -4.0, num=8)
    Ntot = len(all_bs)*len(all_lr)
    print("\n")
    print("all batch_size:",all_bs)
    print("all lr:",all_lr)
    print("\n")

    for batch_size in all_bs :

        for lr in all_lr:
            
            print("\n#########################\n")
            print("\tbatch_size={:d}\t|\tlr={:.1e}\t|\tn={:d}/{:d}".format(batch_size,lr,n,Ntot))

            print("\n\trebuilding network...\n")
            net = SabiaNetwork(**model_kwargs)

            hyperparameters = {
                'batch_size': batch_size,
                'n_epochs'  : 20,
                'optimizer' : "Adam",
                'lr'        : lr,
                'loss'      : "MSE"
            }

            out_model, arrays = train(  model=copy(net),\
                                        train_dataset=train_dataset,\
                                        val_dataset=val_dataset,\
                                        hyperparameters=hyperparameters,\
                                        pp_pred=None,\
                                        get_real=None,\
                                        make_dataloader=None)
            
            savefile = "./results/networks/bs={:d}.{:.1e}.torch".format(batch_size,lr)                
            print("saving network to file {:s}".format(savefile))
            torch.save(out_model, savefile)

            savefile = "results/dataframes/bs={:d}.{:.1e}.csv".format(batch_size,lr)  
            print("saving arrays to file {:s}".format(savefile))
            arrays.to_csv(savefile,index=False)

            try :
                train_loss = arrays["train_loss"]
                val_loss = arrays["val_loss"]

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
                savefile = "results/images/bs={:d}.{:.1e}.pdf".format(batch_size,lr)
                plt.savefig(savefile)

                plt.figure().clear()
                plt.close()
                plt.cla()
                plt.clf()


            except:
                print("Some error during plotting")

            n += 1

    print("\nJob done :)")

if __name__ == "__main__":
    main()
