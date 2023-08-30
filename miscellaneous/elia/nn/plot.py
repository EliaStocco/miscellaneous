import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_learning_curves(train_loss,val_loss,file,title=None,opts=None):
    if opts is None:
        opts["N"] = 1
    N = len(train_loss)
    if N % opts["N"] != 0 :
        return
    else :
        try :

            matplotlib.use('Agg')
            fig,ax = plt.subplots(figsize=(10,4))
            x = np.arange(len(train_loss))+1

            ax.plot(x,val_loss,  color="red" ,label="val",  marker="x",linewidth=0.7,markersize=2)
            ax.plot(x,train_loss,color="navy",label="train",marker=".",linewidth=0.7,markersize=2)

            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.grid(True, which="both",ls="-")
            plt.xlim(1,x.max())
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if title is not None :
                plt.title(title)

            plt.tight_layout()
            plt.savefig(file)
            plt.close()

            # plt.figure().clear()
            # plt.cla()
            # plt.clf()

        except:
            print("Some error during plotting")
        return
