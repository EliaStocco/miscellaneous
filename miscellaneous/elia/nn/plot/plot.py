import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
# from reloading import reloading

# @reloading
def plot_learning_curves(arrays,file,title=None,opts=None):

    train_loss  = arrays["train"]   if "train"   in arrays else None
    val_loss    = arrays["val"]     if "val"     in arrays else None
    train_loss2 = arrays["train-2"] if "train-2" in arrays else None
    errors      = arrays["std"]     if "std"     in arrays else None
    ratio       = arrays["ratio"]   if "ratio"   in arrays else None
    ratio2      = arrays["ratio-2"] if "ratio-2" in arrays else None


    if opts is None:
        opts = {}
        opts["N"] = 1
    N = len(train_loss)
    if N % opts["N"] != 0 :
        return
    else :
        try :

            matplotlib.use('Agg')
            fig,ax = plt.subplots(figsize=(10,4))
            x = np.arange(len(train_loss))+1

            ax.plot(x,val_loss,  color="red" ,label="val",  marker=".",linewidth=0.7,markersize=2,linestyle="-")
            ax.plot(x,train_loss,color="navy",label="$\\mu$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
            if errors is not None :
                # ax.errorbar(x,train_loss,errors,color="navy",alpha=0.5,linewidth=0.5)
                ax.plot(x,errors,color="purple",label="$\\sigma$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
            if train_loss2 is not None :
                ax.plot(x,train_loss2,color="green",label="train$^*$",marker=".",linewidth=0.7,markersize=2,linestyle="-")

            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend(loc="upper left")
            ax.grid(True, which="both",ls="-")
            xlim = ax.get_xlim()
            ax.set_xlim(1,xlim[1])
            ax.set_xticks([20, 200, 500])
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


            # Create a twin axis on the right with a log scale
            if ratio2 is not None or errors is not None or ratio is not None :

                ax2 = ax.twinx()

                if errors is not None :
                    ax2.plot(x,errors/train_loss,color="brown",label="$\\mu$-train/$\\sigma$-train",marker=".",linestyle="dotted",linewidth=0.7,markersize=2)

                if ratio is not None :
                    ax2.plot(x, ratio, color="purple", label="val/train",marker=".", linestyle="dotted",linewidth=0.7,markersize=2)

                if ratio2 is not None :
                    ax2.plot(x, ratio2, color="black", label="train*/val",marker=".", linestyle="dotted",linewidth=0.7,markersize=2)
                    
                if ratio2 is not None or ratio is not None:
                    xlim = ax2.get_xlim()
                    ax2.hlines(y=1,xmin=xlim[0],xmax=xlim[1],linestyle="--",linewidth=0.7,alpha=0.5,color="black")

                ax2.set_yscale("log")
                ax2.set_ylabel("ratio")
                ax2.legend(loc="upper right")

            if title is not None :
                plt.title(title)

            plt.tight_layout()
            plt.savefig(file)
            plt.close()

        except:
            print("Some error during plotting")
        return
