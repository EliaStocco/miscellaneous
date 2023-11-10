import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
# from reloading import reloading

# @reloading
def plot_learning_curves(arrays,file,title=None,opts=None):

    cols = ["train","val","train-2","std","ratio","ratio-2"]
    for k in cols:
        if k not in arrays:
            arrays[k] = None

    if opts is None:
        opts = {}
        opts["N"] = 1
    N = len(arrays["train"])
    if N % opts["N"] != 0 :
        return
    else :
        try :

            matplotlib.use('Agg')
            fig,ax = plt.subplots(figsize=(10,4))
            x = np.arange(len(arrays["train"]))+1

            ax.plot(x,arrays["val"],  color="red" ,label="val",  marker=".",linewidth=0.7,markersize=2,linestyle="-")
            ax.plot(x,arrays["train"],color="navy",label="$\\mu$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
            if arrays["std"] is not None :
                # ax.errorbar(x,arrays["train"],arrays["std"],color="navy",alpha=0.5,linewidth=0.5)
                ax.plot(x,arrays["std"],color="purple",label="$\\sigma$-train",marker=".",linewidth=0.7,markersize=2,linestyle="-")
            if arrays["train-2"] is not None :
                ax.plot(x,arrays["train-2"],color="green",label="train$^*$",marker=".",linewidth=0.7,markersize=2,linestyle="-")

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
            conditions = ["ratio-2","std","ratio","lr"]
            if np.any( [ arrays[k].isna().all() for k in conditions ] ):
            # if arrays["ratio-2"] is not None or arrays["std"] is not None or arrays["ratio"] is not None :
                # argv = {
                #     "linestyle":"--",
                #     "linewidth":0.7,
                #     "alpha":0.7,
                #     "marker":"x",
                # }

                ax2 = ax.twinx()

                if arrays["std"].isna().all() :
                    ax2.plot(x,arrays["std"]/arrays["train"],color="brown",label="$\\mu$-train/$\\sigma$-train",marker="x",linestyle="dashed",linewidth=0.5,markersize=2)

                if arrays["ratio"].isna().all() :
                    ax2.plot(x, arrays["ratio"], color="green", label="train/val",marker="x", linestyle="dashed",linewidth=0.5,markersize=2)

                if arrays["ratio-2"].isna().all() :
                    ax2.plot(x, arrays["ratio-2"], color="black", label="train*/val",marker="x", linestyle="dashed",linewidth=0.5,markersize=2)
                    
                if arrays["ratio-2"].isna().all() or arrays["ratio"].isna().all():
                    xlim = ax2.get_xlim()
                    ax2.hlines(y=1,xmin=xlim[0],xmax=xlim[1],linestyle="--",linewidth=0.7,alpha=0.7,color="black")
                
                if arrays["lr"].isna().all() :
                    ax2.plot(x,arrays["lr"],color="orange",label="lr",marker=".",linewidth=0.7,markersize=2,linestyle="--")

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
