import matplotlib.pyplot as plt
import numpy as np

def tilted_loss(e, q):
    loss = max(q * e, (q - 1) * e)
    return loss

def plot_tilted_loss():
    quantiles = [.1, .5, .9]
    errors = np.arange(-3, 3.5, .5)
    losses = np.array([[tilted_loss(e, q) for e in errors] for q in quantiles])
    print(losses.shape)
    
    plt.plot(errors, losses[0,:], label=r"$\tau=0.1$")
    plt.plot(errors, losses[1,:], label=r"$\tau=0.5$")
    plt.plot(errors, losses[2,:], label=r"$\tau=0.9$")

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Time", fontsize=16)

    plt.xlabel("$MAE$")
    plt.ylabel("$\mathcal{L}_{tilted}$")

    plt.legend(prop={"size": 18})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # plt.xlim((-2.5,3))

    plt.show()

plot_tilted_loss()