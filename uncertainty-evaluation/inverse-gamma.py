import matplotlib.pyplot as plt
import numpy as np
import math

def gamma(n: int) -> int:
    return math.factorial(n-1)

def inverse_gamma(X, a: int, b: float = 1):
    return [
        (b ** a) / gamma(a) * (1 / x) ** (a + 1) * math.exp(-b / x)
        for x in X
    ]

for a, b in [(1,1),(2,1),(3,1),(3,.5)]:

    x = np.linspace(.0001, 2, 1000)
    plt.plot(x, inverse_gamma(x, a, b), label=rf"$\alpha={a}, \beta={b}$", lw=5, alpha=.6)

plt.legend(loc="best", frameon=False, prop={"size": 24})

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.show()
