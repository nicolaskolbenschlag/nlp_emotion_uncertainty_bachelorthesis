import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plot_multiple_rators():
    r1 = list(range(11))
    r2 = np.array([0,1,2,5,4,5,4,4,2,1,0]) + 2
    r3 = np.array([0,1,2,4,4,5,4,0,3.1,3.2,3.3]) - 3
    
    raters = [r1,r2,r3]
    

    mean = np.mean(raters, axis=0)
    
    confidence = []
    for k, r in enumerate(raters):
        for r_ in raters[k+1:]:
            corr = [pd.Series(r[i-3:i]).corr(pd.Series(r_[i-3:i])) for i in range(3, len(mean)+1)]
            confidence += [corr]
        
    confidence = np.mean(confidence, axis=0)
    uncertainty = np.concatenate([[np.nan] * 2, np.abs(confidence - 1) / 2], axis=0) * 2

    for rater in raters:
        plt.plot(rater, color="gray", alpha=.5, linewidth=2)
    
    plt.plot(mean, label="Mean", color="red")
    plt.fill_between(range(len(mean)), mean - uncertainty, mean + uncertainty, label=r"Subjectivity $\frac{|S-1|}{2}$", color="orange", alpha=.2)
    
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Emotional state", fontsize=12)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.legend(prop={"size": 10})
    plt.show()

def plot_predictive_performance():
    label =         [1,2,3,4,4,5,3,6,3,8,2,4,2,5,7,8]
    prediction =    [2,4,5,2,5,3,4,4,6,8,2,3,1,4,3,2]

    confidence = np.array([pd.Series(label[i-3:i]).corr(pd.Series(prediction[i-3:i])) for i in range(3, len(label)+1)])
    uncertainty = np.concatenate([[np.nan] * 2, np.abs(confidence - 1) / 2], axis=0) * 2

    plt.plot(label, color="red", alpha=.5, label="Label")
    # plt.plot(prediction, color="blue", alpha=.5, label="Prediction")
    plt.plot(prediction, color="blue", label="Prediction")
    
    # plt.plot(mean, label="Mean", color="red")
    plt.fill_between(range(len(label)), prediction - uncertainty, prediction + uncertainty, label=r"Predictive Performance $\frac{|P-1|}{2}$", color="lightblue")
    
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Emotional state", fontsize=12)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.legend(prop={"size": 10})
    plt.show()

def plot_reliability_diagram_well_calibrated():

    bins = np.linspace(0, 1, 10)
    accs = [b + np.random.uniform(-b / 3, b / 3) for b in bins]
    plt.bar(bins, accs, width=0.1, alpha=1, edgecolor="black", color="b")

    plt.plot([0,1], [0,1], color="gray", ls="--", linewidth=5)
    
    
    plt.xlabel("Predicted Confidence", fontsize=14)
    plt.ylabel("Performance (e.g. accuracy)", fontsize=14)

    plt.show()

if __name__ == "__main__":
    plot_multiple_rators()
    # plot_predictive_performance()
    # plot_reliability_diagram_well_calibrated()