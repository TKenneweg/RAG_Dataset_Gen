from matplotlib import pyplot as plt
import numpy as np
import pickle

def plotHistogrammOverTruthfulness(filename):
    with open(filename, "rb") as f:
        dictlist = pickle.load(f)
    print(len(dictlist))
    sumscore = []
    for elem in dictlist:
        sumscore.append(elem["truthfulness"]+ elem["relevance"])
        # truthfulness.append(elem["truthfulness"])
        # truthfulness.append(elem["relevance"])

    # Use a style
    # plt.style.use('seaborn-v0_8')

    # Increase the figure size
    plt.figure(figsize=(10, 6))

    # Create histogram with 10 bins
    n_bins = 10
    # Define your bins
    bins = np.arange(2, 12, 1)- 0.5

    plt.hist(sumscore, bins=bins, color='royalblue', edgecolor='black', alpha=0.7)

    # Add mean or median line
    mean_value = np.mean(sumscore)
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_value+0.01, max(plt.ylim())/2, 'Mean', rotation=90, color='red')

    # Set plot title and axis labels
    plt.title("Histogram of Truthfulness + Relevance", fontsize=16)
    plt.xlabel("Score", fontsize=14,)
    plt.ylabel("#Answers", fontsize=14)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #ABARAG
    plotHistogrammOverTruthfulness("wikirag_090124/A_r_first300_20240125_111901_scored.pkl")
