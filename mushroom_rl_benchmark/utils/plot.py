import numpy as np
import scipy.stats as st


def get_mean_and_confidence(data):
    """
    Compute the mean and 95% confidence interval

    Args:
        data (np.ndarray): Array of experiment data of shape (n_runs, n_epochs).

    Returns:
        The mean of the dataset at each epoch along with the confidence interval.

    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n-1, scale=se)
    return mean, interval


def plot_mean_conf(data, ax, color='blue', line='-', facecolor=None, alpha=0.4, label=None):
    """
    Method to plot mean and confidence interval for data on pyplot axes.

    """
    facecolor = color if facecolor is None else facecolor

    mean, conf = get_mean_and_confidence(np.array(data))
    upper_bound = mean + conf
    lower_bound = mean - conf

    ax.plot(mean, color=color, linestyle=line, label=label)
    ax.fill_between(np.arange(np.size(mean)), upper_bound, lower_bound, facecolor=facecolor, alpha=alpha)