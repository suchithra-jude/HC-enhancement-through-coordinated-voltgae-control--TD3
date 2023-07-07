import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file, name):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.clf()
    plt.plot(x, running_avg)
    plt.title('Running average of previous 10 ' + name)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.savefig(figure_file)

def plot_result_curve(x, scores, figure_file, name):
    plt.clf()
    plt.plot(x, scores)
    plt.title('Running average of previous 10 ' + name)
    plt.xlabel('Episodes')
    plt.ylabel(name)
    plt.savefig(figure_file)