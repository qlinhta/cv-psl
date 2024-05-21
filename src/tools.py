import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)


def figure_train_val(model_name, acc_train, acc_val, loss_train, loss_val, save=False):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    marker = ["o", "v"]
    color = ["blue", "red"]
    linestyle = ["--", "-"]
    markerfacecolor = ["white", "white"]
    markersize = 10
    grid = True
    epochs = len(acc_train)
    ax[0].plot(range(1, epochs + 1), acc_train, label='Train', marker=marker[0], color=color[0], linestyle=linestyle[0],
               markerfacecolor=markerfacecolor[0], markersize=markersize)
    ax[0].plot(range(1, epochs + 1), acc_val, label='Validation', marker=marker[1], color=color[1],
               linestyle=linestyle[1], markerfacecolor=markerfacecolor[1], markersize=markersize)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy (\%)')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[0].grid(grid)

    ax[1].plot(range(1, epochs + 1), loss_train, label='Train', marker=marker[0], color=color[0],
               linestyle=linestyle[0], markerfacecolor=markerfacecolor[0], markersize=markersize)
    ax[1].plot(range(1, epochs + 1), loss_val, label='Validation', marker=marker[1], color=color[1],
               linestyle=linestyle[1], markerfacecolor=markerfacecolor[1], markersize=markersize)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss')
    ax[1].legend()
    ax[1].grid(grid)
    fig.suptitle('Train and Val Metrics for ' + model_name)

    if save:
        path = os.path.join('figures', 'train_val_metrics_' + model_name + '.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')

    plt.tight_layout()
    plt.show()
