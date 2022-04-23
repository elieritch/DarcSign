# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================
import pandas as pd
import numpy as np
import os
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

# ================ Toggle variables =======================================
pd.options.mode.chained_assignment = None
pd.set_option('max_columns', None)
mpl.rcParams['savefig.transparent'] = "False"
mpl.rcParams['axes.facecolor'] = "white"
mpl.rcParams['figure.facecolor'] = "white"
output_dir = os.path.dirname(__file__)


# utility functions
def conf_matrix(df, label, threshold):
    table = df.copy(deep=True)
    label = str(label)
    threshold = float(threshold)
    prob_column = f"{label}_prob_of_true"
    table["TP"] = 0
    table.loc[(table['label'] == label) & (table[prob_column] >= threshold), 'TP'] = 1
    table["FP"] = 0
    table.loc[(table['label'] != label) & (table[prob_column] >= threshold), 'FP'] = 1
    table["FN"] = 0
    table.loc[(table['label'] == label) & (table[prob_column] <= threshold), 'FN'] = 1
    table["TN"] = 0
    table.loc[(table['label'] != label) & (table[prob_column] <= threshold), 'TN'] = 1
    TP = table["TP"].sum()
    FP = table["FP"].sum()
    FN = table["FN"].sum()
    TN = table["TN"].sum()
    return np.array([[TP, FP], [FN, TN]])

def accuracy(TP, TN, FP, FN):
    return ((TP+TN)/(TP + TN + FP + FN))
def precision(TP, TN, FP, FN):
    return ((TP)/(TP + FP))
def recall(TP, TN, FP, FN):
    return ((TP)/(TP + FN))

def plot_matrix(cm_array):
    fig, ax = plt.subplots(figsize=(3, 3))
    group_names = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    group_counts = cm_array.flatten()

    labels = [f"{name}\n{count}" for name, count in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm_array, annot=labels, annot_kws={"size":8}, fmt='', cmap='Blues', ax=ax)
    ax.set_xlabel("Published labels", fontsize=8)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel("Predicted labels", fontsize=8)
    ax.set_xticklabels(["yes", "no"])
    ax.set_yticklabels(["yes", "no"])
    ax.tick_params(axis = 'both', which="major", length=0, pad=0, labelsize=8, reset=False)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=8)
    return fig, ax

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig6")
datadir = os.path.join(rootdir, "data")
probs_data = os.path.join(datadir, "darcsign_probability.tsv")
probs = pd.read_csv(probs_data, sep='\t', low_memory=False)

def calc_makefig_print(gened, threshold):
    cm = conf_matrix(probs, gened, threshold)
    fig, ax = plot_matrix(cm)
    plt.savefig(os.path.join(figdir, f"{gened}_cm.pdf"))
    plt.savefig(os.path.join(figdir, f"{gened}_cm.png"), dpi=500, transparent=False)
    # plt.close()
    TP = cm.flatten()[0]
    FP = cm.flatten()[1]
    FN = cm.flatten()[2]
    TN = cm.flatten()[3]
    print(f"{gened} accuracy = {accuracy(TP, TN, FP, FN)}")
    print(f"{gened} precision = {precision(TP, TN, FP, FN)}")
    print(f"{gened} recall = {recall(TP, TN, FP, FN)}")
    save_stats_path = os.path.join(figdir, "stats.txt")
    with open(save_stats_path, "a") as f:
        print("#==============================================", file=f)
        print(f"{gened} accuracy = {accuracy(TP, TN, FP, FN)}", file=f)
        print(f"{gened} precision = {precision(TP, TN, FP, FN)}", file=f)
        print(f"{gened} recall = {recall(TP, TN, FP, FN)}", file=f)


#thresholds
brca2d_threshold = 0.69
cdk12d_threshold = 0.09
mmrd_threshold = 0.07
if os.path.exists(os.path.join(figdir, "stats.txt")):
    os.remove(os.path.join(figdir, "stats.txt"))

calc_makefig_print("BRCA2d", brca2d_threshold)
calc_makefig_print("CDK12d", cdk12d_threshold)
calc_makefig_print("MMRd", mmrd_threshold)

#%%
