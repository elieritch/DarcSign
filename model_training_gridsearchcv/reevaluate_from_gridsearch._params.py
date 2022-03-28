# -*- coding: utf-8 -*-
# """@author: Elie"""
# run locally on python 3.8.5('dec1st_py38_xgboostetal':conda)
# =============================================================================
# %% Libraries
# =============================================================================
import pandas as pd
import numpy as np
import datetime
from functools import partial, reduce
from joblib import load, dump
import os
import sys
#plotting
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
plt.rcParams["font.size"] = "4"
import seaborn as sns
import matplotlib as mpl
#ML/Stats
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc,precision_recall_curve, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import shap
import xgboost
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None
# import matplotlib as mpl
# mpl.matplotlib_fname()
# plt.matplotlib_fname()
mpl.rcParams['savefig.transparent'] = "False"
mpl.rcParams['axes.facecolor'] = "white"
mpl.rcParams['figure.facecolor'] = "white"
mpl.rcParams['font.size'] = "5"
plt.rcParams['savefig.transparent'] = "False"
plt.rcParams['axes.facecolor'] = "white"
plt.rcParams['figure.facecolor'] = "white"


# =============================================================================
# %% define these feature/headers here in case the headers 
# are out of order in input files (often the case)
# =============================================================================

snv_categories = ["sample", 
                "A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T", 
                "C[C>A]A", "C[C>A]C", "C[C>A]G", "C[C>A]T", 
                "G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T", 
                "T[C>A]A", "T[C>A]C", "T[C>A]G", "T[C>A]T", 
                "A[C>G]A", "A[C>G]C", "A[C>G]G", "A[C>G]T", 
                "C[C>G]A", "C[C>G]C", "C[C>G]G", "C[C>G]T", 
                "G[C>G]A", "G[C>G]C", "G[C>G]G", "G[C>G]T", 
                "T[C>G]A", "T[C>G]C", "T[C>G]G", "T[C>G]T", 
                "A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T", 
                "C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T", 
                "G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T", 
                "T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T", 
                "A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T", 
                "C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T", 
                "G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T", 
                "T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T", 
                "A[T>C]A", "A[T>C]C", "A[T>C]G", "A[T>C]T", 
                "C[T>C]A", "C[T>C]C", "C[T>C]G", "C[T>C]T", 
                "G[T>C]A", "G[T>C]C", "G[T>C]G", "G[T>C]T", 
                "T[T>C]A", "T[T>C]C", "T[T>C]G", "T[T>C]T", 
                "A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T", 
                "C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T",
                "G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T", 
                "T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"]

indel_categories = ["sample", 
                "1:Del:C:0", "1:Del:C:1", "1:Del:C:2", "1:Del:C:3", "1:Del:C:4", "1:Del:C:5", 
                "1:Del:T:0", "1:Del:T:1", "1:Del:T:2", "1:Del:T:3", "1:Del:T:4", "1:Del:T:5", 
                "1:Ins:C:0", "1:Ins:C:1", "1:Ins:C:2", "1:Ins:C:3", "1:Ins:C:4", "1:Ins:C:5", 
                "1:Ins:T:0", "1:Ins:T:1", "1:Ins:T:2", "1:Ins:T:3", "1:Ins:T:4", "1:Ins:T:5", 
                "2:Del:R:0", "2:Del:R:1", "2:Del:R:2", "2:Del:R:3", "2:Del:R:4", "2:Del:R:5", 
                "3:Del:R:0", "3:Del:R:1", "3:Del:R:2", "3:Del:R:3", "3:Del:R:4", "3:Del:R:5", 
                "4:Del:R:0", "4:Del:R:1", "4:Del:R:2", "4:Del:R:3", "4:Del:R:4", "4:Del:R:5", 
                "5:Del:R:0", "5:Del:R:1", "5:Del:R:2", "5:Del:R:3", "5:Del:R:4", "5:Del:R:5", 
                "2:Ins:R:0", "2:Ins:R:1", "2:Ins:R:2", "2:Ins:R:3", "2:Ins:R:4", "2:Ins:R:5", 
                "3:Ins:R:0", "3:Ins:R:1", "3:Ins:R:2", "3:Ins:R:3", "3:Ins:R:4", "3:Ins:R:5", 
                "4:Ins:R:0", "4:Ins:R:1", "4:Ins:R:2", "4:Ins:R:3", "4:Ins:R:4", "4:Ins:R:5", 
                "5:Ins:R:0", "5:Ins:R:1", "5:Ins:R:2", "5:Ins:R:3", "5:Ins:R:4", "5:Ins:R:5", 
                "2:Del:M:1", "3:Del:M:1", "3:Del:M:2", "4:Del:M:1", "4:Del:M:2", "4:Del:M:3", 
                "5:Del:M:1", "5:Del:M:2", "5:Del:M:3", "5:Del:M:4", "5:Del:M:5"]
                
cnv_categories = ["sample", 
                "BCper10mb_0", "BCper10mb_1", "BCper10mb_2", "BCper10mb_3", 
                "CN_0", "CN_1", "CN_2", "CN_3", "CN_4", "CN_5", "CN_6", "CN_7", "CN_8", 
                "CNCP_0", "CNCP_1", "CNCP_2", "CNCP_3", "CNCP_4", "CNCP_5", "CNCP_6", "CNCP_7", 
                "BCperCA_0", "BCperCA_1", "BCperCA_2", "BCperCA_3", "BCperCA_4", "BCperCA_5", 
                "SegSize_0", "SegSize_1", "SegSize_2", "SegSize_3", "SegSize_4", "SegSize_5", 
                "SegSize_6", "SegSize_7", "SegSize_8", "SegSize_9", "SegSize_10", 
                "CopyFraction_0", "CopyFraction_1", "CopyFraction_2", "CopyFraction_3", "CopyFraction_4", 
                "CopyFraction_5", "CopyFraction_6"]
                
### ==========================================================
# make concat sig dataframe 
# ============================================================
"""load the 3 data frames and merge to one df"""
def load_data(snv_counts_path, indel_counts_path, cnv_counts_path):
	df_snv = pd.read_csv(snv_counts_path, sep='\t', low_memory=False)
	df_snv = df_snv[snv_categories]
	df_snv["sample"] = df_snv["sample"].astype(str)
	
	df_indel = pd.read_csv(indel_counts_path, sep='\t', low_memory=False)
	df_indel = df_indel[indel_categories]
	df_indel["sample"] = df_indel["sample"].astype(str)
	
	df_cnv = pd.read_csv(cnv_counts_path, sep='\t', low_memory=False)
	df_cnv = df_cnv[cnv_categories]
	df_cnv["sample"] = df_cnv["sample"].astype(str)
	
	df_sigs = pd.merge(df_snv, df_indel, on="sample", how='left').fillna(0)
	df_sigs = pd.merge(df_sigs, df_cnv, on="sample", how='left').reset_index(drop=True)
	return df_sigs
    
def get_data_and_labels_from_df(df, gene_name):
    #first encode gene lable as binary
    combined_matrix_for_gene = df.copy(deep=True)
    gene_name = str(gene_name)
    combined_matrix_for_gene.loc[(combined_matrix_for_gene["primary_label"] == gene_name), 'primary_label'] = 1
    combined_matrix_for_gene.loc[(combined_matrix_for_gene["primary_label"] != 1), 'primary_label'] = 0
    #amazingly stupid, if dont specify astype int, the 1/0 remain an object and dont work with gridsearchcv
    combined_matrix_for_gene["primary_label"] = combined_matrix_for_gene["primary_label"].astype('int')
    #now extract 2d matrix of feature values and 1d matrix of labels
    features_list = snv_categories[1:] + indel_categories[1:] + cnv_categories[1:]
    X_data = combined_matrix_for_gene[features_list]
    X_data.columns = X_data.columns.str.replace("[", "mm").str.replace("]", "nn").str.replace(">", "rr")
    Y_labels = combined_matrix_for_gene["primary_label"]
    return X_data, Y_labels

"""Can use this function on the server with many cores, takes long time without many cores"""
def do_grid_search_for_best_params(xtrain, ytrain, xtest, ytest, paramgrid):
    estimator = XGBClassifier(objective='binary:logistic', nthread=1, seed=42)
    grid_search = GridSearchCV(estimator=estimator, param_grid=paramgrid, scoring = 'roc_auc', n_jobs = 60, cv = 10, verbose=True)
    fit_params={"eval_metric" : ['auc', 'error', 'logloss'], "eval_set" : [[xtest, ytest]]}
    fitted_model = grid_search.fit(xtrain, ytrain, **fit_params)
    cv_results = pd.DataFrame(fitted_model.cv_results_)
    return fitted_model.best_score_, fitted_model.best_params_, fitted_model.best_estimator_, cv_results

def model_with_params(trainX, trainY, testX, testY, params, max_rounds):
    estimator = XGBClassifier(n_estimators=max_rounds, nthread=10, **params)
    fitted_model = estimator.fit(trainX, trainY, verbose=True)
    
    prediction_binary_test = fitted_model.predict(testX, ntree_limit=max_rounds)
    prediction_probability_test = fitted_model.predict_proba(testX, ntree_limit=max_rounds)
    prediction_prob_of_true_test = prediction_probability_test[:,1]
    
    prediction_binary_train = fitted_model.predict(trainX, ntree_limit=max_rounds)
    prediction_probability_train = fitted_model.predict_proba(trainX, ntree_limit=max_rounds)
    prediction_prob_of_true_train = prediction_probability_train[:,1]
    
    return fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train

def kfold_cv(Knumber, Xdata, Ylabels, model):
    kfold = KFold(n_splits=Knumber)
    results = cross_val_score(model, Xdata, Ylabels, cv=kfold)
    return results

def shapely_values(model, Xdata, Nvalues):
    import inspect
    print(os.path.abspath(inspect.getfile(shap.summary_plot)))
    X = Xdata.copy(deep=True)
    shap_values = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent').shap_values(X, check_additivity=False)
    X.columns = X.columns.str.replace("mm", "[").str.replace("nn", "]").str.replace("rr", ">")
    fig, ax = plt.subplots(figsize=(7,4))
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=Nvalues, show=False, plot_size=(6,3), alpha=0.7)
    plt.subplots_adjust(left=0.3, right=0.94, top=0.9, bottom=0.1)
    ax = plt.gca()
    fig = plt.gcf()
    mpl.rcParams['savefig.transparent'] = "False"
    mpl.rcParams['axes.facecolor'] = "white"
    mpl.rcParams['figure.facecolor'] = "white"
    mpl.rcParams['font.size'] = "5"
    plt.rcParams['savefig.transparent'] = "False"
    plt.rcParams['axes.facecolor'] = "white"
    plt.rcParams['figure.facecolor'] = "white"
    return fig, ax

def my_roc(data, prob_of_true):
    fpr, tpr, thresholds = roc_curve(data, prob_of_true)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(1.3,1.4))
    lw = 1
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=4, labelpad=0.75)
    ax.set_ylabel('True Positive Rate', fontsize=4, labelpad=0.75)
    #ax.set_title('ROC curve', fontsize=6, pad=1)
    ax.legend(loc="lower right", fontsize=4)
    tick_numbers = [round(x,1) for x in np.arange(0, 1.1, 0.2)]
    ax.set_xticks(tick_numbers)
    ax.tick_params(axis='both', which="major", length=2, labelsize=4, pad=0.5, reset=False)
    fig.subplots_adjust(left=0.15, right=0.965, top=0.98, bottom=0.12)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    return fig, ax

def precision_recall(data, prob_of_true):
    precision, recall, thresholds = precision_recall_curve(data, prob_of_true)
    fig, ax = plt.subplots(figsize=(1.3,1.4))
    lw = 1    
    # ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.plot(recall, precision, color='darkorange', lw=lw, label='PR curve')
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.5, 1.05])
    # axis labels
    ax.set_xlabel('Recall', fontsize=4, labelpad=0.75)
    ax.set_ylabel('Precision', fontsize=4, labelpad=0.75)
    ax.legend(loc="lower left", fontsize=4)
    tick_numbers = [round(x,1) for x in np.arange(0, 1.1, 0.2)]
    ax.set_xticks(tick_numbers)
    ax.tick_params(axis='both', which="major", length=2, labelsize=4, pad=0.5, reset=False)
    fig.subplots_adjust(left=0.15, right=0.965, top=0.98, bottom=0.12)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    return fig, ax

def plot_precision_recall_vs_threshold(data, prob_of_true):
    """Modified from:     Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    #first generate and find fscores for all possible thresholds:
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    #evaluate each threshold
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(data, to_labels(prob_of_true, t)) for t in thresholds]
    ix = np.argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    best_threshold = thresholds[ix]
    Fscore = scores[ix]
    
    #now plot precision recall as a function of threshold
    precisions, recalls, thresholds = precision_recall_curve(data, prob_of_true)
    fig, ax = plt.subplots(figsize=(1.3,1.4))
    lw = 1
    #plt.title("Precision and Recall Scores as a function of the decision threshold")
    ax.plot(thresholds, precisions[:-1], color="#CD5C5C", label="Precision", lw=lw)
    ax.plot(thresholds, recalls[:-1], "#197419", label="Recall", lw=lw)
    ax.axvline(x=best_threshold, color="b",linestyle="--", label=f'Threshold={best_threshold:.2f},\nF-Score={Fscore:.2f}')
    ax.set_ylabel("Score", fontsize=4, labelpad=0.75)
    ax.set_xlabel("Decision Threshold", fontsize=4, labelpad=0.75)
    ax.legend(loc="lower center", fontsize=4)
    tick_numbers = [round(x,1) for x in np.arange(0, 1.1, 0.2)]
    ax.set_xticks(tick_numbers)
    ax.tick_params(axis='both', which="major", length=2, labelsize=4, pad=0.5, reset=False)
    fig.subplots_adjust(left=0.15, right=0.965, top=0.98, bottom=0.12)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    return fig, ax, best_threshold, Fscore

def makepredictions(loadedmodel, dfgood, xdata, ylabels):
    prediction_probability = loadedmodel.predict_proba(xdata)
    pred_prob = prediction_probability[:,1]
    allpredprob_df = pd.DataFrame(data={"labels":ylabels.values, "prob_of_true": pred_prob})
    all_data_with_preds = pd.merge(dfgood, allpredprob_df, left_index=True, right_index=True)
    pred_data = all_data_with_preds[["sample", "primary_label", "prob_of_true"]]
    pred_data["primary_label"] = pred_data["primary_label"].fillna("DRp")
    all_data_with_preds = all_data_with_preds.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
    return pred_prob, pred_data

def least_sub_rank1_model_params(cv_results_path):
    rank1_cv_results = pd.read_csv(cv_results_path, sep="\t").query('(rank_test_score < 2)').query('(param_colsample_bylevel > 0.3) and (param_colsample_bynode > 0.3) and (param_colsample_bytree > 0.3) and (param_subsample > 0.3)')
    rank1_cv_results["total_subsample"] = rank1_cv_results['param_colsample_bylevel'] * rank1_cv_results['param_colsample_bynode'] * rank1_cv_results['param_colsample_bytree'] * rank1_cv_results['param_subsample']
    rank1_cv_results = rank1_cv_results.sort_values(by="total_subsample", ascending=False).head(n=1)
    params = rank1_cv_results["params"].iloc[0]
    params_dict = eval(params)
    return params_dict

def probability_bar_graph(gene_oi, pos_color, neg_color, legend_d, legend_p, all_data_with_preds):
    all_prob_table = all_data_with_preds.copy(deep=True)
    pos = all_prob_table.query('(primary_label == @gene_oi)').sort_values(f"{gene_oi}_prob_of_true", ascending=False)
    pos["color"] = pos_color
    neg = all_prob_table.query('(primary_label != @gene_oi)').sort_values(f"{gene_oi}_prob_of_true", ascending=False)
    neg["color"] = neg_color
    bargraph = pd.concat([pos, neg]).reset_index(drop=True)

    def fig_aesthetic(ax, df):
        ax.set_ylim(0,1)
        ax.set_xlim(df.index[0]-0.5,df.index[-1]+0.5)
        ax.grid(b=False, which='both', axis='y', color='0.4', linewidth=0.9, linestyle='dotted', zorder=0)
        ax.tick_params(axis='both', which="major", length=3, labelsize=5, pad=1, reset=False)
        ax.set_xticks([])
        # ax[0].set_ylabel("Signature Weights", fontsize=8, horizontalalignment="center", labelpad=0.5)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_xlabel("")
        ax.set_ylabel("Probability", fontsize=5, horizontalalignment="center", labelpad=0.6)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        return ax

    fig, ax = plt.subplots(figsize=(3.2,1.5))
    ax.bar(x=bargraph.index, height=bargraph[f"{gene_oi}_prob_of_true"], width=0.8, edgecolor=None, linewidth=0, color=bargraph["color"], zorder=10)
    ax = fig_aesthetic(ax, bargraph)
    handles = []
    handles.append(mlines.Line2D([], [], color=pos_color, markeredgecolor=pos_color, marker='s', lw=0, markersize=8, label=legend_d))
    handles.append(mlines.Line2D([], [], color=neg_color, markeredgecolor=neg_color, marker='s', lw=0, markersize=8, label=legend_p))
    ax.legend(handles=handles,loc='upper left', edgecolor='0.5', frameon=False, ncol=2, fontsize=5, handletextpad=0.001, bbox_to_anchor=(0.45, 0.72), borderpad=0, columnspacing=0.9)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.99, bottom=0.03)
    return fig, ax

def conf_matrix(df, label, threshold):
    table = df.copy(deep=True) #df is all data_with_preds
    label = str(label) #label is primary label column
    threshold = float(threshold)
    prob_column = f"{label}_prob_of_true"
    table["TP"] = 0
    table.loc[(table['primary_label'] == label) & (table[prob_column] >= threshold), 'TP'] = 1
    table["FP"] = 0
    table.loc[(table['primary_label'] != label) & (table[prob_column] >= threshold), 'FP'] = 1
    table["FN"] = 0
    table.loc[(table['primary_label'] == label) & (table[prob_column] <= threshold), 'FN'] = 1
    table["TN"] = 0
    table.loc[(table['primary_label'] != label) & (table[prob_column] <= threshold), 'TN'] = 1
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

### ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")
outputdir = os.path.dirname(__file__)
cv_results_dir = os.path.dirname(__file__)
print('Loading data at '+str(datetime.datetime.now()))
sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)

print('Finished loading data at '+str(datetime.datetime.now()))

all_probabilites_list = []

# color list for bargraphs
color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drp
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr    

# %% 
# model BRCA2
# =============================================================================

goi = "BRCA2d"
goi = str(goi)

print('Loading data at '+str(datetime.datetime.now()))
sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)
print('Finished loading data at '+str(datetime.datetime.now()))

print(f"start splitting data for {goi} at {str(datetime.datetime.now())}")
X_data, Y_labels = get_data_and_labels_from_df(df_good, goi)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_labels, test_size=0.4, random_state=42, stratify=Y_labels)

# model_path = "c:/Users/ElieRitch/Desktop/signatures_aug2021/gridsearch_models6/BRCA2_gridparams_refitmodel.joblib.model.dat"
# modelpath = os.path.expanduser(model_path)
# model = load(modelpath)
# PredProbs, PredData = makepredictions(model, df_good, X_data, Y_labels)

print(f"start making model for {goi} at {str(datetime.datetime.now())}")
max_rounds = 1000000
# cv_grid_path = f"{cv_results_dir}/{goi}_cv_results.tsv"
# best_params_ = least_sub_rank1_model_params(cv_grid_path)
best_params_ = {'colsample_bylevel': 0.3, 'colsample_bynode': 0.3, 'colsample_bytree': 0.3, 'eta': 0.001, 'max_depth': 3, 'seed': 32, 'subsample': 0.4}
fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train = model_with_params(X_train, Y_train, X_test, Y_test, best_params_, max_rounds)

test_df = pd.DataFrame(data={"labels":Y_test.values, "prob_of_true": prediction_prob_of_true_test, "pred_binary":prediction_binary_test})
test_df.index = Y_test.index
train_df = pd.DataFrame(data={"labels":Y_train.values, "prob_of_true": prediction_prob_of_true_train, "pred_binary":prediction_binary_train})
train_df.index = Y_train.index
all_preds_df = pd.concat([test_df, train_df])
all_data_with_preds = pd.merge(df_good, all_preds_df, left_index=True, right_index=True)
all_data_with_preds = all_data_with_preds.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
all_data_with_preds = all_data_with_preds.drop(columns="labels").rename(columns={"prob_of_true": goi+"_prob_of_true", "pred_binary": goi+"_pred_binary"})
all_probabilites_list.append(all_data_with_preds)
all_data_with_preds.to_csv(outputdir+"/"+goi+"_predictions.tsv",sep='\t', index=False)

saved_model_path = outputdir+"/"+goi+".joblib.model.dat"
dump(fitted_model, saved_model_path)

all_data = pd.concat([Y_test, Y_train])
all_prob_of_true = np.concatenate([prediction_prob_of_true_test, prediction_prob_of_true_train])
print(f"finished making model for {goi} at {str(datetime.datetime.now())}")

#####ROC for all data and for test ##############
print(f"start graphing model for {goi} at {str(datetime.datetime.now())}")
fig, ax = my_roc(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_ROC.png", dpi=500)
plt.close()
fig, ax = my_roc(Y_test, prediction_prob_of_true_test)
plt.savefig(outputdir+"/"+goi+"_test_ROC.png", dpi=500)
plt.close()
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
plt.close()
# plt.savefig(outputdir+"/"+goi+"_PreRec.pdf", dpi=500)
plt.close()
fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.pdf", dpi=500)
plt.close()
print(f"start graphing shap for {goi} at {str(datetime.datetime.now())}")
fig, ax = shapely_values(fitted_model, X_data, 15)
ax.set_xticks([-0.5, 0,0.5,1])
plt.savefig(outputdir+"/"+goi+"_shap15.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_shap15.pdf", dpi=500)
plt.close()
print(f"start graphing bars for {goi} at {str(datetime.datetime.now())}")
fig, ax = probability_bar_graph(goi, red, blue, f"{goi}d", f"{goi}p", all_data_with_preds)
plt.savefig(f"{outputdir}/{goi}_prob_of_class.png", dpi=500, transparent=False, facecolor="w")
plt.close()
print(f"finished graphing model for {goi} at {str(datetime.datetime.now())}")

print(f"Confusion metric and graph for {goi} at {str(datetime.datetime.now())}")
confusion_matrix = conf_matrix(all_data_with_preds, goi, best_threshold)
TruePos = confusion_matrix.flatten()[0]
FalsePos = confusion_matrix.flatten()[1]
FalseNeg = confusion_matrix.flatten()[2]
TrueNeg = confusion_matrix.flatten()[3]
accuracy_of_model = accuracy(TruePos, TrueNeg, FalsePos, FalseNeg)
precision_of_model = precision(TruePos, TrueNeg, FalsePos, FalseNeg)
recall_of_model = recall(TruePos, TrueNeg, FalsePos, FalseNeg)
print(confusion_matrix)
print(f"{goi} model accuracy = {accuracy_of_model}")
print(f"{goi} model precision = {precision_of_model}")
print(f"{goi} model recall = {recall_of_model}")
fig, ax = plot_matrix(confusion_matrix)
plt.savefig(f"{outputdir}/{goi}_confusion_matrix.png", dpi=500, transparent=False, facecolor="w")
plt.close()


# %% 
# model CDK12
# =============================================================================

goi = "CDK12"
goi = str(goi)

print('Loading data at '+str(datetime.datetime.now()))
sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)
print('Finished loading data at '+str(datetime.datetime.now()))

print('Start '+ goi + ' at '+str(datetime.datetime.now()))
X_data, Y_labels = get_data_and_labels_from_df(df_good, goi)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_labels, test_size=0.4, random_state=42, stratify=Y_labels)

print(f"start making model for {goi} at {str(datetime.datetime.now())}")
max_rounds = 1000000
# cv_grid_path = f"{cv_results_dir}/{goi}_cv_results.tsv"
# best_params_ = least_sub_rank1_model_params(cv_grid_path)
best_params_ = {'colsample_bylevel': 0.6, 'colsample_bynode': 0.9, 'colsample_bytree': 0.7, 'eta': 0.001, 'max_depth': 3, 'seed': 47, 'subsample': 0.7}
fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train = model_with_params(X_train, Y_train, X_test, Y_test, best_params_, max_rounds)

test_df = pd.DataFrame(data={"labels":Y_test.values, "prob_of_true": prediction_prob_of_true_test, "pred_binary":prediction_binary_test})
test_df.index = Y_test.index
train_df = pd.DataFrame(data={"labels":Y_train.values, "prob_of_true": prediction_prob_of_true_train, "pred_binary":prediction_binary_train})
train_df.index = Y_train.index
all_preds_df = pd.concat([test_df, train_df])
all_data_with_preds = pd.merge(df_good, all_preds_df, left_index=True, right_index=True)
all_data_with_preds = all_data_with_preds.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
all_data_with_preds = all_data_with_preds.drop(columns="labels").rename(columns={"prob_of_true": goi+"_prob_of_true", "pred_binary": goi+"_pred_binary"})
all_probabilites_list.append(all_data_with_preds)
all_data_with_preds.to_csv(outputdir+"/"+goi+"_predictions.tsv",sep='\t', index=False)

saved_model_path = outputdir+"/"+goi+".joblib.model.dat"
dump(fitted_model, saved_model_path)

all_data = pd.concat([Y_test, Y_train])
all_prob_of_true = np.concatenate([prediction_prob_of_true_test, prediction_prob_of_true_train])
print(f"finished making model for {goi} at {str(datetime.datetime.now())}")

#####ROC for all data and for test ##############
print(f"start graphing model for {goi} at {str(datetime.datetime.now())}")
fig, ax = my_roc(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_ROC.png", dpi=500)
plt.close()
fig, ax = my_roc(Y_test, prediction_prob_of_true_test)
plt.savefig(outputdir+"/"+goi+"_test_ROC.png", dpi=500)
plt.close()
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
plt.close()
# plt.savefig(outputdir+"/"+goi+"_PreRec.pdf", dpi=500)
plt.close()
fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.pdf", dpi=500)
plt.close()
print(f"start graphing shap for {goi} at {str(datetime.datetime.now())}")
fig, ax = shapely_values(fitted_model, X_data, 15)
ax.set_xticks([-0.5, 0,0.5,1])
plt.savefig(outputdir+"/"+goi+"_shap15.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_shap15.pdf", dpi=500)
plt.close()
print(f"start graphing bars for {goi} at {str(datetime.datetime.now())}")
fig, ax = probability_bar_graph(goi, green, blue, f"{goi}d", f"{goi}p", all_data_with_preds)
plt.savefig(f"{outputdir}/{goi}_prob_of_class.png", dpi=500, transparent=False, facecolor="w")
plt.close()
print(f"finished graphing model for {goi} at {str(datetime.datetime.now())}")

print(f"Confusion metric and graph for {goi} at {str(datetime.datetime.now())}")
confusion_matrix = conf_matrix(all_data_with_preds, goi, best_threshold)
TruePos = confusion_matrix.flatten()[0]
FalsePos = confusion_matrix.flatten()[1]
FalseNeg = confusion_matrix.flatten()[2]
TrueNeg = confusion_matrix.flatten()[3]
accuracy_of_model = accuracy(TruePos, TrueNeg, FalsePos, FalseNeg)
precision_of_model = precision(TruePos, TrueNeg, FalsePos, FalseNeg)
recall_of_model = recall(TruePos, TrueNeg, FalsePos, FalseNeg)
print(confusion_matrix)
print(f"{goi} model accuracy = {accuracy_of_model}")
print(f"{goi} model precision = {precision_of_model}")
print(f"{goi} model recall = {recall_of_model}")
fig, ax = plot_matrix(confusion_matrix)
plt.savefig(f"{outputdir}/{goi}_confusion_matrix.png", dpi=500, transparent=False, facecolor="w")
plt.close()

#%% MMRD
goi = "MMRd"
goi = str(goi)

print('Loading data at '+str(datetime.datetime.now()))
sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)
print('Finished loading data at '+str(datetime.datetime.now()))

print(f"start splitting data for {goi} at {str(datetime.datetime.now())}")
X_data, Y_labels = get_data_and_labels_from_df(df_good, goi)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_labels, test_size=0.4, random_state=42, stratify=Y_labels)

# model_path = "c:/Users/ElieRitch/Desktop/signatures_aug2021/gridsearch_models6/BRCA2_gridparams_refitmodel.joblib.model.dat"
# modelpath = os.path.expanduser(model_path)
# model = load(modelpath)
# PredProbs, PredData = makepredictions(model, df_good, X_data, Y_labels)
print(f"start making model for {goi} at {str(datetime.datetime.now())}")
max_rounds = 1000000
# cv_grid_path = f"{cv_results_dir}/{goi}_cv_results.tsv"
# best_params_ = least_sub_rank1_model_params(cv_grid_path)
best_params_ = {'colsample_bylevel': 0.3, 'colsample_bynode': 0.3, 'colsample_bytree': 0.3, 'eta': 0.001, 'max_depth': 3, 'seed': 0, 'subsample': 0.4}
fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train = model_with_params(X_train, Y_train, X_test, Y_test, best_params_, max_rounds)

test_df = pd.DataFrame(data={"labels":Y_test.values, "prob_of_true": prediction_prob_of_true_test, "pred_binary":prediction_binary_test})
test_df.index = Y_test.index
train_df = pd.DataFrame(data={"labels":Y_train.values, "prob_of_true": prediction_prob_of_true_train, "pred_binary":prediction_binary_train})
train_df.index = Y_train.index
all_preds_df = pd.concat([test_df, train_df])
all_data_with_preds = pd.merge(df_good, all_preds_df, left_index=True, right_index=True)
all_data_with_preds = all_data_with_preds.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
all_data_with_preds = all_data_with_preds.drop(columns="labels").rename(columns={"prob_of_true": goi+"_prob_of_true", "pred_binary": goi+"_pred_binary"})
all_probabilites_list.append(all_data_with_preds)
all_data_with_preds.to_csv(outputdir+"/"+goi+"_predictions.tsv",sep='\t', index=False)

saved_model_path = outputdir+"/"+goi+".joblib.model.dat"
dump(fitted_model, saved_model_path)

all_data = pd.concat([Y_test, Y_train])
all_prob_of_true = np.concatenate([prediction_prob_of_true_test, prediction_prob_of_true_train])
print(f"finished making model for {goi} at {str(datetime.datetime.now())}")

#####ROC for all data and for test ##############
print(f"start graphing model for {goi} at {str(datetime.datetime.now())}")
fig, ax = my_roc(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_ROC.png", dpi=500)
plt.close()
fig, ax = my_roc(Y_test, prediction_prob_of_true_test)
plt.savefig(outputdir+"/"+goi+"_test_ROC.png", dpi=500)
plt.close()
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
fig, ax = precision_recall(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec.png", dpi=500)
plt.close()
# plt.savefig(outputdir+"/"+goi+"_PreRec.pdf", dpi=500)
plt.close()
fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(all_data, all_prob_of_true)
plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_PreRec_vs_Thresh.pdf", dpi=500)
plt.close()
print(f"start graphing shap for {goi} at {str(datetime.datetime.now())}")
fig, ax = shapely_values(fitted_model, X_data, 15)
ax.set_xticks([-0.5, 0,0.5,1])
plt.savefig(outputdir+"/"+goi+"_shap15.png", dpi=500)
# plt.savefig(outputdir+"/"+goi+"_shap15.pdf", dpi=500)
plt.close()
print(f"start graphing bars for {goi} at {str(datetime.datetime.now())}")
fig, ax = probability_bar_graph(goi, purple, blue, f"MMRd", f"MMRp", all_data_with_preds)
plt.savefig(f"{outputdir}/{goi}_prob_of_class.png", dpi=500, transparent=False, facecolor="w")
plt.close()
print(f"finished graphing model for {goi} at {str(datetime.datetime.now())}")

print(f"Confusion metric and graph for {goi} at {str(datetime.datetime.now())}")
confusion_matrix = conf_matrix(all_data_with_preds, goi, best_threshold)
TruePos = confusion_matrix.flatten()[0]
FalsePos = confusion_matrix.flatten()[1]
FalseNeg = confusion_matrix.flatten()[2]
TrueNeg = confusion_matrix.flatten()[3]
accuracy_of_model = accuracy(TruePos, TrueNeg, FalsePos, FalseNeg)
precision_of_model = precision(TruePos, TrueNeg, FalsePos, FalseNeg)
recall_of_model = recall(TruePos, TrueNeg, FalsePos, FalseNeg)
print(confusion_matrix)
print(f"{goi} model accuracy = {accuracy_of_model}")
print(f"{goi} model precision = {precision_of_model}")
print(f"{goi} model recall = {recall_of_model}")
fig, ax = plot_matrix(confusion_matrix)
plt.savefig(f"{outputdir}/{goi}_confusion_matrix.png", dpi=500, transparent=False, facecolor="w")
plt.close()
#%%