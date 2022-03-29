# -*- coding: utf-8 -*-
# """@author: Elie"""
# run locally on python 3.8.5('dec1st_py38_xgboostetal':conda)

# %% 
# Libraries
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
import seaborn as sns
import matplotlib as mpl
#ML/Stats
from sklearn.metrics import roc_curve, auc,precision_recall_curve, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None

mpl.rcParams['savefig.transparent'] = "False"
mpl.rcParams['axes.facecolor'] = "white"
mpl.rcParams['figure.facecolor'] = "white"
mpl.rcParams['font.size'] = "5"
plt.rcParams["font.size"] = "4"
plt.rcParams['savefig.transparent'] = "False"
plt.rcParams['axes.facecolor'] = "white"
plt.rcParams['figure.facecolor'] = "white"
#%% ==========================================================
# define these feature/headers here in case the headers 
# are out of order in input files (often the case)
# ============================================================

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

#%% ==========================================================
# make concat sig dataframe 
# ============================================================

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

def predict_repair_deficiency(modelpath, data, gene):
	df = data.copy()
	features_list = snv_categories[1:] + indel_categories[1:] + cnv_categories[1:]
	x_data = df[features_list]
	x_data.columns = x_data.columns.str.replace("[", "mm").str.replace("]", "nn").str.replace(">", "rr")
	df['sample_labels'] = [1 if x == gene else 0 for x in df['label']]
	y_labels = df["sample_labels"]
	xgbmodel = xgboost.XGBClassifier()
	xgbmodel.load_model(modelpath)
	prediction_prob = xgbmodel.predict_proba(x_data, ntree_limit=1000000)
	df_probs = pd.DataFrame(data={"sample_labels":y_labels.values, "prob_of_true": prediction_prob[:,1]})
	df_probs.index = y_labels.index

	df_probs["sample"] = df["sample"]
	prob_col = f"prob_of_{gene}"
	preds = df_probs.rename(columns={"prob_of_true": prob_col})
	preds = preds[["sample", prob_col]]
	return preds, x_data, y_labels, xgbmodel

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
	"""Modified from: 	Hands-On Machine learning with Scikit-Learn
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

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

figdir = os.path.join(rootdir, "figures", "fig3")

datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
# sample_labels = sample_labels[sample_labels['manual check for usefullness (0=Fail)'] != 0]
df_pc = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)

#%%
gene = "BRCA2d"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions, xdata, ylabels, loadedmodel = predict_repair_deficiency(model_path, df_pc, gene)

fig, ax = my_roc(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_roc.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_roc.pdf"))
plt.close()

fig, ax = precision_recall(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.pdf"))
plt.close()

fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.pdf"))
plt.close()

#%%
gene = "CDK12d"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions, xdata, ylabels, loadedmodel = predict_repair_deficiency(model_path, df_pc, gene)

fig, ax = my_roc(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_roc.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_roc.pdf"))
plt.close()

fig, ax = precision_recall(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.pdf"))
plt.close()

fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.pdf"))
plt.close()

#%%
gene = "MMRd"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions, xdata, ylabels, loadedmodel = predict_repair_deficiency(model_path, df_pc, gene)

fig, ax = my_roc(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_roc.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_roc.pdf"))
plt.close()

fig, ax = precision_recall(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec.pdf"))
plt.close()

fig, ax, best_threshold, Fscore = plot_precision_recall_vs_threshold(ylabels, predictions[f"prob_of_{gene}"].values)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_PreRec_vs_Thresh.pdf"))
plt.close()

#%%