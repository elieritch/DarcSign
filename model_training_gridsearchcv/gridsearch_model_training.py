# -*- coding: utf-8 -*-
"""
@author: Elie
"""

# Libraries
import datetime
import numpy as np
import pandas as pd
#plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import os
#sklearn
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
									cross_val_score, train_test_split)
from sklearn.preprocessing import label_binarize
#xgboost etc
import xgboost
from xgboost import XGBClassifier
import shap

# =============================================================================
# define these feature/headers here in case the headers 
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
	grid_search = GridSearchCV(estimator=estimator, param_grid=paramgrid, scoring = 'roc_auc', n_jobs = 60, cv = 6, verbose=True)
	fit_params={"eval_metric" : ['auc', 'error', 'logloss'], "eval_set" : [[xtest, ytest]]}
	fitted_model = grid_search.fit(xtrain, ytrain, **fit_params)
	cv_results = pd.DataFrame(fitted_model.cv_results_)
	return fitted_model.best_score_, fitted_model.best_params_, fitted_model.best_estimator_, cv_results

def model_with_params(trainX, trainY, testX, testY, params, max_rounds):
	estimator = XGBClassifier(n_estimators=max_rounds, nthread=40, **params)
	fitted_model = estimator.fit(trainX, trainY, verbose=True)
	
	prediction_binary_test = fitted_model.predict(testX, ntree_limit=max_rounds)
	prediction_probability_test = fitted_model.predict_proba(testX, ntree_limit=max_rounds)
	prediction_prob_of_true_test = prediction_probability_test[:,1]
	
	prediction_binary_train = fitted_model.predict(trainX, ntree_limit=max_rounds)
	prediction_probability_train = fitted_model.predict_proba(trainX, ntree_limit=max_rounds)
	prediction_prob_of_true_train = prediction_probability_train[:,1]
	
	return fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train

def draw_roc_curve_for_test(testY, prediction_prob_of_true_test):
	fpr, tpr, _ = roc_curve(testY, prediction_prob_of_true_test)
	roc_auc = auc(fpr, tpr)
	fig, ax = plt.subplots()
	lw = 2
	ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax.set_xlim([-0.02, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('ROC curve')
	ax.legend(loc="lower right")
	return fig, ax
	
def draw_roc_curve_for_all_data(testY, trainY, prediction_prob_of_true_test, prediction_prob_of_true_train):
	all_data = pd.concat([testY, trainY])
	all_prob_of_true = np.concatenate([prediction_prob_of_true_test, prediction_prob_of_true_train])
	fpr, tpr, _ = roc_curve(all_data, all_prob_of_true)
	roc_auc = auc(fpr, tpr)
	fig, ax = plt.subplots()
	lw = 2
	ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax.set_xlim([-0.02, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('ROC curve')
	ax.legend(loc="lower right")
	return fig, ax

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

print('Loading data at '+str(datetime.datetime.now()))
sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)
print('Finished loading data at '+str(datetime.datetime.now()))

# =============================================================================
# model gridsearch
# =============================================================================

def gridsearch_model(gene, outputdir):
	goi = str(gene)
	df_good = df.copy(deep=True)
	print('Start '+ goi + ' at '+str(datetime.datetime.now()))
	X_data, Y_labels = get_data_and_labels_from_df(df_good, goi)
	X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_labels, test_size=0.4, random_state=42, stratify=Y_labels)

	print('Grid search for '+goi+' parameters '+str(datetime.datetime.now()))
	max_rounds = 150000
	# params = { "eta": 0.001, 'max_depth': 3, 'subsample': 0.9, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'eval_metric':['auc', 'error', 'logloss'], 'nthread':12}

	parameter_grid = {'max_depth': [i for i in range(3,4)], 
					'eta': [0.001], 
					'subsample': [i.round(1) for i in np.arange(0.5,1.01,0.1)], 
					'colsample_bytree': [i.round(1) for i in np.arange(0.5,1.01,0.1)],
					'colsample_bylevel': [i.round(1) for i in np.arange(0.5,1.01,0.1)],
					'colsample_bynode': [i.round(1) for i in np.arange(0.5,1.01,0.1)],
					'seed': [i for i in range(0,50)]}
	
	best_score_, best_params_, best_estimator_, cv_results = do_grid_search_for_best_params(X_train, Y_train, X_test, Y_test, parameter_grid)
	print(goi + f" best score = {best_score_}")
	print(goi + f" best parameters = {best_params_}")
	cv_results.to_csv(outputdir+"/"+goi+"_cv_results.tsv",sep='\t', index=False)

	fitted_model, prediction_binary_test, prediction_prob_of_true_test, prediction_binary_train, prediction_prob_of_true_train = model_with_params(X_train, Y_train, X_test, Y_test, best_params_, max_rounds)

	test_df = pd.DataFrame(data={"labels":Y_test.values, "prob_of_true": prediction_prob_of_true_test, "pred_binary":prediction_binary_test})
	test_df.index = Y_test.index
	train_df = pd.DataFrame(data={"labels":Y_train.values, "prob_of_true": prediction_prob_of_true_train, "pred_binary":prediction_binary_train})
	train_df.index = Y_train.index
	all_preds_df = pd.concat([test_df, train_df])
	all_data_with_preds = pd.merge(df_good, all_preds_df, left_index=True, right_index=True)
	all_data_with_preds = all_data_with_preds.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
	all_data_with_preds.to_csv(outputdir+"/"+goi+"_predictions.tsv",sep='\t', index=False)

	fig, ax = draw_roc_curve_for_test(Y_test, prediction_prob_of_true_test)
	plt.savefig(outputdir+"/"+goi+"_test_ROC.png", dpi=500)
	plt.savefig(outputdir+"/"+goi+"_test_ROC.pdf", dpi=500)
	plt.close()
	fig, ax = draw_roc_curve_for_all_data(Y_test, Y_train, prediction_prob_of_true_test, prediction_prob_of_true_train)
	plt.savefig(outputdir+"/"+goi+"_all_data_ROC.png", dpi=500)
	plt.savefig(outputdir+"/"+goi+"_all_data_ROC.pdf", dpi=500)
	plt.close()

	CV_results = kfold_cv(10, X_data, Y_labels, fitted_model)
	print(goi +' Accuracy: %.3f (%.3f)' % (np.mean(CV_results), np.std(CV_results)))
	fig, ax = shapely_values(fitted_model, X_data, 15)
	plt.savefig(outputdir+"/"+goi+"_shap15.png", dpi=500)
	plt.savefig(outputdir+"/"+goi+"_shap15.pdf", dpi=500)

gridsearch_model("BRCA2d", outputdir)
gridsearch_model("CDK12d", outputdir)
gridsearch_model("MMRd", outputdir)