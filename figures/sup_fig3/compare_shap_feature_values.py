# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================
# Libraries
import pandas as pd
import numpy as np
import os
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator
import matplotlib.lines as mlines
# stats
from scipy import stats
pd.options.mode.chained_assignment = None

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

#%% ==========================================================
# figure functions and calculations
# ============================================================

def make_seabornstyle_table(df, feature_list):
	table_list = []
	for i, j in enumerate(feature_list):
		#adding primary_label_actual to have two pos/neg and each subtype seperately
		df_feature = df[["sample", "label", "primary_label_actual", feature_list[i]]]
		df_feature["feature"] = feature_list[i]
		df_feature = df_feature.rename(columns={feature_list[i]:"feature_value"})
		table_list.append(df_feature)
	graphtable = pd.concat(table_list)
	return graphtable

class MinorSymLogLocator(Locator):
	"""
	Dynamically find minor tick positions based on the positions of
	major ticks for a symlog scaling.
	"""
	def __init__(self, linthresh):
		"""
		Ticks will be placed between the major ticks.
		The placement is linear for x between -linthresh and linthresh,
		otherwise its logarithmically
		"""
		self.linthresh = linthresh

	def __call__(self):
		'Return the locations of the ticks'
		majorlocs = self.axis.get_majorticklocs()

		# iterate through minor locs
		minorlocs = []

		# handle the lowest part
		for i in range(1, len(majorlocs)):
			majorstep = majorlocs[i] - majorlocs[i-1]
			if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
				ndivs = 10
			else:
				ndivs = 9
			minorstep = majorstep / ndivs
			locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
			minorlocs.extend(locs)

		return self.raise_if_exceeds(np.array(minorlocs))

	def tick_values(self, vmin, vmax):
		raise NotImplementedError('Cannot get tick locations for a '
								  '%s type.' % type(self))

#same color list as always
color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #DRwt
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr

def common_settings(fig, ax):
	# fig.set_size_inches(6, 2)
	ax.set_yscale('symlog', linthresh=10, base=10)
	yaxis = plt.gca().yaxis
	yaxis.set_minor_locator(MinorSymLogLocator(10e1))
	ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=7, ha='right', va='top', rotation_mode="anchor")
	ax.tick_params(axis='both', which="major", length=3, labelsize=7, pad=0.5, reset=False)
	# ax.set_ylim([-2, 35])
	ax.set_xlabel("")
	ax.set_ylabel("Feature value", labelpad=0, fontsize=7)
	ax.set_ylim(bottom=-0.9)
	sns.despine(top=True, right=True)
	fig.subplots_adjust(left=0.05, top=0.97, bottom=0.16, right=0.96)
	return fig, ax
	
def make_feature_pvalues(graphtable, feature_list, gene_def, gene_pos):
	pvalues = []
	gene_def = str(gene_def)
	gene_pos = str(gene_pos)
	for feature in feature_list:
		# print(f"Mann-Whitney of BRCA2 for {feature}")
		feature_table = graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == @gene_def)')["feature_value"]
		proficient = feature_table.query('(label == @gene_pos)')["feature_value"]
		# print(stats.mannwhitneyu(deficient, proficient))
		u, p = stats.mannwhitneyu(deficient, proficient) 
		pvalues.append(p)
	feature_pvalues = dict(zip(feature_list, pvalues))
	return feature_pvalues
	
def plot_legend():
	handles = []
	handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='o', lw=0, markersize=5, label='DRwt'))
	handles.append(mlines.Line2D([], [], color=orange, markeredgecolor=orange, marker='o', lw=0, markersize=5, label='ATMd'))
	handles.append(mlines.Line2D([], [], color=green, markeredgecolor=green, marker='o', lw=0, markersize=5, label='CDK12d'))
	handles.append(mlines.Line2D([], [], color=red, markeredgecolor=red, marker='o', lw=0, markersize=5, label='BRCA2d'))
	handles.append(mlines.Line2D([], [], color=purple, markeredgecolor=purple, marker='o', lw=0, markersize=5, label='MMRd'))
	plt.legend(handles=handles,loc=2, edgecolor='0.5', fancybox=True, frameon=False, facecolor='white', ncol=1, fontsize=7, labelspacing=0.1, handletextpad=-0.2, columnspacing=0.5, bbox_to_anchor=(0.94,0.95))

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig3")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)

#%% ==========================================================
# Calcs and table manipulations
# ============================================================

#in order to display the fraction differences make them into percentages
cols = df.columns[df.columns.str.contains('CopyFraction')]
df[cols] = df[cols] * 100

brca2_shap_features = ["A[C>G]G", "5:Del:R:0", "CopyFraction_3", "T[C>G]A", "C[C>A]C", "T[C>G]T", "CopyFraction_5", "C[C>G]T", "SegSize_2"]
brca2_table = df.copy(deep=True)
brca2_table["primary_label_actual"] = brca2_table["label"]
brca2_table.loc[brca2_table["label"] != "BRCA2d", "label"] = "BRCA2p"
brca_graphtable = make_seabornstyle_table(brca2_table, brca2_shap_features)

cdk12_shap_features = ["CN_2", "CopyFraction_2", "SegSize_0", "G[T>G]C", "CNCP_3", "1:Ins:C:1", "BCperCA_0", "1:Del:T:2", "CNCP_1"]
cdk12_table = df.copy(deep=True)
cdk12_table["primary_label_actual"] = cdk12_table["label"]
cdk12_table.loc[cdk12_table["label"] != "CDK12d", "label"] = "CDK12p"
cdk12_graphtable = make_seabornstyle_table(cdk12_table, cdk12_shap_features)

mmrd_shap_features = ["1:Del:C:5", "G[C>T]G", "G[T>C]G", "A[C>T]G", "CN_6", "G[C>T]T", "CopyFraction_6", "C[C>T]T", "C[C>T]G"]
mmrd_table = df.copy(deep=True)
mmrd_table["primary_label_actual"] = mmrd_table["label"]
mmrd_table.loc[mmrd_table["label"] != "MMRd", "label"] = "MMRp"
mmrd_graphtable = make_seabornstyle_table(mmrd_table, mmrd_shap_features)

#%% ==========================================================
# Plots, too differnt for a single plotting function
# ============================================================

# ============================================================
# BRCA2d plot
grey="#CACACA"
face_pal = {'BRCA2d': grey, 'BRCA2p': grey}
hue_order = ['BRCA2d', 'BRCA2p']
stripplot_kwargs = {'linewidth': 0, 'size': 4, 'alpha': 0.6, 'hue_order': hue_order}

fig, ax = plt.subplots(figsize=(7.1,2.5))
# sns.boxplot(x='feature', y='feature_value', hue='label', data=brca_graphtable, ax=ax, fliersize=0, **boxplot_kwargs)
sns.violinplot(x='feature', y='feature_value', hue='label', data=brca_graphtable, ax=ax, color="lightgrey", scale="width", hue_order=hue_order, width=0.8, cut=0, bw=.3, linewidth=0, inner=None, split=False, palette=face_pal)
ax.set_alpha(0.6)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "BRCA2d")'), ax=ax, jitter=0.15, dodge=True, palette={'BRCA2d': red, 'BRCA2p': "white"}, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "DRwt")'), ax=ax, jitter=0.15, dodge=True, color=blue, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "ATMd")'), ax=ax, jitter=0.15, dodge=True, color=orange, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "CDK12d")'), ax=ax, jitter=0.15, dodge=True, color=green, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "MMRd")'), ax=ax, jitter=0.15, dodge=True, color=purple, **stripplot_kwargs)

brca_feature_pvalues = make_feature_pvalues(brca_graphtable, brca2_shap_features, "BRCA2d", "BRCA2p")

for i in ax.get_xticks():
	xstart = i-0.2
	xend = i+0.2
	feat=brca2_shap_features[i]
	height=brca_graphtable.query('(feature == @feat)')["feature_value"].max()
	if height > 100:
		y=600
		h=100
	if height < 100:
		y=90
		h=20
	if height < 10:
		y=12
		h=2
	col='k'
	ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=col)
	if brca_feature_pvalues[feat] > 0.05:
		annot="n.s."
	if brca_feature_pvalues[feat] <= 0.05:
		annot="*"
	if brca_feature_pvalues[feat] <= 0.01:
		annot="**"
	if brca_feature_pvalues[feat] <= 0.001:
		annot="***"
	ax.text((xstart+xend)*.5, y+h-1, annot, ha='center', va='bottom', color=col, fontsize=7, linespacing=0)

plot_legend()
common_settings(fig, ax)
plt.savefig(os.path.join(figdir, "brca2_shapfeature_values.pdf"))
plt.savefig(os.path.join(figdir, "brca2_shapfeature_values.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

# ============================================================
# CDK12d plot

grey="#CACACA"
face_pal = {'CDK12d': grey, 'CDK12p': grey}
hue_order = ['CDK12d', 'CDK12p']
stripplot_kwargs = {'linewidth': 0, 'size': 4, 'alpha': 0.6, 'hue_order': hue_order}

fig, ax = plt.subplots(figsize=(7.1,2.5))
# sns.boxplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable, ax=ax, fliersize=0, **boxplot_kwargs)
sns.violinplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable, ax=ax, color="lightgrey", scale="width", hue_order=hue_order, width=0.8, cut=0, bw=.3, linewidth=0, inner=None, split=False, palette=face_pal)
ax.set_alpha(0.6)
sns.stripplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable.query('(primary_label_actual == "CDK12d")'), ax=ax, jitter=0.15, dodge=True, palette={'CDK12d': green, 'CDK12p': "white"}, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable.query('(primary_label_actual == "DRwt")'), ax=ax, jitter=0.15, dodge=True, color=blue, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable.query('(primary_label_actual == "ATMd")'), ax=ax, jitter=0.15, dodge=True, color=orange, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable.query('(primary_label_actual == "BRCA2d")'), ax=ax, jitter=0.15, dodge=True, color=red, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=cdk12_graphtable.query('(primary_label_actual == "MMRd")'), ax=ax, jitter=0.15, dodge=True, color=purple, **stripplot_kwargs)

cdk12_feature_pvalues = make_feature_pvalues(cdk12_graphtable, cdk12_shap_features, "CDK12d", "CDK12p")

for i in ax.get_xticks():
	xstart = i-0.2
	xend = i+0.2
	feat=cdk12_shap_features[i]
	height=cdk12_graphtable.query('(feature == @feat)')["feature_value"].max()
	if height > 100:
		y=400
		h=100
	if height < 100:
		y=140
		h=20
	if height < 10:
		y=12
		h=2
	col='k'
	ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=col)
	if cdk12_feature_pvalues[feat] > 0.05:
		annot="n.s."
	if cdk12_feature_pvalues[feat] <= 0.05:
		annot="*"
	if cdk12_feature_pvalues[feat] <= 0.01:
		annot="**"
	if cdk12_feature_pvalues[feat] <= 0.001:
		annot="***"
	ax.text((xstart+xend)*.5, y+h, annot, ha='center', va='bottom', color=col, fontsize=7, linespacing=0)

plot_legend()
common_settings(fig, ax)
plt.savefig(os.path.join(figdir, "cdk12_shapfeature_values.pdf"))
plt.savefig(os.path.join(figdir, "cdk12_shapfeature_values.png"), dpi=500, transparent=False, facecolor="w")

# ============================================================
# MMRd plot

grey="#CACACA"
face_pal = {'MMRd': grey, 'MMRp': grey}
hue_order = ['MMRd', 'MMRp']
stripplot_kwargs = {'linewidth': 0, 'size': 4, 'alpha': 0.6, 'hue_order': hue_order}

fig, ax = plt.subplots(figsize=(7.1,2.5))
# sns.boxplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable, ax=ax, fliersize=0, **boxplot_kwargs)
sns.violinplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable, ax=ax, color="lightgrey", scale="width", hue_order=hue_order, width=0.8, cut=0, bw=.3, linewidth=0, inner=None, split=False, palette=face_pal)
ax.set_alpha(0.6)

sns.stripplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable.query('(primary_label_actual == "MMRd")'), ax=ax, jitter=0.15, dodge=True, palette={'MMRd': purple, 'MMRp': "white"}, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable.query('(primary_label_actual == "DRwt")'), ax=ax, jitter=0.15, dodge=True, color=blue, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable.query('(primary_label_actual == "ATMd")'), ax=ax, jitter=0.15, dodge=True, color=orange, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable.query('(primary_label_actual == "CDK12d")'), ax=ax, jitter=0.15, dodge=True, color=green, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=mmrd_graphtable.query('(primary_label_actual == "BRCA2d")'), ax=ax, jitter=0.15, dodge=True, color=red, **stripplot_kwargs)

mmrd_feature_pvalues = make_feature_pvalues(mmrd_graphtable, mmrd_shap_features, "MMRd", "MMRp")

for i in ax.get_xticks():
	xstart = i-0.2
	xend = i+0.2
	feat=mmrd_shap_features[i]
	height=mmrd_graphtable.query('(feature == @feat)')["feature_value"].max()
	if height > 100:
		y=600
		h=100
	if height < 100:
		y=90
		h=20
	if height < 10:
		y=12
		h=2
	col='k'
	ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=col)
	if mmrd_feature_pvalues[feat] > 0.05:
		annot="n.s."
	if mmrd_feature_pvalues[feat] <= 0.05:
		annot="*"
	if mmrd_feature_pvalues[feat] <= 0.01:
		annot="**"
	if mmrd_feature_pvalues[feat] <= 0.001:
		annot="***"
	ax.text((xstart+xend)*.5, y+h-1, annot, ha='center', va='bottom', color=col, fontsize=7, linespacing=0)

plot_legend()
common_settings(fig, ax)
plt.savefig(os.path.join(figdir, "mmr_shapfeature_values.pdf"))
plt.savefig(os.path.join(figdir, "mmr_shapfeature_values.png"), dpi=500, transparent=False, facecolor="w")

#%% ==========================================================
# save stats for manuscript
# ============================================================

with open(os.path.join(figdir, "shap_feature_comparison_stats.txt"), "w") as f:
	print("#==============================================", file=f)
	print("BRCA2 statistics:", file=f)
	for feature in brca2_shap_features:
		print(f"Mann-Whitney of BRCA2 for {feature}", file=f)
		feature_table = brca_graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == "BRCA2d")')["feature_value"]
		proficient = feature_table.query('(label == "BRCA2p")')["feature_value"]
		print(f"{stats.mannwhitneyu(deficient, proficient)}", file=f)
	print("#==============================================", file=f)
	print("CDK12 statistics:", file=f)
	for feature in cdk12_shap_features:
		print(f"Mann-Whitney of CDK12 for {feature}:", file=f)
		feature_table = cdk12_graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == "CDK12d")')["feature_value"]
		proficient = feature_table.query('(label == "CDK12p")')["feature_value"]
		print(f"{stats.mannwhitneyu(deficient, proficient)}", file=f)
	print("#==============================================", file=f)
	print("MMR statistics:", file=f)
	for feature in mmrd_shap_features:
		print(f"Mann-Whitney of CDK12 for {feature}:", file=f)
		feature_table = mmrd_graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == "MMRd")')["feature_value"]
		proficient = feature_table.query('(label == "MMRp")')["feature_value"]
		print(f"{stats.mannwhitneyu(deficient, proficient)}", file=f)

#%%