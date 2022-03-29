# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================

import pandas as pd
import numpy as np
import os
pd.options.mode.chained_assignment = None #Pandas warnings off
#plotting
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
# stats
from scipy import stats

#set matplotlib rcparams
mpl.rcParams['savefig.transparent'] = "False"
mpl.rcParams['axes.facecolor'] = "white"
mpl.rcParams['figure.facecolor'] = "white"
mpl.rcParams['font.size'] = "5"
plt.rcParams['savefig.transparent'] = "False"
plt.rcParams['axes.facecolor'] = "white"
plt.rcParams['figure.facecolor'] = "white"
plt.rcParams['font.size'] = "5"

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
# get paths, load data and make df with each file merged
# ============================================================

#file from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig1")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False).query('(cancer != "BC")').reset_index(drop=True)
df = pd.merge(sample_labels, sigs, how='left', on='sample')

#%% ==========================================================
# some calculation/manipulations needed for the graphs
# ============================================================

df_mut_rate = df.copy(deep=True)
df_mut_rate["total_snv"] = df_mut_rate[snv_categories[1:]].sum(axis=1)
df_mut_rate["snv_rate"] = df_mut_rate["total_snv"]/45

df_mut_rate["total_indel"] = df_mut_rate[indel_categories[1:]].sum(axis=1)
df_mut_rate["indel_rate"] = df_mut_rate["total_indel"]/45

cpf_list = ["CopyFraction_0", "CopyFraction_1", "CopyFraction_2", "CopyFraction_3", "CopyFraction_4", "CopyFraction_5", "CopyFraction_6"]
# df_ai = df_ai.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:])
df_mut_rate['max_cn_fraction'] = df_mut_rate[cpf_list].max(axis=1)
df_mut_rate['not_max_cn_fraction'] = 1 - df_mut_rate['max_cn_fraction']

copy_fraction_table = df_mut_rate[cpf_list]
copy_fraction_table['max_value'] = copy_fraction_table.apply(lambda x: copy_fraction_table.columns[x.argmax()], axis = 1)
copy_fraction_table['max_value_int'] = copy_fraction_table['max_value'].replace("CopyFraction_", "", regex=True).astype(int)
copy_fraction_table["lessthanmax"] = 0
copy_fraction_table["greaterthanmax"] = 0
for i, r in copy_fraction_table.iterrows():
	maxvalue = r['max_value_int']
	columns_less_than_max = copy_fraction_table.columns[:maxvalue]
	lessthanmax = r[columns_less_than_max].sum()
	copy_fraction_table.loc[i,'lessthanmax'] = lessthanmax
	columns_greater_than_max = copy_fraction_table.columns[maxvalue+1:7]
	greaterthanmax = r[columns_greater_than_max].sum()
	copy_fraction_table.loc[i,'greaterthanmax'] = greaterthanmax
# 	print(columns_greater_than_max, greaterthanmax)
copy_fraction_table["label"] = df_mut_rate["label"]
copy_fraction_table.loc[copy_fraction_table['max_value_int'] > 2, 'wgd'] = "More Than 2 genome equivalents"
copy_fraction_table.loc[copy_fraction_table['max_value_int'] <= 2, 'wgd'] = "2 genome equivalents"

df_mut_rate = df_mut_rate.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
brca2_mut_rate = df_mut_rate.query('(label == "BRCA2d")')
cdk12_mut_rate = df_mut_rate.query('(label == "CDK12d")')
mmrd_mut_rate = df_mut_rate.query('(label == "MMRd")')
atm_mut_rate = df_mut_rate.query('(label == "ATMd")')
drp_mut_rate = df_mut_rate.query('(label == "DRwt")')
all_class = pd.concat([drp_mut_rate, atm_mut_rate, cdk12_mut_rate, brca2_mut_rate, mmrd_mut_rate]).reset_index(drop=True)


brca2_cop_rate = copy_fraction_table.query('(label == "BRCA2d")')
cdk12_cop_rate = copy_fraction_table.query('(label == "CDK12d")')
mmrd_cop_rate = copy_fraction_table.query('(label == "MMRd")')
atm_cop_rate = copy_fraction_table.query('(label == "ATMd")')
drp_cop_rate = copy_fraction_table.query('(label == "DRwt")')
all_class_copy = pd.concat([drp_cop_rate, atm_cop_rate, cdk12_cop_rate, brca2_cop_rate, mmrd_cop_rate]).reset_index(drop=True)

#%% ==========================================================
# Graphing
# ============================================================

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6.5, 4.1), sharex=True, gridspec_kw={"height_ratios":[1,1]})
sns.boxplot(x="label", y="lessthanmax", hue="wgd", data=all_class_copy, ax=ax[0], hue_order=["2 genome equivalents", "More Than 2 genome equivalents"])
sns.swarmplot(x="label", y="lessthanmax", hue="wgd", data=all_class_copy, ax=ax[0], hue_order=["2 genome equivalents", "More Than 2 genome equivalents"], size=3, edgecolor='black', linewidth=0.3)
# sns.swarmplot(x="label", y="snv_rate", data=all_class, color=".3", size=2)
sns.boxplot(x="label", y="greaterthanmax", hue="wgd", data=all_class_copy, ax=ax[1], hue_order=["2 genome equivalents", "More Than 2 genome equivalents"])
sns.swarmplot(x="label", y="greaterthanmax", hue="wgd", data=all_class_copy, ax=ax[1], hue_order=["2 genome equivalents", "More Than 2 genome equivalents"], size=3, edgecolor='black', linewidth=0.3)
for i in range(2):
	ax[i].set_xlabel("")
	ax[i].tick_params(axis='both', which="major", length=3, labelsize=8, pad=1, reset=False)
	sns.despine(ax=ax[i], top=True, right=True, left=False, bottom=False)
	ax[i].legend_.remove()
# 	ax.yaxis.set_label_coords(-0.08, 0.5)
# fig.subplots_adjust(left=0.11, right=0.995, top=0.93, bottom=0.09)
fig.subplots_adjust(hspace=0.3, bottom=0.06, left=0.1, right=0.95, top=0.925)
ax[0].set_ylabel("Copy Loss\n(genome fraction)", fontsize=8)
ax[1].set_ylabel("Copy Gain\n(genome fraction)", fontsize=8)

color_list = list(sns.color_palette().as_hex())
handles=[]
handles.append(mlines.Line2D([], [], color=color_list[0], markeredgecolor='black', markeredgewidth=0.4, marker='o', lw=0, markersize=6, label='2 genome equivalents', linestyle=None))
handles.append(mlines.Line2D([], [], color=color_list[1], markeredgecolor='black', markeredgewidth=0.4, marker='o', lw=0, markersize=6, label='More than 2 genome equivalents', linestyle=None))
# handles.append(Patch(facecolor='#3F60AC', edgecolor='none', label='Homozygous deletion'))
legend = ax[0].legend(handles=handles,loc='center', fontsize=8, bbox_to_anchor=(0.7,1.115), facecolor='white', ncol=2, handletextpad=-0.4, columnspacing=0.5, edgecolor='0.8', fancybox=True, frameon=True)

drp_two = drp_cop_rate.query('(max_value_int <= 2)')["lessthanmax"]
drp_gtt = drp_cop_rate.query('(max_value_int > 2)')["lessthanmax"]
u, p = stats.mannwhitneyu(drp_two, drp_gtt)
annot = "{:.1e}".format(p)
xstart = -0.2
xend = 0.2
y=0.65
h=0.02
ax[0].plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=".3", clip_on = False)
ax[0].text((xstart+xend)*.5, y+h+0.02, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=6, linespacing=0)

brc_two = brca2_cop_rate.query('(max_value_int <= 2)')["lessthanmax"]
brc_gtt = brca2_cop_rate.query('(max_value_int > 2)')["lessthanmax"]
u, p = stats.mannwhitneyu(brc_two, brc_gtt)
annot = "{:.1e}".format(p)
xstart = 2.8
xend = 3.2
y=0.65
h=0.02
ax[0].plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=".3", clip_on = False)
ax[0].text((xstart+xend)*.5, y+h+0.02, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=6, linespacing=0)


drp_two = drp_cop_rate.query('(max_value_int <= 2)')["greaterthanmax"]
drp_gtt = drp_cop_rate.query('(max_value_int > 2)')["greaterthanmax"]
u, p = stats.mannwhitneyu(drp_two, drp_gtt)
annot = "{:.1e}".format(p)
xstart = -0.2
xend = 0.2
y=0.65
h=0.02
ax[1].plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=".3", clip_on = False)
ax[1].text((xstart+xend)*.5, y+h+0.02, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=6, linespacing=0)

brc_two = brca2_cop_rate.query('(max_value_int <= 2)')["greaterthanmax"]
brc_gtt = brca2_cop_rate.query('(max_value_int > 2)')["greaterthanmax"]
u, p = stats.mannwhitneyu(brc_two, brc_gtt)
annot = "{:.1e}".format(p)
xstart = 2.8
xend = 3.2
y=0.65
h=0.02
ax[1].plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=".3", clip_on = False)
ax[1].text((xstart+xend)*.5, y+h+0.02, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=6, linespacing=0)

plt.savefig(os.path.join(figdir, "cn_incontextof_wgd.png"), dpi=500)
plt.savefig(os.path.join(figdir, "cn_incontextof_wgd.pdf"))

#%%