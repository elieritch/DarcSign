# -*- coding: utf-8 -*-
# @author: Elie
# run on laptop vscode under env py37_xgboost_ml (python 3.7.9)
#%%
# Libraries
import pandas as pd
import os
import numpy as np
import datetime
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

import umap
# import math
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
# make concat sig dataframe  and umap functions. 
# differnt aesthetics for big / small plot
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

def run_umaps(df, x, y, neighbor, md, rs, output, legendtitle):
	mpl.rcParams['savefig.transparent'] = "False"
	mpl.rcParams['axes.facecolor'] = "white"
	mpl.rcParams['figure.facecolor'] = "white"
	fs = 6 #fontsize

	standard_embedding = umap.UMAP(random_state=rs, n_neighbors=neighbor, metric="euclidean", n_epochs=500, min_dist=md, n_components=2).fit_transform(x)
	mu = pd.DataFrame(data = standard_embedding , columns = ['umap X', 'umap Y'])
	mu["sample"] = y
	mu = pd.merge(mu, df, how='left', on="sample").drop_duplicates(subset=['sample']).reset_index(drop=True)
	mu["cellularity_sequenza"] = mu["cellularity_sequenza"]*175

	bcscat = mu.query('(label == "BRCA2d")')
	cdk12scat = mu.query('(label == "CDK12d")')
	mmrdscat = mu.query('(label == "MMRd")')
	ddrpscat = mu.query('(label == "DRwt")')

	color_list = list(sns.color_palette().as_hex())
	blue = color_list[0] #drp
	orange = color_list[1] #atm
	green = color_list[2] #cdk12
	red = color_list[3] #brca2
	purple = color_list[4] #mmr
	fig, ax = plt.subplots(figsize=(2.75,1.5))
	ax.scatter(x=ddrpscat["umap X"], y=ddrpscat["umap Y"], color=blue, s=ddrpscat["cellularity_sequenza"], alpha=0.6, label="DRp", zorder=100, linewidth=0)
	ax.scatter(x=bcscat["umap X"], y=bcscat["umap Y"], color=red, s=bcscat["cellularity_sequenza"], alpha=0.6, label="BRCA2d", zorder=100, linewidth=0)
	ax.scatter(x=cdk12scat["umap X"], y=cdk12scat["umap Y"], color=green, s=cdk12scat["cellularity_sequenza"], alpha=0.6, label="CDK12d", zorder=100, linewidth=0)
	ax.scatter(x=mmrdscat["umap X"], y=mmrdscat["umap Y"], color=purple, s=mmrdscat["cellularity_sequenza"], alpha=0.6, label="MMRd", zorder=100, linewidth=0)
	ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
	ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
	ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
	ax.set_ylabel(f"Umap Y {legendtitle}", fontsize=fs, labelpad=4, verticalalignment='center')
	ax.yaxis.set_label_coords(-0.10, 0.5)
	ax.set_xlabel(f"Umap X {legendtitle}", fontsize=fs, labelpad=3, verticalalignment='center')
	sns.despine(ax=ax, top=True, right=True)

	fig.subplots_adjust(left=0.11, right=0.999, top=0.97, bottom=0.16)
	plt.savefig(output,dpi=500)
	# plt.close()

def run_umap_bigger_graph(df, x, y, neighbor, md, rs, output, legendtitle):
	mpl.rcParams['savefig.transparent'] = "False"
	mpl.rcParams['axes.facecolor'] = "white"
	mpl.rcParams['figure.facecolor'] = "white"
	fs = 6 #fontsize

	standard_embedding = umap.UMAP(random_state=rs, n_neighbors=neighbor, metric="euclidean", n_epochs=500, min_dist=md, n_components=2).fit_transform(x)
	mu = pd.DataFrame(data = standard_embedding , columns = ['umap X', 'umap Y'])
	mu["sample"] = y
	mu = pd.merge(mu, df, how='left', on="sample").drop_duplicates(subset=['sample']).reset_index(drop=True)
	mu["cellularity_sequenza"] = mu["cellularity_sequenza"]*300

	bcscat = mu.query('(label == "BRCA2d")')
	cdk12scat = mu.query('(label == "CDK12d")')
	mmrdscat = mu.query('(label == "MMRd")')
	ddrpscat = mu.query('(label == "DRwt")')
	color_list = list(sns.color_palette().as_hex())
	blue = color_list[0] #drp
	orange = color_list[1] #atm
	green = color_list[2] #cdk12
	red = color_list[3] #brca2
	purple = color_list[4] #mmr
	fig, ax = plt.subplots(figsize=(4,3.5))
	ax.scatter(x=ddrpscat["umap X"], y=ddrpscat["umap Y"], color=blue, s=ddrpscat["cellularity_sequenza"], alpha=0.6, label="DRp", zorder=100, linewidth=0)
	ax.scatter(x=bcscat["umap X"], y=bcscat["umap Y"], color=red, s=bcscat["cellularity_sequenza"], alpha=0.6, label="BRCA2d", zorder=100, linewidth=0)
	ax.scatter(x=cdk12scat["umap X"], y=cdk12scat["umap Y"], color=green, s=cdk12scat["cellularity_sequenza"], alpha=0.6, label="CDK12d", zorder=100, linewidth=0)
	ax.scatter(x=mmrdscat["umap X"], y=mmrdscat["umap Y"], color=purple, s=mmrdscat["cellularity_sequenza"], alpha=0.6, label="MMRd", zorder=100, linewidth=0)
	ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
	ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
	ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
	ax.set_ylabel(f"Umap Y {legendtitle}", fontsize=fs, labelpad=4, verticalalignment='center')
	ax.yaxis.set_label_coords(-0.07, 0.5)
	ax.set_xlabel(f"Umap X {legendtitle}", fontsize=fs, labelpad=4, verticalalignment='center')
	sns.despine(ax=ax, top=True, right=True)

	fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.07)
	plt.savefig(output,dpi=500)
	# plt.close()

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================
#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "fig4")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)

df_pc = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)

#%% ==========================================================
# run umap plots
# ============================================================
print(f"snvs starting at {datetime.datetime.now()}")
features = df_pc[snv_categories[1:]].columns
x = df_pc.loc[:, features].values
y = df_pc.loc[:,['sample']].values
neighbor=16
md=0.4
rs=2
output = os.path.join(figdir, f"umap_snv_{neighbor}neighbor_{md}mindist_{rs}rando.png")
run_umaps(df_pc, x, y, neighbor, md, rs, output, "SNV Features")

print(f"indel starting at {datetime.datetime.now()}")
features = df_pc[indel_categories[1:]].columns
x = df_pc.loc[:, features].values
y = df_pc.loc[:,['sample']].values
neighbor=17
md=0.4
rs=9
output = os.path.join(figdir, f"umap_indel_{neighbor}neighbor_{md}mindist_{rs}rando.png")
run_umaps(df_pc, x, y, neighbor, md, rs, output, "InDel Features")

print(f"cnv starting at {datetime.datetime.now()}")
features = df_pc[cnv_categories[1:]].columns
x = df_pc.loc[:, features].values
y = df_pc.loc[:,['sample']].values
neighbor=10
md=0.3
rs=4
output = os.path.join(figdir, f"umap_cnv_{neighbor}neighbor_{md}mindist_{rs}rando.png")
run_umaps(df_pc, x, y, neighbor, md, rs, output, "Segment Features")

print(f"comb starting at {datetime.datetime.now()}")
all_categories = snv_categories[1:] + indel_categories[1:] + cnv_categories[1:]
features = df_pc[all_categories].columns
x = df_pc.loc[:, features].values
y = df_pc.loc[:,['sample']].values
neighbor=18
md=0.35
rs=1
output = os.path.join(figdir, f"umap_comb_{neighbor}neighbor_{md}mindist_{rs}rando.png")
run_umap_bigger_graph(df_pc, x, y, neighbor, md, rs, output, "Combined Features")

#%% legend
color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drp
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr

max_ccf = df_pc["cellularity_sequenza"].max()*200
min_ccf = df_pc["cellularity_sequenza"].min()*200
half_ccf = (max_ccf - min_ccf)/2
# for x in [15, 80, 150]:

fig, ax = plt.subplots(figsize=(1.5,0.3))
ax.axis('off')
plt.grid(b=False, which='both')
plt.scatter([], [], c=blue, alpha=0.6, s=min_ccf*10, label="DRwt", marker='o', linewidth=0)
plt.scatter([], [], c=red, alpha=0.6, s=min_ccf*10, label="BRCA2d", marker='o', linewidth=0)
plt.scatter([], [], c=green, alpha=0.6, s=min_ccf*10, label="CDK12d", marker='o', linewidth=0)
plt.scatter([], [], c=purple, alpha=0.6, s=min_ccf*10, label="MMRd", marker='o', linewidth=0)
plt.legend(loc='center', ncol=2, borderaxespad=0., fontsize=6, labelspacing=0.4, handletextpad=0, borderpad=0.3, handlelength=2, framealpha=0, markerscale=0.6)
output = os.path.join(figdir, "umap_gene_legend.png")
plt.savefig(output,dpi=500, transparent=True)



fig, ax = plt.subplots(figsize=(1.5,0.3))
ax.axis('off')
ax.grid(b=False, which='both')
ax.scatter(x=0.02, y=0.5, alpha=0.8, s=min_ccf, marker='o', linewidth=0, c="#909090")
ax.arrow(x=0.055, y=0.5, dx=0.1, dy=0, length_includes_head=True, head_length=0.04, head_width=half_ccf*1.5, width=min_ccf*2, linewidth=0, alpha=0.8, color="#858585")
ax.scatter(x=0.2, y=0.5, alpha=0.8, s=half_ccf, marker='o', linewidth=0, c="#808080")
ax.arrow(x=0.255, y=0.5, dx=0.09, dy=0, length_includes_head=True, head_length=0.04, head_width=max_ccf, width=half_ccf, linewidth=0, alpha=0.8, color="#757575")
ax.scatter(x=0.41, y=0.5, alpha=0.8, s=max_ccf, marker='o', linewidth=0, c="#707070")

ax.text(x=0.5, y=0.5, s=r'0.1$\rightarrow$1.0'+'\ncellularity', ha='left', va='center', ma='center', color="k", fontsize=6, linespacing=1)
ax.set_xlim(-0.05, 0.7)
output = os.path.join(figdir, "umap_size_legend.png")
plt.savefig(output,dpi=500, transparent=True)

#%%
