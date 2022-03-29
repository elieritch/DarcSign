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
from matplotlib.ticker import Locator
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
# need to seperately handle minor ticks on sym log axis. Taken from:
# https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
# ============================================================
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

#%% ==========================================================
# some other fig settings that are same for all 4 figs.
# ============================================================
def common_settings(fig, ax):
	fig.set_size_inches(3.25, 1.1)
	ax.set_xlabel("")
	ax.tick_params(axis='y', which="major", length=2, labelsize=6, pad=1, reset=False)
	ax.tick_params(axis='x', which="major", length=2, labelsize=7, pad=0, reset=False)
	sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
	ax.yaxis.set_label_coords(-0.08, 0.5)
	ax.set_xticklabels(["DRwt", "ATMd", "CDK12d", "BRCA2d", "MMRd", "Bladder"])
	fig.subplots_adjust(left=0.11, right=0.995, top=0.91, bottom=0.1)
	return fig, ax

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#file from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "fig1")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
# sample_labels = sample_labels[sample_labels['manual check for usefullness (0=Fail)'] != 0]
df = pd.merge(sample_labels, sigs, how='left', on='sample')
# df.loc[df["cancer"] == "BC", 'primary_label'] = "Bladder"
#%% ==========================================================
# some calculation/manipulations needed for the graphs
# ============================================================

def make_df_for_graph(original_df):
	df_mut_rate = original_df.copy(deep=True)
	df_mut_rate.loc[df_mut_rate["cancer"] == "BC", 'label'] = "Bladder"
	df_mut_rate["total_snv"] = df_mut_rate[snv_categories[1:]].sum(axis=1)
	df_mut_rate["snv_rate"] = df_mut_rate["total_snv"]/45 #45mb is size of medexome panel

	df_mut_rate["total_indel"] = df_mut_rate[indel_categories[1:]].sum(axis=1)
	df_mut_rate["indel_rate"] = df_mut_rate["total_indel"]/45 #45mb is size of medexome panel

	cpf_list = ["CopyFraction_0", "CopyFraction_1", "CopyFraction_2", "CopyFraction_3", "CopyFraction_4", "CopyFraction_5", "CopyFraction_6"]
	df_mut_rate['max_cn_fraction'] = df_mut_rate[cpf_list].max(axis=1)
	df_mut_rate['not_max_cn_fraction'] = 1 - df_mut_rate['max_cn_fraction']

	# This is just to order the df for graphing and using seperate dfs is good for calculating stats later.
	df_mut_rate = df_mut_rate.drop(columns=snv_categories[1:]).drop(columns=indel_categories[1:]).drop(columns=cnv_categories[1:])
	brca2_mut_rate = df_mut_rate.query('(label == "BRCA2d")')
	cdk12_mut_rate = df_mut_rate.query('(label == "CDK12d")')
	mmrd_mut_rate = df_mut_rate.query('(label == "MMRd")')
	atm_mut_rate = df_mut_rate.query('(label == "ATMd")')
	drp_mut_rate = df_mut_rate.query('(label == "DRwt")')
	bladder_mut_rate = df_mut_rate.query('(label == "Bladder")')
	df_for_graph = pd.concat([drp_mut_rate, atm_mut_rate, cdk12_mut_rate, brca2_mut_rate, mmrd_mut_rate, bladder_mut_rate]).reset_index(drop=True)
	return df_for_graph

graphdata = make_df_for_graph(df)

#%% ==========================================================
# setup graph values
# ============================================================
#%% values for annot lines for snv and indel
h = 0.015 #amount of y for line nub
text_pos = 0.013
space = 0.09 #total space between lines
bladder_line = 0.99
top_line = bladder_line #height of top line 
mmrd_line = top_line-(space*1)
brca2_line = top_line-(space*2)
cdk12_line = top_line-(space*3)
atm_line = top_line-(space*4)

# bladder_line
mmrd_line = top_line-(space*1)
brca2_line = top_line-(space*2)
cdk12_line = top_line-(space*3)
atm_line = top_line-(space*4)

psize = 5 #font size for pvalues
fliersize = 1.9
swarmsize = 1.8
total_values = 6
position = 1/(total_values*2)
label_fs = 7
ytickfs = 6
ticklength = 2
lineweight=0.7

#%% ==========================================================
# SNV comparisons
# ============================================================
ax = sns.boxplot(x="label", y="snv_rate", data=graphdata, linewidth=0.7, fliersize=fliersize)
ax = sns.swarmplot(x="label", y="snv_rate", data=graphdata, color=".3", size=swarmsize)
ax.set_yscale('symlog', linthresh=1, base=10)
xtickslocs = ax.get_xticks()

yaxis = plt.gca().yaxis
yaxis.set_minor_locator(MinorSymLogLocator(10e0))
ax.set_ylim(0, 300)
ax.set_ylabel("SNVs/Mb", labelpad=1, fontsize = label_fs)
common_settings(plt.gcf(), ax)

y=atm_line
xstart = position
xend = 3*position
a = graphdata.query('(label == "DRwt")')["snv_rate"]
b = graphdata.query('(label == "ATMd")')["snv_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=cdk12_line
xstart = position
xend = 5*position
b = graphdata.query('(label == "CDK12d")')["snv_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=brca2_line
xstart = position
xend = 7*position
b = graphdata.query('(label == "BRCA2d")')["snv_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=mmrd_line
xstart = position
xend = 9*position
b = graphdata.query('(label == "MMRd")')["snv_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=bladder_line
xstart = position
xend = 11*position
b = graphdata.query('(label == "Bladder")')["snv_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

plt.savefig(os.path.join(figdir, "snv_rate_comparison.png"), dpi=500)
plt.savefig(os.path.join(figdir, "snv_rate_comparison.pdf"))
plt.close()

#%% ==========================================================
# Indel comparisons
# ============================================================
ax = sns.boxplot(x="label", y="indel_rate", data=graphdata, linewidth=0.7, fliersize=fliersize)
ax = sns.swarmplot(x="label", y="indel_rate", data=graphdata, color=".3", size=swarmsize)
ax.set_yscale('symlog', linthresh=1, base=10)
xtickslocs = ax.get_xticks()

yaxis = plt.gca().yaxis
yaxis.set_minor_locator(MinorSymLogLocator(10e0))
ax.set_ylim(0,60)
ax.set_ylabel("INDELs/Mb", labelpad=1, fontsize = label_fs)
common_settings(plt.gcf(), ax)

y=atm_line
xstart = position
xend = 3*position
a = graphdata.query('(label == "DRwt")')["indel_rate"]
b = graphdata.query('(label == "ATMd")')["indel_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=cdk12_line
xstart = position
xend = 5*position
b = graphdata.query('(label == "CDK12d")')["indel_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=brca2_line
xstart = position
xend = 7*position
b = graphdata.query('(label == "BRCA2d")')["indel_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=mmrd_line
xstart = position
xend = 9*position
b = graphdata.query('(label == "MMRd")')["indel_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=bladder_line
xstart = position
xend = 11*position
b = graphdata.query('(label == "Bladder")')["indel_rate"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

plt.savefig(os.path.join(figdir, "indel_rate_comparison.png"), dpi=500)
plt.savefig(os.path.join(figdir, "indel_rate_comparison.pdf"))
plt.close()

#%% ==========================================================
# Ploidy comparisons
# ============================================================
ax = sns.boxplot(x="label", y="ploidy_estimate_sequenza", data=graphdata, linewidth=0.7, fliersize=fliersize)
ax = sns.swarmplot(x="label", y="ploidy_estimate_sequenza", data=graphdata, color=".2", size=swarmsize)

ax.set_ylim(1.0, 9)
ax.set_yticks([x.round(2) for x in np.arange(1.0, 8.1, 1.0)])
ax.set_yticklabels([x.round(2) for x in np.arange(1.0, 8.1, 1.0)])

ax.set_ylabel("Estimated Ploidy", labelpad=1, fontsize = label_fs)
common_settings(plt.gcf(), ax)

y=atm_line
xstart = position
xend = 3*position
a = graphdata.query('(label == "DRwt")')["ploidy_estimate_sequenza"]
b = graphdata.query('(label == "ATMd")')["ploidy_estimate_sequenza"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=cdk12_line
xstart = position
xend = 5*position
b = graphdata.query('(label == "CDK12d")')["ploidy_estimate_sequenza"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=brca2_line
xstart = position
xend = 7*position
b = graphdata.query('(label == "BRCA2d")')["ploidy_estimate_sequenza"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=mmrd_line
xstart = position
xend = 9*position
b = graphdata.query('(label == "MMRd")')["ploidy_estimate_sequenza"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=bladder_line
xstart = position
xend = 11*position
b = graphdata.query('(label == "Bladder")')["ploidy_estimate_sequenza"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

plt.savefig(os.path.join(figdir, "ploidy_estimate_comparison.png"), dpi=500)
plt.savefig(os.path.join(figdir, "ploidy_estimate_comparison.pdf"))
plt.close()

#%% ==========================================================
# Allelic imbalance comparisons
# ============================================================
ax = sns.boxplot(x="label", y="not_max_cn_fraction", data=graphdata, linewidth=0.7, fliersize=fliersize)
ax = sns.swarmplot(x="label", y="not_max_cn_fraction", data=graphdata, color=".3", size=swarmsize)

ax.set_ylim(0, 1.2)
ax.set_yticks([x.round(2) for x in np.arange(0.0, 1.1, 0.2)])
ax.set_yticklabels([x.round(2) for x in np.arange(0.0, 1.1, 0.2)])
ax.set_ylabel("Fraction with CNV", labelpad=1, fontsize = label_fs)
common_settings(plt.gcf(), ax)

y=atm_line
xstart = position
xend = 3*position
a = graphdata.query('(label == "DRwt")')["not_max_cn_fraction"]
b = graphdata.query('(label == "ATMd")')["not_max_cn_fraction"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=cdk12_line
xstart = position
xend = 5*position
b = graphdata.query('(label == "CDK12d")')["not_max_cn_fraction"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=brca2_line
xstart = position
xend = 7*position
b = graphdata.query('(label == "BRCA2d")')["not_max_cn_fraction"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=mmrd_line
xstart = position
xend = 9*position
b = graphdata.query('(label == "MMRd")')["not_max_cn_fraction"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

y=bladder_line
xstart = position
xend = 11*position
b = graphdata.query('(label == "Bladder")')["not_max_cn_fraction"]
u, p = stats.mannwhitneyu(a,b)
annot = "{:.1e}".format(p)
ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=lineweight, c=".3", transform=ax.transAxes, clip_on = False)
ax.text((xstart+xend)*.5, y+h+text_pos, f"p = {annot}", ha='center', va='baseline', color="k", fontsize=psize, linespacing=0, transform=ax.transAxes)

plt.savefig(os.path.join(figdir, "allelic_imbalance_comparison.png"), dpi=500)
plt.savefig(os.path.join(figdir, "allelic_imbalance_comparison.pdf"))
plt.close()

