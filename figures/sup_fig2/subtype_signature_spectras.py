# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================

import pandas as pd
import os
pd.options.mode.chained_assignment = None #Pandas warnings off
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
# import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

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
# plot average naive tns profile \
# (axis differernt then main figure)
# ============================================================

def SNV_naive_signature(df_sig_count):
	# df_sig_count = df_cdk12
	df = df_sig_count[snv_categories].drop(columns=["sample"])
	total_mutations = df.values.sum()
	sum_of_each_column = df.sum(axis=0)
	all_normal = sum_of_each_column/total_mutations
	normalized_sigs = all_normal.copy(deep=True).reset_index().rename(columns={"index":"base", 0:"amount"})
	c2a = normalized_sigs[0:16]
	c2g = normalized_sigs[16:32]
	c2t = normalized_sigs[32:48]
	t2a = normalized_sigs[48:64]
	t2c = normalized_sigs[64:80]
	t2g = normalized_sigs[80:96]

	sns.set()
	sns.set_style(style="whitegrid")
	fig, ax = plt.subplots(nrows=1, ncols=6, sharey='all', figsize=(4.4, 1.0))
	plt.subplots_adjust(wspace=0.0)
	sns.barplot(x=c2a["base"], y=c2a["amount"], ax=ax[0], color="#3498db", edgecolor=None)#niagarablue#578CA9
	sns.barplot(x=c2g["base"], y=c2g["amount"], ax=ax[1], color="#2ecc71", edgecolor=None)#greenery#92B558
	sns.barplot(x=c2t["base"], y=c2t["amount"], ax=ax[2], color="#e74c3c", edgecolor=None)#aurorared#B93A32
	sns.barplot(x=t2a["base"], y=t2a["amount"], ax=ax[3], color="#95a5a6", edgecolor=None)#harbormist
	sns.barplot(x=t2c["base"], y=t2c["amount"], ax=ax[4], color="#FFAE42", edgecolor=None)#greenery
	sns.barplot(x=t2g["base"], y=t2g["amount"], ax=ax[5], color="#9b59b6", edgecolor=None)#Yellow-Orange

	#fasr left axis
	ax[0].set_ylabel("Proportion", fontsize=7, horizontalalignment="center", labelpad=0.5)
	# ax[0].set_yticks([0.00,0.05,0.10])
	ax[0].tick_params(axis='y', which='major', labelsize=5, pad=-5)
	ax[0].set_ylim(0,0.19)
# 	yticks = np.around(ax[0].get_yticks(), decimals=2)
	ax[0].set_yticks([0.05,0.10,0.15])
# 	yrange = ax[0].get_ylim()
# 	ax[0].set_ylim(yrange)
# 	ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

	for i in range(1,6):
		sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=False)
		ax[i].tick_params(axis='y', which='major', length=0)
		ax[i].set_ylabel("")
	ax[0].set_ylim(0,0.19)
# 	yticks = np.around(ax[0].get_yticks(), decimals=2)
	ax[0].set_yticks([0.05,0.10,0.15])
	#loop affects all axis
	for i, j in enumerate(ax):
		ax[i].tick_params(axis='x', which='both', length=-0.2)
		ax[i].set_xlabel("")
		ax[i].grid(which='both', axis='y', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-1)
		ax[i].tick_params(axis='x', which='major', labelsize=3.2, pad=1, labelrotation=90)
		ax[i].set_xticklabels(ax[i].get_xticklabels(), fontname="DejaVu Sans Mono")

	# top_of_graph = normalized_sigs["amount"].max() + 0.01
	top_of_graph = normalized_sigs["amount"].max()
	top_of_graph = 0.18
	top_of_graph_plusspace = top_of_graph+(0.1*top_of_graph)
	top_of_graph_plusplusspace = top_of_graph+(0.1*top_of_graph)+(0.05*top_of_graph)
	
	#need to change height to a proportion of graph, this is going to screw you over!!!!
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#3498db", edgecolor=None, clip_on=False)
	ax[0].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#2ecc71", edgecolor=None, clip_on=False)
	ax[1].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#e74c3c", edgecolor=None, clip_on=False)
	ax[2].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#95a5a6", edgecolor=None, clip_on=False)
	ax[3].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#FFAE42", edgecolor=None, clip_on=False)
	ax[4].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=16, height=0.035, fill=True, facecolor="#9b59b6", edgecolor=None, clip_on=False)
	ax[5].add_patch(rect)

	# ax[0].annotate("C>A", xy=(0.5,0.9), xycoords='figure fraction', fontsize=6)
	ax[0].annotate("C>A", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)
	ax[1].annotate("C>G", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)
	ax[2].annotate("C>T", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)
	ax[3].annotate("T>A", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)
	ax[4].annotate("T>C", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)
	ax[5].annotate("T>G", (7.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="center", annotation_clip=False)

	# yticks = np.around(ax[2].get_yticks(), decimals=2)
	# yrange = ax[5].get_ylim()
	# for i, j in enumerate(ax):
	# # for i in range(0,6):
	# 	ax[i].set_ylim(yrange)
	# 	ax[i].set_yticks(yticks)

	plt.subplots_adjust(wspace=0.0, left=0.055, right=0.998, bottom=0.22, top=0.83)
	return fig, ax

#%% plot average naive indel signature for cohort or subset of cohort
# =============================================================================

def INDEL_naive_signature(df_sig_count):
	df = df_sig_count[indel_categories].drop(columns=["sample"])
	total_mutations = df.values.sum()
	sum_of_each_column = df.sum(axis=0)
	all_normal = sum_of_each_column/total_mutations
	normalized_sigs = all_normal.copy(deep=True).reset_index().rename(columns={"index":"base", 0:"amount"})

	one_del_c = normalized_sigs[0:6]
	one_del_t = normalized_sigs[6:12]
	xcat = ['1', '2', '3', '4', '5', '6+']
	one_del_c["xcat"] = xcat
	one_del_t["xcat"] = xcat

	one_ins_c = normalized_sigs[12:18]
	one_ins_t = normalized_sigs[18:24]
	xcat = ['0', '1', '2', '3', '4', '5+']
	one_ins_c["xcat"] = xcat
	one_ins_t["xcat"] = xcat

	two_del_R = normalized_sigs[24:30]
	three_del_R = normalized_sigs[30:36]
	four_del_R = normalized_sigs[36:42]
	five_del_R = normalized_sigs[42:48]
	xcat = ['1', '2', '3', '4', '5', '6+']
	two_del_R["xcat"] = xcat
	three_del_R["xcat"] = xcat
	four_del_R["xcat"] = xcat
	five_del_R["xcat"] = xcat

	two_ins_R = normalized_sigs[48:54]
	three_ins_R = normalized_sigs[54:60]
	four_ins_R = normalized_sigs[60:66]
	five_ins_R = normalized_sigs[66:72]
	xcat = ['0', '1', '2', '3', '4', '5+']
	two_ins_R["xcat"] = xcat
	three_ins_R["xcat"] = xcat
	four_ins_R["xcat"] = xcat
	five_ins_R["xcat"] = xcat

	microhomo2 = normalized_sigs[72:73]
	xcat = ['1']
	microhomo2["xcat"] = xcat

	microhomo3 = normalized_sigs[73:75]
	xcat = ['1', '2']
	microhomo3["xcat"] = xcat

	microhomo4 = normalized_sigs[75:78]
	xcat = ['1', '2', '3']
	microhomo4["xcat"] = xcat

	microhomo5 = normalized_sigs[78:83]
	xcat = ['1', '2', '3', '4', '5+']
	microhomo5["xcat"] = xcat

	sns.set()
	sns.set_style(style="whitegrid")
	fig, ax = plt.subplots(nrows=1, ncols=16, sharey='all', figsize=(4.4, 1.0), gridspec_kw={'width_ratios':[6,6,6,6,6,6,6,6,6,6,6,6,1,2,3,5]})
	#plt.subplots_adjust(wspace=0.0)

	sns.barplot(x=one_del_c["xcat"], y=one_del_c["amount"], ax=ax[0], color="#fdae6b", edgecolor=None)#orange
	sns.barplot(x=one_del_t["xcat"], y=one_del_t["amount"], ax=ax[1], color="#e6550d", edgecolor=None)#orange

	sns.barplot(x=one_ins_c["xcat"], y=one_ins_c["amount"], ax=ax[2], color="#a1d99b", edgecolor=None)#greenery#92B558
	sns.barplot(x=one_ins_t["xcat"], y=one_ins_t["amount"], ax=ax[3], color="#31a354", edgecolor=None)#greenery#92B558

	sns.barplot(x=two_del_R["xcat"], y=two_del_R["amount"], ax=ax[4], color="#fee5d9", edgecolor=None)#red
	sns.barplot(x=three_del_R["xcat"], y=three_del_R["amount"], ax=ax[5], color="#fcae91", edgecolor=None)#red
	sns.barplot(x=four_del_R["xcat"], y=four_del_R["amount"], ax=ax[6], color="#fb6a4a", edgecolor=None)#red
	sns.barplot(x=five_del_R["xcat"], y=five_del_R["amount"], ax=ax[7], color="#cb181d", edgecolor=None)#red

	sns.barplot(x=two_ins_R["xcat"], y=two_ins_R["amount"], ax=ax[8], color="#bdd7e7", edgecolor=None)#red
	sns.barplot(x=three_ins_R["xcat"], y=three_ins_R["amount"], ax=ax[9], color="#6baed6", edgecolor=None)#red
	sns.barplot(x=four_del_R["xcat"], y=four_del_R["amount"], ax=ax[10], color="#3182bd", edgecolor=None)#red
	sns.barplot(x=five_ins_R["xcat"], y=five_ins_R["amount"], ax=ax[11], color="#08519c", edgecolor=None)#red

	sns.barplot(x=microhomo2["xcat"], y=microhomo2["amount"], ax=ax[12], color="#cbc9e2", edgecolor=None)
	sns.barplot(x=microhomo3["xcat"], y=microhomo3["amount"], ax=ax[13], color="#9e9ac8", edgecolor=None)
	sns.barplot(x=microhomo4["xcat"], y=microhomo4["amount"], ax=ax[14], color="#6a51a3", edgecolor=None)
	sns.barplot(x=microhomo5["xcat"], y=microhomo5["amount"], ax=ax[15], color="#54278f", edgecolor=None)

	for i in range(1,16):
		sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=False)
		ax[i].tick_params(axis='y', which='major', length=0)
		ax[i].set_ylabel("")
	#loop affects all axis
	for i, j in enumerate(ax):
		ax[i].tick_params(axis='x', which='both', length=0)
		#plt.setp(ax[i].get_xticklabels(), rotation=90, fontsize=4)
		ax[i].set_xlabel("")
		ax[i].grid(b=True, which='both', axis='y', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-1)
		ax[i].tick_params(axis='x', which='major', labelsize=3.2, pad=1)
		ax[i].set_xticklabels(ax[i].get_xticklabels(), fontname="DejaVu Sans Mono")
		# ax[i].set_yticks([0.00, 0.05, 0.10])
		ax[i].set_ylim([0.0, normalized_sigs["amount"].max()])
		ax[i].set_ylim([0.0, 0.28])

	# #fasr left axis
	ax[0].set_ylabel("Proportion", fontsize=7, horizontalalignment="center", labelpad=0.5)
	# ax[0].set_yticks([0.00,0.10])
	ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax[0].tick_params(axis='y', which='major', labelsize=5, pad=-5)
	# ax[0].yaxis.set_label_coords(-0.42, 0.5)

	sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)
	ht = 0.05
	# top_of_graph = normalized_sigs["amount"].max()

	top_of_graph = normalized_sigs["amount"].max()
	top_of_graph = 0.28
	top_of_graph_plusspace = top_of_graph+(0.05*top_of_graph)
	top_of_graph_plusplusspace = top_of_graph+(0.05*top_of_graph)+(0.02*top_of_graph)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#fdae6b", edgecolor=None, clip_on=False)
	ax[0].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#e6550d", edgecolor=None, clip_on=False)
	ax[1].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#a1d99b", edgecolor=None, clip_on=False)
	ax[2].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#31a354", edgecolor=None, clip_on=False)
	ax[3].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#fee5d9", edgecolor=None, clip_on=False)
	ax[4].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#fcae91", edgecolor=None, clip_on=False)
	ax[5].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#fb6a4a", edgecolor=None, clip_on=False)
	ax[6].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#cb181d", edgecolor=None, clip_on=False)
	ax[7].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#bdd7e7", edgecolor=None, clip_on=False)
	ax[8].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#6baed6", edgecolor=None, clip_on=False)
	ax[9].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#3182bd", edgecolor=None, clip_on=False)
	ax[10].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=6, height=ht, fill=True, facecolor="#08519c", edgecolor=None, clip_on=False)
	ax[11].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=1, height=ht, fill=True, facecolor="#cbc9e2", edgecolor=None, clip_on=False)
	ax[12].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=2, height=ht, fill=True, facecolor="#9e9ac8", edgecolor=None, clip_on=False)
	ax[13].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=3, height=ht, fill=True, facecolor="#6a51a3", edgecolor=None, clip_on=False)
	ax[14].add_patch(rect)
	rect = Rectangle((-0.5,top_of_graph_plusspace), width=5, height=ht, fill=True, facecolor="#54278f", edgecolor=None, clip_on=False)
	ax[15].add_patch(rect)

	ax[0].annotate("C",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[1].annotate("T",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[2].annotate("C",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[3].annotate("T",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[4].annotate("2",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[5].annotate("3",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[6].annotate("4",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[7].annotate("5+", (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[8].annotate("2",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[9].annotate("3",  (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[10].annotate("4", (2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[11].annotate("5+",(2,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[12].annotate("2",(-0.35,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[13].annotate("3", (0,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[14].annotate("4", (0.5,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)
	ax[15].annotate("5+",(1,top_of_graph_plusplusspace), fontsize=5, va="bottom", ha="left", annotation_clip=False)

	lns=1 #linespacing
	lp = 0.45
	ylc = -0.09 #y label vertical coordiante
	ax[0].set_xlabel("1bp Deletion\n(Homopol length)", fontsize=4, horizontalalignment="center", labelpad=lp, ma='center', linespacing=lns)
	ax[0].xaxis.set_label_coords(1, ylc)

	ax[2].set_xlabel("1bp Insertion\n(Homopol length)", fontsize=4, horizontalalignment="center", labelpad=lp, ma='center', linespacing=lns)
	ax[2].xaxis.set_label_coords(1, ylc)

	ax[5].set_xlabel("Deletion Length at Repeats\n(Number of Repeats)", fontsize=4, horizontalalignment="center", labelpad=lp, ma='center', linespacing=lns)
	ax[5].xaxis.set_label_coords(1, ylc)

	ax[9].set_xlabel("Insertion Length at Repeats\n(Number of Repeats)", fontsize=4, horizontalalignment="center", labelpad=lp, ma='center', linespacing=lns)
	ax[9].xaxis.set_label_coords(1, ylc)

	ax[14].set_xlabel("Microhomology\n(Microhomology Length)", fontsize=4, horizontalalignment="center", labelpad=lp, ma='center', linespacing=lns)
	ax[14].xaxis.set_label_coords(0.5, ylc)

	plt.subplots_adjust(wspace=0.01, left=0.055, right=0.998, bottom=0.17, top=0.85)
	return fig, ax

#%% plot average naive copy number signature for cohort or subset of cohort
# =============================================================================

def CN_naive_signature(df_sig_count):
	
	BCper10mb_categories = 4 # 0, 1, 2, >2
	CN_categories = 9 # 0, 1, 2, 3, 4, 5, 6, 7, >7
	CNCP_categories = 8 # 0, 1, 2, 3, 4, 5, 6, >6
	BCperCA_categories = 6 # 0, 1, 2, 3, 4, >4
	SegSize_categories = 11 # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, >9
	CNFraction_categories = 7 # 0, 1, 2, 3, 4, 5, >5
	
	#normalize within categories so that each category sums to 1
	def normalize_the_table(count_table):
		#table = count_table.copy(deep=True)
		table = count_table[cnv_categories].drop(columns=["sample"])
		#sum each row of each column and make a dataframe
		sum_values = list(table.sum(axis=0))
		header = table.columns
		table = pd.DataFrame(data=[sum_values], columns=header)
		
		BCper10mb_cat = BCper10mb_categories
		total = float(table.iloc[:,0:BCper10mb_cat].sum(axis=1))
		table.iloc[:,0:BCper10mb_cat] = table.iloc[:,0:BCper10mb_cat].apply(lambda x: x/total, axis=1)
		
		CN_cat = CN_categories + BCper10mb_cat
		total = float(table.iloc[:,BCper10mb_cat:CN_cat].sum(axis=1))
		table.iloc[:,BCper10mb_cat:CN_cat] = table.iloc[:,BCper10mb_cat:CN_cat].apply(lambda x: x/total, axis=1)
		
		CNCP_cat = CNCP_categories + CN_cat
		total = float(table.iloc[:,CN_cat:CNCP_cat].sum(axis=1))
		table.iloc[:,CN_cat:CNCP_cat] = table.iloc[:,CN_cat:CNCP_cat].apply(lambda x: x/total, axis=1)
		
		BCperCA_cat = BCperCA_categories + CNCP_cat
		total = float(table.iloc[:,CNCP_cat:BCperCA_cat].sum(axis=1))
		table.iloc[:,CNCP_cat:BCperCA_cat] = table.iloc[:,CNCP_cat:BCperCA_cat].apply(lambda x: x/total, axis=1)
		
		SegSize_cat = SegSize_categories + BCperCA_cat
		total = float(table.iloc[:,BCperCA_cat:SegSize_cat].sum(axis=1))
		table.iloc[:,BCperCA_cat:SegSize_cat] = table.iloc[:,BCperCA_cat:SegSize_cat].apply(lambda x: x/total, axis=1)
		
		CNFraction_cat = CNFraction_categories + SegSize_cat
		total = float(table.iloc[:,SegSize_cat:CNFraction_cat].sum(axis=1))
		table.iloc[:,SegSize_cat:CNFraction_cat] = table.iloc[:,SegSize_cat:CNFraction_cat].apply(lambda x: x/total, axis=1)
		
		table = table.round(decimals=3)
		return table
	
	normtable = normalize_the_table(df_sig_count)
	
	# make the figure like https://www.nature.com/articles/s41588-018-0179-8/figures/2
	sns.set()
	sns.set_style(style="whitegrid")
	fig, ax = plt.subplots(nrows=1, ncols=6, sharey="all", figsize=(4.4, 1.0), gridspec_kw={"width_ratios":[4,9,8,6,11,7]})
	sns.set_style(style="whitegrid")
	plt.rcParams["font.family"] = "monospace"
	plt.subplots_adjust(wspace=0.0)

	# category colors
	BCper10mb_color = '#4c72b0'
	CN_color = '#dd8452'
	CNCP_color = '#55a868'
	BCperCA_color = '#c44e52'
	SegSize_color = '#8172b3'
	CNFraction_color = '#937860'
	
	#BCper10mb
	BCper10mb_cat = BCper10mb_categories
	x=[str(x) for x in range(0,BCper10mb_cat-1)]
	x.append(">2")
	value_index_first = 0
	value_index_last = BCper10mb_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[0], color=BCper10mb_color)

	#CN
	CN_cat = CN_categories
	x=[str(x) for x in range(0,CN_cat-1)]
	x.append(">7")
	value_index_first = value_index_last
	value_index_last = value_index_first + CN_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[1], color=CN_color)

	#CNCP
	CNCP_cat = CNCP_categories
	x=[str(x) for x in range(0,CNCP_cat-1)]
	x.append(">6")
	value_index_first = value_index_last
	value_index_last = value_index_first + CNCP_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[2], color=CNCP_color)

	#BCperCA
	BCperCA_cat = BCperCA_categories
	x=[str(x) for x in range(0,BCperCA_cat-1)]
	x.append(">4")
	value_index_first = value_index_last
	value_index_last = value_index_first + BCperCA_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[3], color=BCperCA_color)

	#SegSize
	SegSize_cat = SegSize_categories
	x=[str(x) for x in range(0,SegSize_cat-1)]
	x.append(">9")
	value_index_first = value_index_last
	value_index_last = value_index_first + SegSize_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[4], color=SegSize_color)

	#CNFraction
	CNFraction_cat = CNFraction_categories
	x=[str(x) for x in range(0,CNFraction_cat-1)]
	x.append(">5")
	value_index_first = value_index_last
	value_index_last = value_index_first + CNFraction_cat
	y=normtable[normtable.columns[value_index_first:value_index_last]].loc[0, :].values.tolist()
	sns.barplot(x=x, y=y, ax=ax[5], color=CNFraction_color)
	
	for i in range(1,6):
		sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=False)
		ax[i].tick_params(axis='y', which='major', length=0)
		ax[i].set_ylabel("")
		#loop affects all axis
	for i, j in enumerate(ax):
		ax[i].tick_params(axis='x', which='both', length=0)
		#plt.setp(ax[i].get_xticklabels(), rotation=90, fontsize=4)
		ax[i].set_xlabel("")
		ax[i].grid(b=True, which='both', axis='y', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-1)
		ax[i].tick_params(axis='x', which='major', labelsize=5, pad=1)
		ax[i].set_xticklabels(ax[i].get_xticklabels(), fontname="DejaVu Sans Mono")
		# ax[i].set_yticks([0.00, 0.05, 0.10])
		ax[i].set_ylim(0.0,1.1)

	# #fasr left axis
	ax[0].set_ylabel("Proportion", fontsize=7, horizontalalignment="center", labelpad=0.5)
	ax[0].set_yticks([0.0,0.5])
	ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax[0].tick_params(axis='y', which='major', labelsize=5, pad=-5)
	# ax[0].yaxis.set_label_coords(-0.42, 0.5)
	sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

# 	sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)
# 	for i in range(1,6):
# 		sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=False)

# 	for i, j in enumerate(ax):
# 		ax[i].tick_params(axis="x", which="both", length=0)
# 		plt.setp(ax[i].get_xticklabels(), rotation=0, fontname="Arial")
# 		ax[i].tick_params(labelsize=5)
# 	ax[0].tick_params(axis="y", which="both", length=0)
# 	ax[0].set_ylim(0.0,1.1)
	#xlabel params
	fs=5
	lp=0.4

	rect = Rectangle((-0.5,0.91), width=BCper10mb_cat, height=0.42, fill=True, facecolor=BCper10mb_color, edgecolor=None, clip_on=False)
	ax[0].add_patch(rect)
	ax[0].annotate("Breaks\nper 10Mb", (BCper10mb_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[0].set_xlabel("(BCper10mb)", fontsize=fs, horizontalalignment="center", labelpad=lp)

	rect = Rectangle((-0.5,0.91), width=CN_cat, height=0.42, fill=True, facecolor=CN_color, edgecolor=None, clip_on=False)
	ax[1].add_patch(rect)
	ax[1].annotate("Copy number count", (CN_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[1].set_xlabel("(CN)", fontsize=fs, horizontalalignment="center", labelpad=lp)

	rect = Rectangle((-0.5,0.91), width=CNCP_cat, height=0.42, fill=True, facecolor=CNCP_color, edgecolor=None, clip_on=False)
	ax[2].add_patch(rect)
	ax[2].annotate("Difference between\nadjacent segments", (CNCP_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[2].set_xlabel("(CNCP)", fontsize=fs, horizontalalignment="center", labelpad=lp)

	rect = Rectangle((-0.5,0.91), width=BCperCA_cat, height=0.42, fill=True, facecolor=BCperCA_color, edgecolor=None, clip_on=False)
	ax[3].add_patch(rect)
	ax[3].annotate("Breaks per\nchromosome arm", (BCperCA_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[3].set_xlabel("(BCperCA)", fontsize=fs, horizontalalignment="center", labelpad=lp)

	rect = Rectangle((-0.5,0.91), width=SegSize_cat, height=0.42, fill=True, facecolor=SegSize_color, edgecolor=None, clip_on=False)
	ax[4].add_patch(rect)
	ax[4].annotate("Segment size", (SegSize_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[4].set_xlabel("(SegSize)", fontsize=fs, horizontalalignment="center", labelpad=lp)

	rect = Rectangle((-0.5,0.91), width=CNFraction_cat, height=0.42, fill=True, facecolor=CNFraction_color, edgecolor=None, clip_on=False)
	ax[5].add_patch(rect)
	ax[5].annotate("Fraction of genome", (CNFraction_cat/2-0.5,1.1), fontsize=5, va="center", ha="center", ma="center", linespacing=1.0, clip_on=False, fontname="Arial")
	ax[5].set_xlabel("(CopyFraction)", fontsize=fs, horizontalalignment="center", labelpad=lp)

# 	ax[0].set_ylabel("Proportion", fontsize=8, horizontalalignment="center", fontname="Arial", labelpad=0.5)

	plt.subplots_adjust(wspace=0.00, left=0.055, right=0.998, bottom=0.15, top=0.87)
	sns.set_style(style="whitegrid")
	return fig, ax

def plotfigs(df, subtype):
	df_sub = df.query('(label == @subtype)')
	SNV_naive_signature(df_sub)
	plt.savefig(os.path.join(figdir, f"{subtype}_snv96.png"), dpi=500)
	plt.savefig(os.path.join(figdir, f"{subtype}_snv96.pdf"))
	# plt.close()
	INDEL_naive_signature(df_sub)
	plt.savefig(os.path.join(figdir, f"{subtype}_indel83.png"), dpi=500)
	plt.savefig(os.path.join(figdir, f"{subtype}_indel83.pdf"))
	# plt.close()
	CN_naive_signature(df_sub)
	plt.savefig(os.path.join(figdir, f"{subtype}_segs45.png"), dpi=500)
	plt.savefig(os.path.join(figdir, f"{subtype}_segs45.pdf"))

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig2")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)

#%% ==========================================================
# make and save plots
# ============================================================

plotfigs(df, "BRCA2d")
plotfigs(df, "CDK12d")
plotfigs(df, "MMRd")
plotfigs(df, "DRwt")

#%%
