# -*- coding: utf-8 -*-
"""
@author: Elie
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
if os.name == 'posix' and "DISPLAY" not in os.environ:
	mpl.use('Agg')
from matplotlib.patches import Rectangle
pd.options.mode.chained_assignment = None

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--segmentation_file", help="Path to seg file (tab seperated with header=[chr, start, end, CNt] each row describing a genomic segments position and copynumber)", dest="seg_path", type=str, required=True)
	parser.add_argument("-c", "--centromere_file", help="Path to centromere file (http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz)", dest="centro_path", type=str, required=True)
	parser.add_argument("-o", "--output_path", help="Output directory for the output files, tsv and figures" ,dest="output_dir", type=str, required=True)
	parser.add_argument("-sn", "--sample_name", help="All files will have prefix sample_name_", dest="sn", type=str, required=True)

	args = parser.parse_args()
	args.seg_path = os.path.expanduser(args.seg_path)
	args.centro_path = os.path.expanduser(args.centro_path)
	args.output_dir = os.path.expanduser(args.output_dir)
	args.sn = str(args.sn)

	# =============================================================================
	# load data
	# =============================================================================

	"""use for testing"""
	# seg_path = "C:/Users/Elie/Desktop/signature_apr2020/scripts/seggenerator_v3/test_cnmatrixgenerator/M1RP_ID1_MLN4_seg.txt"
	# centro_path = "C:/Users/Elie/Desktop/signature_apr2020/scripts/seggenerator_v3/test_cnmatrixgenerator/cytoBandhg38.txt"
	# out_path = "C:/Users/Elie/Desktop/signature_apr2020/scripts/seggenerator_v3/test_cnmatrixgenerator/copy_sig_matrix.tsv"
	
	seg = pd.read_csv(args.seg_path, sep="\t", low_memory=False, dtype={"chromosome":str, "start.pos":np.int32, "end.pos":np.int32, "CNt":np.int32}) #load seg file
	seg = seg[["chromosome", "start.pos", "end.pos", "CNt"]].rename(columns={"chromosome":"chr", "start.pos":"start", "end.pos":"end"})
	centro = pd.read_csv(args.centro_path, sep="\t", low_memory=False, names=["chr", "start", "end", "band", "annot"], dtype={"chr":str, "start":np.int32, "end":np.int32, "band":str, "annot":str})

	# =============================================================================
	# Number of categories for each class
	# =============================================================================

	BCper10mb_categories = 4 # 0, 1, 2, >2
	CN_categories = 9 # 0, 1, 2, 3, 4, 5, 6, 7, >7
	CNCP_categories = 8 # 0, 1, 2, 3, 4, 5, 6, >6
	BCperCA_categories = 6 # 0, 1, 2, 3, 4, >4
	SegSize_categories = 11 # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, >9
	CNFraction_categories = 7 # 0, 1, 2, 3, 4, 5, >5

	# =============================================================================
	# make cytoband file into df with arm starts and ends
	# inputs = path to cytoband file and segfile data frame
	# outputs = chromosome arm list, chromosome list and chr_arm defining arm boundaries
	# =============================================================================
	'''turn cytoband file from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz 
	to a dataframe with columns ['chr', 'start', 'end', 'arm', 'chr_arm'].
	p starts at 0, goes to start of centromere. q starts at q centromere and goes to max position in
	the seg file position'''
	def make_centro(seg,centro):
		centro.columns = ["chr", "start", "end", "band", "annot"] #centromere position file
		#get rid of stupid chr in chrom names
		centro["chr"] = centro["chr"].replace("chr", "", regex=True)
		#acen is a stain that ids centromeres. 
		#This removes all the cyto bands that arent centromeres
		centro = centro.query('annot == "acen"')
		centro["arm"] = centro["band"].replace("[0-9.]", "", regex=True)
		#p arm goes 0 to centromere
		#q goes centromere to max seg position
		centro.loc[centro["arm"] == "p", "start"] = 0
		maxseg = seg.groupby("chr")["end"].max().reset_index()
		centro = pd.merge(centro, maxseg, how="left", on="chr").dropna()
		centro.loc[centro["arm"] == "q", "end_x"] = centro["end_y"]
		centro = centro.rename(columns={"end_x":"end"})
		centro = centro.drop(columns=["band", "annot", "end_y"]).reset_index(drop=True)
		# chr_arm for later groupbys
		centro["chr_arm"] = centro[["chr", "arm"]].apply(lambda x: "".join(x), axis=1)
		centro = centro.rename(columns={"start":"arm_start", "end":"arm_end"})
		# type checks, if no x and y pandas turns chr to ints
		centro["arm_start"] = centro["arm_start"].astype(dtype=int, errors="ignore")
		centro["arm_end"] = centro["arm_end"].astype(dtype=int, errors="ignore")
		centro["chr"] = centro["chr"].astype(dtype=str, errors="ignore")
		return centro

	chr_arms = make_centro(seg, centro)

	chroms = chr_arms[["chr", "chr_arm"]]
	chroms["chrom"] = chroms["chr"].replace("X", "23").replace("Y", "24").astype(int)
	chroms = chroms.sort_values(["chrom", "chr_arm"])
	chr_list = [chrom for chrom in chroms["chr"].drop_duplicates().tolist()]
	chr_arm_list = [chrarm for chrarm in chroms["chr_arm"].drop_duplicates().tolist()]

	# =============================================================================
	# make chrom arm dict and chrom dict
	# =============================================================================

	'''want to handle the case that segment overlap the centromere, starting in p and ending in q
	define a segment in p as its start is betwwen 0 to centromere
	define q as end is after centromere'''
	segs_arms = pd.merge(seg,chr_arms, how="left", on="chr")
	segs_p_arms = segs_arms.query('(arm == "p")')
	segs_p_arms = segs_p_arms.query('(start < arm_end)')
	segs_q_arms = segs_arms.query('(arm == "q")')
	segs_q_arms = segs_q_arms.query('(end > arm_start)')
	segs_all_arms = pd.concat([segs_p_arms, segs_q_arms])
	# keys are each chromosome arm and values are dataframes with all segs in that arm
	chr_arm_seg_dict = {chrarm:segs_all_arms.query('(chr_arm == @chrarm)').reset_index(drop=True) for chrarm in chr_arm_list}
	chr_seg_dict = {chrom:segs_all_arms.query('(chr == @chrom)').reset_index(drop=True) for chrom in chr_list}

	# =============================================================================
	# break point count per chromosome arm is number of segments per chr arm -1
	# takes the chr_arm_seg_dict from above and N = number of categories (0-N) where last category is >=N
	# =============================================================================

	'''breaks per arm is = to number of segments - 1'''
	def count_number_of_breaks_per_arm(chr_arm_seg_dict, N):
		seg_counts = []
		chrom_arm = []
		for index, (arm, data) in enumerate(chr_arm_seg_dict.items()):
			number_of_segs = len(data.index)
			seg_counts.append(number_of_segs)
			chrom_arm.append(arm)
		number_of_segs_per_arm = dict(zip(chrom_arm,seg_counts))
		number_of_breaks_per_arm = {chromarm:(number_of_segs_per_arm[chromarm]-1) for chromarm in chr_arm_list if number_of_segs_per_arm[chromarm] > 0} #exome will have some zeros
		#then make a count of values across the genome
		
		number_of_breaks_per_arm_count_vector = []
		for i in range(0,N):
			number_of_breaks_per_arm_count_vector.append(sum(1 for value in number_of_breaks_per_arm.values() if value == i))
		number_of_breaks_per_arm_count_vector.append(sum(1 for value in number_of_breaks_per_arm.values() if value > (N-1)))
		return number_of_breaks_per_arm_count_vector

	breaks_per_arm_count_vector = count_number_of_breaks_per_arm(chr_arm_seg_dict, 5)

	# =============================================================================
	# number of breaks per 10mb
	# takes the chr_seg_dict from above and N = number of categories (0-N) where last category is >=N
	# =============================================================================
	''' make a dictionary called window_dict that divides each chrom by 10mb
	then loop over each of those lists. count number of segments that overlap position i,
	but stop before i+1.'''
	def count_number_of_breaks_per_10mb(chr_seg_dict, N):
		positions = []
		chroms = []
		for chromosome, data in chr_seg_dict.items():
			smallest = int(data["start"].min())
			largest = int(data["end"].max())
			position_list = list(range(smallest,largest,10000000))
			positions.append(position_list)
			chroms.append(chromosome)
		window_dict = dict(zip(chroms, positions))
		
		ten_mb_windows_counts = []
		for index, (chromosome, position_lists) in enumerate(window_dict.items()):
			for i, position in enumerate(position_lists[:-1]):
				first_position = position_lists[i]
				second_position = position_lists[i+1]
				segs_that_broke = chr_seg_dict[chromosome].query('(start < @first_position) and (end > @first_position) and (end < @second_position)').reset_index(drop=True)
				number_of_seg_endpoints_per10mb = len(segs_that_broke.index)
				ten_mb_windows_counts.append(number_of_seg_endpoints_per10mb)
		
		number_of_breaks_per_10mbwindow_count_vector = []
		for i in range(0,N):
			number_of_breaks_per_10mbwindow_count_vector.append(sum(1 for value in ten_mb_windows_counts if value == i))
		number_of_breaks_per_10mbwindow_count_vector.append(sum(1 for value in ten_mb_windows_counts if value > (N-1)))
		return number_of_breaks_per_10mbwindow_count_vector

	breaks_per_10mb_count_vector = count_number_of_breaks_per_10mb(chr_seg_dict, 3)

	# =============================================================================
	# number of segments with copynumber N 
	# takes the seg dataframe and and N = number of categories (0-N) where last category is >=N
	# =============================================================================
	def count_number_of_segments_with_Ncopies(seg, N):
		number_of_segs_with_copynumber_count_vector = []
		for i in range(0,N):
			number_of_segs_with_copynumber_count_vector.append(sum(1 for value in seg["CNt"].tolist() if value == i))
		number_of_segs_with_copynumber_count_vector.append(sum(1 for value in seg["CNt"].tolist() if value > (N-1)))
		return number_of_segs_with_copynumber_count_vector

	segments_with_Ncopies_count_vector = count_number_of_segments_with_Ncopies(seg, 8)

	# =============================================================================
	# difference in copy number between segments
	# takes the chr_seg_dict from above and N = number of categories (0-N) where last category is >=N
	# =============================================================================
	def count_number_of_copy_difference_between_segments(chr_seg_dict, N):
		change_list = []
		for index, (chromosome, data) in enumerate(chr_seg_dict.items()):
			changes = data["CNt"].shift(1) - data["CNt"]
			changes = changes.dropna().tolist()
			for number in changes:
				change_list.append(int(abs(number)))
		
		number_of_copy_difference_between_seg_count_vector = []
		for i in range(0,N):
			number_of_copy_difference_between_seg_count_vector.append(sum(1 for value in change_list if value == i))
		number_of_copy_difference_between_seg_count_vector.append(sum(1 for value in change_list if value > (N-1)))
		return number_of_copy_difference_between_seg_count_vector

	copy_difference_between_seg_count_vector = count_number_of_copy_difference_between_segments(chr_seg_dict, 7)

	# =============================================================================
	# Segment size
	# =============================================================================
	def count_segment_sizes(segment_DF, N):
		segment = segment_DF.copy(deep=True)
		segment["length"] = segment["end"] - segment["start"]
		segment["10mb_length"] = segment["length"] / 10000000
		segment["10mb_length_round"] = segment["10mb_length"].round(0)
		number_of_segment_sizes_count_vector = []
		for i in range(0,N):
			number_of_segment_sizes_count_vector.append(sum(1 for value in segment["10mb_length_round"].tolist() if value == i))
		number_of_segment_sizes_count_vector.append(sum(1 for value in segment["10mb_length_round"].tolist() if value > (N-1)))
		return number_of_segment_sizes_count_vector

	segment_size_count_vector = count_segment_sizes(seg, 10)

	# =============================================================================
	# Copy number proportion
	# =============================================================================
	def proportion_of_genome_with_copynumber(segment_DF, N):
		segment = segment_DF.copy(deep=True)
		segment["length"] = segment["end"] - segment["start"]
		total_genome_size = segment["length"].sum()
		#segment["fraction_of_genome"] = segment["length"] / total_genome_size
		copy_fractions = []
		for i in range(0,N):
			length_of_copynum = segment.query('(CNt == @i)')["length"].sum()
			fraction_of_genome = length_of_copynum / total_genome_size
			copy_fractions.append(fraction_of_genome)
		foo = N-1
		length_of_copynum = segment.query('(CNt > @foo)')["length"].sum()
		fraction_of_genome = length_of_copynum / total_genome_size
		copy_fractions.append(fraction_of_genome)
		copy_fraction_list = list(np.around(np.array(copy_fractions),4))
		return copy_fraction_list

	copy_fraction_vector = proportion_of_genome_with_copynumber(seg, 6)
	

	# =============================================================================
	# put all the vectors into one table for output and graphing
	# =============================================================================
	def make_the_table_header():
		header = []
		BCper10mb_categories = 4
		for i in range(0,BCper10mb_categories):
			header.append("BCper10mb_"+str(i))
		CN_categories = 9
		for i in range(0,CN_categories):
			header.append("CN_"+str(i))
		CNCP_categories = 8
		for i in range(0,CNCP_categories):
			header.append("CNCP_"+str(i))
		BCperCA_categories = 6
		for i in range(0,BCperCA_categories):
			header.append("BCperCA_"+str(i))
		SegSize_categories = 11
		for i in range(0,SegSize_categories):
			header.append("SegSize_"+str(i))
		CopyFraction_categories = 7
		for i in range(0,CopyFraction_categories):
			header.append("CopyFraction_"+str(i))
		return header

	table_header = make_the_table_header()
	all_count_vectors = breaks_per_10mb_count_vector + segments_with_Ncopies_count_vector + copy_difference_between_seg_count_vector + breaks_per_arm_count_vector + segment_size_count_vector + copy_fraction_vector
	table_of_count_vectors = pd.DataFrame(data=[all_count_vectors], columns=table_header)
	table_of_count_vectors.to_csv(args.output_dir+"/"+args.sn+"_copy_feature_matrix.tsv", sep="\t", index=False)

	# =============================================================================
	# make normalized table
	# =============================================================================
	''' normalize within categories so that each category sums to 1'''
	def normalize_the_table(count_table):
		table = count_table.copy(deep=True)
		
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
		
		table = table.round(decimals=3)
		return table

	normtable = normalize_the_table(table_of_count_vectors)

	# =============================================================================
	# make a figure similar to 
	# https://www.nature.com/articles/s41588-018-0179-8/figures/2
	# =============================================================================

	sns.set()
	sns.set_style(style="whitegrid")
	fig, ax = plt.subplots(nrows=1, ncols=6, sharey="all", figsize=(12, 4), gridspec_kw={"width_ratios":[4,9,8,6,11,7]})
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
	x.append(">5")
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


	sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)
	for i in range(1,6):
		sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=False)

	for i, j in enumerate(ax):
		ax[i].tick_params(axis="x", which="both", length=0)
		plt.setp(ax[i].get_xticklabels(), rotation=0, fontname="Arial")
		ax[i].tick_params(labelsize=13)
	ax[0].tick_params(axis="y", which="both", length=0)
	ax[0].set_ylim(0.0,1.0)

	rect = Rectangle((-0.5,0.91), width=BCper10mb_cat, height=0.18, fill=True, facecolor=BCper10mb_color, edgecolor=None, clip_on=False)
	ax[0].add_patch(rect)
	ax[0].annotate("Breakpoints\nper 10Mb", (BCper10mb_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	rect = Rectangle((-0.5,0.91), width=CN_cat, height=0.18, fill=True, facecolor=CN_color, edgecolor=None, clip_on=False)
	ax[1].add_patch(rect)
	ax[1].annotate("Copy number\ncount", (CN_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	rect = Rectangle((-0.5,0.91), width=CNCP_cat, height=0.18, fill=True, facecolor=CNCP_color, edgecolor=None, clip_on=False)
	ax[2].add_patch(rect)
	ax[2].annotate("Difference in copy\nnumber between\nadjacent segments", (CNCP_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	rect = Rectangle((-0.5,0.91), width=BCperCA_cat, height=0.18, fill=True, facecolor=BCperCA_color, edgecolor=None, clip_on=False)
	ax[3].add_patch(rect)
	ax[3].annotate("Breakpoints per\nchromosome arm", (BCperCA_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	rect = Rectangle((-0.5,0.91), width=SegSize_cat, height=0.18, fill=True, facecolor=SegSize_color, edgecolor=None, clip_on=False)
	ax[4].add_patch(rect)
	ax[4].annotate("Segment size", (SegSize_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	rect = Rectangle((-0.5,0.91), width=CNFraction_cat, height=0.18, fill=True, facecolor=CNFraction_color, edgecolor=None, clip_on=False)
	ax[5].add_patch(rect)
	ax[5].annotate("Fraction of genome", (CNFraction_cat/2-0.5,1.0), fontsize=13, va="center", ha="center", ma="center", linespacing=1.05, clip_on=False, fontname="Arial")

	ax[0].set_ylabel("Proportion of Class", fontsize=20, horizontalalignment="center", fontname="Arial")

	plt.subplots_adjust(wspace=0.0, left=0.055, right=0.995, bottom=0.055, top=0.92)
	sns.set_style(style="whitegrid")
	plt.savefig(args.output_dir+"/"+args.sn+"_copy_feature_matrix.pdf", dpi=900)
	plt.savefig(args.output_dir+"/"+args.sn+"_copy_feature_matrix.png", dpi=900)

if __name__ == "__main__":
	main()
