#!/bin/bash
#SBATCH --job-name=sequenza
#SBATCH -p debug,express,normal,big-mem,long
#SBATCH --cpus-per-task=10
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 05:59:00 # time (D-HH:MM or HH:MM:SS)
#SBATCH --export=all
#SBATCH --workdir=/groups/wyattgrp/log
#SBATCH --output=/groups/wyattgrp/log/%j.log
#SBATCH --error=/groups/wyattgrp/log/%j.log

printf "SLURM_JOB_ID=$SLURM_JOB_ID\n";

fr=$(free -hm);
echo "total memory:";
echo "$fr";

printf "Start at $(date +"%D %H:%M")\n";

samplename="$1"
tumor="$2"
normal="$3"
outputdir="$4"
ref=$(readlink -ve /groups/wyattgrp/reference/grch38/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa);
gcwig=$(readlink -ve /groups/wyattgrp/reference/grch38/grch38.gc50Base.wig);
tumorbam=$(readlink -ve $tumor);
normalbam=$(readlink -ve $normal);
mkdir -p ${outputdir};

shopt -s extglob;
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sequenzautils;

seqzfile=${outputdir}/${samplename}.seqz;
printf "sequenza-utils bam2seqz --fasta ${ref} -n ${normalbam} -t ${tumorbam} -gc ${gcwig} --het 0.4 -N 40 |  grep -v ^K | grep -v ^G | grep -v ^M | (sed -u 1q; sort -k1,1V -k2,2n) > ${seqzfile}\n";
/groups/wyattgrp/bin/time -v sequenza-utils bam2seqz --fasta ${ref} -n ${normalbam} -t ${tumorbam} -gc ${gcwig} --het 0.4 -N 40 |  grep -v ^K | grep -v ^G | grep -v ^M | (sed -u 1q; sort -k1,1V -k2,2n) > ${seqzfile};

runseq="/groups/wyattgrp/eritch/projects/ghent_m1rp/wxs/scripts/runseg/run_sequenza.R";
printf "Rscript ${runseq} ${seqzfile} ${outputdir} ${samplename}\n";
/groups/wyattgrp/bin/time -v Rscript ${runseq} ${seqzfile} ${outputdir} ${samplename};

printf "Finished at $(date +"%D %H:%M")\n";
cp /groups/wyattgrp/log/$SLURM_JOB_ID.log ${outputdir}/$SLURM_JOB_ID.log;
