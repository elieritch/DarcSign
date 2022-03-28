script="/groups/wyattgrp/eritch/projects/ghent_m1rp/wxs/scripts/runseg/run_sequenza_slurm.bash";
rootoutdir="/groups/wyattgrp/eritch/projects/ghent_m1rp/wxs/sequenza";
resultdir="/groups/wyattgrp/eritch/projects/ghent_m1rp/wxs/results_july7_20";
mkdir -p $rootoutdir;
cd $resultdir;
for i in `ls`;do
samplename=$(echo $i);
tb=$(readlink -ve $resultdir/${samplename}/data/${samplename}.md.ARRG.bam);
nb=$(find `pwd`/${samplename} -iname "*.bam" | grep -v ${tb});
outdir=${rootoutdir}/${samplename};
mkdir -p ${rootoutdir};
printf "sbatch ${script} ${samplename} ${tb} ${nb} ${outdir}\n";
sbatch ${script} ${samplename} ${tb} ${nb} ${outdir};
done;
