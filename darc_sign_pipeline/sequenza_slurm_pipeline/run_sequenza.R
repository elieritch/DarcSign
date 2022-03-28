library("sequenza");
args <- commandArgs(trailingOnly = TRUE);
seqzfile <- args[1];
outputdir <- args[2];
samplename <- args[3];

extractData <- sequenza.extract(seqzfile, window = 10e4, min.reads.normal=20)
extractData.CP <- sequenza.fit(extractData, female = FALSE)
sequenza.results(extractData, extractData.CP, out.dir = outputdir, sample.id = samplename, female = FALSE)
