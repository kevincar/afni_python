#!/bin/bash

module load r-3.5.1-gcc-8.2.0-djzshna

behavDir=/home/data/madlab/Mattfeld_vCAT/behav
derivDir=/scratch/madlab/nate_vCAT/derivatives
codeDir=~/compute/learn_mvpa
phaseArr=("loc" "task")
runArr=(2 4)

# # For testing
# behavDir=/Users/nmuncy/Projects/learn_mvpa/vCAT_data
# derivDir=/Users/nmuncy/Projects/afni_python
# codeDir=$derivDir
# numRuns=2

cd $behavDir
for i in vCAT*; do

	subj=sub-${i#*_}
	dataDir=${behavDir}/$i
	outDir=${derivDir}/${subj}/ses-S1/timing_files
	mkdir -p $outDir

	if [ -d $outDir ]; then
		for j in ${!phaseArr[@]}; do
			Rscript ${codeDir}/gp_step2_timingFiles.R $dataDir $outDir $i ${runArr[$j]} ${phaseArr[$j]}
		done
	fi
done
