#!/bin/bash

workDir=/scratch/madlab/nate_vCAT/derivatives
cd $workDir

for subj in sub-*; do
    cd ${subj}/ses-S1

    fileA=Study_single_stats_REML+tlrc.HEAD
    fileB=Study_single_cbucket_REML+tlrc.HEAD
    fileC=loc_single_cbucket_REML+tlrc.HEAD
    fileD=loc_single_cbucket_REML+tlrc.HEAD

    if [ -f $fileA ] && [ -f $fileB ] && [ -f $fileC ] && [ -f $fileD ]; then
        rm epi_vrBase_ts*
        rm final_mask_{GM,CSF}*
        rm *errts_REML*
        rm *REMLvar*
        rm *WMe_rall*
        rm tmp*
        rm __tt*
        rm *minVal*
    fi
    cd $workDir
done
