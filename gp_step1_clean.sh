#!/bin/bash

parDir=/scratch/madlab/nate_vCAT
dsetDir=${parDir}/dset
derivDir=${parDir}/derivatives

cd $derivDir
for subj in sub-*; do

    cd ${subj}/ses-S1

    fileA=struct_ns+tlrc.HEAD
    fileB=run-1_loc_scale+tlrc.HEAD
    fileC=run-1_Study_scale+tlrc.HEAD

    # chop some fat
    if [ -f $fileA ] && [ -f $fileB ] && [ -f $fileC ]; then

        rm tmp_*
        rm -r a*
        rm run*+orig.*
        rm run*blur*
        rm run*volreg_clean*
        rm run*warp*
        rm sbatch*
        rm struct_al*
        rm struct_flip*
        rm struct_un*
        rm struct+orig*
        rm blip*
        rm -r unWarp*
    fi
    cd $derivDir
done