#!/bin/bash

parDir=/scratch/madlab/nate_vCAT
dsetDir=${parDir}/dset
sourceDir=${parDir}/sourcedata

print=${parDir}/missing_list.txt
>$print

cd $dsetDir
for subj in sub-*; do

    unset fileList

    subjPath=${dsetDir}/${subj}/ses-S1
    fileList=(
        ${subjPath}/anat/${subj}_ses-S1_T1w.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-loc_run-1_bold.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-loc_run-2_bold.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-Study_run-1_bold.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-Study_run-2_bold.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-Study_run-3_bold.nii.gz
        ${subjPath}/func/${subj}_ses-S1_task-Study_run-4_bold.nii.gz
        ${subjPath}/fmap/${subj}_ses-S1_acq-func_dir-AP_epi.nii.gz
        ${subjPath}/fmap/${subj}_ses-S1_acq-func_dir-PA_epi.nii.gz
    )

    fileCount=0
    for file in ${fileList[@]}; do
        if [ -f $file ]; then
            let fileCount+=1
        else
            echo -e $subj \t "${file##*\/}" >> $print
        fi
    done

    if [ $fileCount == ${#fileList[@]} ]; then
        rm -r ${sourceDir}/$subj
    fi
done


