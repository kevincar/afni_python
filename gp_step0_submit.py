"""
Notes:

Wrapper script for step0_dcm2nii.py.

Usage - update "set up" section.
"""

# %%
import os
from datetime import datetime
import fnmatch
import subprocess
import time
import json

# set up
code_dir = "/home/nmuncy/compute/RE_gPPI"
dcm_dir = "/home/data/madlab/McMakin_EMU/sourcedata/dicomdir/sourcedata"
work_dir = "/scratch/madlab/nate_ppi"
scan_dict = {"func": "Study", "anat": "4-T1w", "fmap": "8-fMRI"}


# %%
def main():

    # set up out_dir to capture stdout/err
    current_time = datetime.now()
    out_dir = f'derivatives/Slurm_out/TS0_{current_time.strftime("%H%M_%d-%m-%y")}'
    slurm_dir = os.path.join(work_dir, out_dir)

    # set up work_dir
    dir_list = ["dset", "derivatives", out_dir]
    for i in dir_list:
        h_dir = os.path.join(work_dir, i)
        if not os.path.exists(h_dir):
            os.makedirs(h_dir)

    # list of dicom dirs
    dcm_list = [x for x in os.listdir(dcm_dir) if fnmatch.fnmatch(x, "McMakin*")]
    dcm_list.sort()

    # write json to avoid quotation issue
    with open(os.path.join(slurm_dir, "scan_dict.json"), "w") as outfile:
        json.dump(scan_dict, outfile)

    # submit jobs
    # for subj in dcm_list:
    subj = "McMakin_EMU-000-1040-S1"

    subj_str = subj.split("-")[2]
    h_out = os.path.join(slurm_dir, f"out_{subj_str}.txt")
    h_err = os.path.join(slurm_dir, f"err_{subj_str}.txt")

    sbatch_job = f"""
        sbatch \
        -J "GP0{subj_str}" -t 2:00:00 --mem=1000 --ntasks-per-node=1 \
        -p IB_44C_512G  -o {h_out} -e {h_err} \
        --account iacc_madlab --qos pq_madlab \
        --wrap="module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
        python {code_dir}/gp_step0_dcm2nii.py {subj} {dcm_dir} {work_dir} {slurm_dir}"
    """

    sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
    job_id = sbatch_submit.communicate()[0]
    print(job_id)

    time.sleep(1)


if __name__ == "__main__":
    main()
# %%
