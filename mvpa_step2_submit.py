"""
Notes

train_str = identifies output of mvpa_step1 to be used for training
"""


import os
import subprocess
import fnmatch
import time
from datetime import datetime


def main():

    # set up
    sess_str = "ses-S1"
    train_str = "MVPA_loc_single"
    parent_dir = "/scratch/madlab/nate_vCAT"
    deriv_dir = os.path.join(parent_dir, "derivatives")
    group_dir = os.path.join(parent_dir, "analyses")
    code_dir = "/home/nmuncy/compute/learn_mvpa"

    # submit job
    current_time = datetime.now()
    out_dir = os.path.join(
        deriv_dir, f'Slurm_out/MVPA2_{current_time.strftime("%H%M_%d-%m-%y")}'
    )
    os.makedirs(out_dir)

    # submit for e/subj
    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    for subj in subj_list:

        # Set stdout/err file
        h_out = os.path.join(out_dir, f"out_{subj}_train.txt")
        h_err = os.path.join(out_dir, f"err_{subj}_train.txt")

        # submit command
        subj_dir = os.path.join(deriv_dir, subj, sess_str)
        if not os.path.exists(os.path.join(subj_dir, "MVPA_train+tlrc.HEAD")):
            sbatch_job = f"""
                sbatch \
                    -J "MVPA2{subj.split("-")[-1]}" \
                    -t 1:00:00 \
                    --mem=4000 \
                    --ntasks-per-node=1 \
                    -p IB_44C_512G \
                    -o {h_out} -e {h_err} \
                    --account iacc_madlab \
                    --qos pq_madlab \
                    --wrap="~/miniconda3/bin/python {code_dir}/mvpa_step2_train.py \
                        {subj_dir} \
                        {train_str} \
                        {group_dir}"
            """
            sbatch_submit = subprocess.Popen(
                sbatch_job, shell=True, stdout=subprocess.PIPE
            )
            job_id = sbatch_submit.communicate()[0].decode("utf-8")
            print(job_id)
            time.sleep(1)


if __name__ == "__main__":
    main()
