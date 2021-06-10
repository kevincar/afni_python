"""
Notes

test_dict = decon behavior and test (truth) for behavior
test_dict = {"Study": {"Decon": ["fbl"], "Test": ["fblf", "fbls"]}}
    Study = phase
    Decon - fbl = relevant volumes for testing (from cbucket)
    Test - behaviors = truth of classification (same length as
        number of training categories, in same order)
        These also correspond to tf e.g. tf_Study_fblf.txt
"""

import os
import time
import json
import fnmatch
import subprocess
from datetime import datetime


def main():

    sess = "ses-S1"
    test_dict = {"Study": {"Decon": ["fbl"], "Test": ["fblf", "fbls"]}}
    deriv_dir = "/scratch/madlab/nate_vCAT/derivatives"
    code_dir = "/home/nmuncy/compute/learn_mvpa"

    # Work
    current_time = datetime.now()
    out_dir = os.path.join(
        deriv_dir, f'Slurm_out/MVPA3_{current_time.strftime("%H%M_%d-%m-%y")}'
    )
    os.makedirs(out_dir)

    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    for subj in subj_list:

        # write json
        subj_dir = os.path.join(deriv_dir, subj, sess)
        with open(os.path.join(subj_dir, "test_dict.json"), "w") as outfile:
            json.dump(test_dict, outfile)

        # Set stdout/err file
        h_out = os.path.join(out_dir, f"out_{subj}.txt")
        h_err = os.path.join(out_dir, f"err_{subj}.txt")

        # submit command
        sbatch_job = f"""
            sbatch \
                -J "MVPA3{subj.split("-")[1]}" \
                -t 3:00:00 \
                --mem=8000 \
                --ntasks-per-node=1 \
                -p IB_44C_512G \
                -o {h_out} -e {h_err} \
                --account iacc_madlab \
                --qos pq_madlab \
                --wrap="~/miniconda3/bin/python {code_dir}/mvpa_step3_test.py \
                    {subj_dir}"
        """
        sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
        job_id = sbatch_submit.communicate()[0].decode("utf-8")
        print(job_id)
        time.sleep(1)


if __name__ == "__main__":
    main()
