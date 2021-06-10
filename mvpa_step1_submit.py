"""
Notes:
    train_dict should contain phase, behaviors to train on:
        phase: [beh1, beh2]
    which should correspond to timing files: tf_phase_beh1.txt

    phase could also be decon identifier (foo_bar_stats_REML+tlrc)
        rather than phase_single_stats_REML+tlrc

        job script assumes phase_single atm.
"""

# %%
import os
import time
import json
import fnmatch
import subprocess
from datetime import datetime


# %%
# set up
sess = "ses-S1"
deriv_dir = "/scratch/madlab/nate_vCAT/derivatives"
code_dir = "/home/nmuncy/compute/learn_mvpa"
train_dict = {"loc": ["face", "scene"]}


# %%
def main():

    current_time = datetime.now()
    out_dir = os.path.join(
        deriv_dir, f'Slurm_out/MVPA1_{current_time.strftime("%H%M_%d-%m-%y")}'
    )
    os.makedirs(out_dir)

    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    for subj in subj_list:

        # write json
        subj_dir = os.path.join(deriv_dir, subj, sess)
        with open(os.path.join(subj_dir, "train_dict.json"), "w") as outfile:
            json.dump(train_dict, outfile)

        # Set stdout/err file
        h_out = os.path.join(out_dir, f"out_{subj}.txt")
        h_err = os.path.join(out_dir, f"err_{subj}.txt")

        # submit command
        sbatch_job = f"""
            sbatch \
                -J "MVPA1{subj.split("-")[1]}" \
                -t 2:00:00 \
                --mem=4000 \
                --ntasks-per-node=1 \
                -p IB_44C_512G  \
                -o {h_out} -e {h_err} \
                --account iacc_madlab \
                --qos pq_madlab \
                --wrap="~/miniconda3/bin/python {code_dir}/mvpa_step1_setup.py \
                    {subj} \
                    {sess} \
                    {deriv_dir}"
        """
        sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
        job_id = sbatch_submit.communicate()[0]
        print(job_id)
        time.sleep(1)


if __name__ == "__main__":
    main()
