"""
Notes:

Wrapper script for step3_decon.py.

Update paths in "set up" section.

decon_type can be dmBLOCK, GAM, 2GAM, or TENT
"""

# %%
import os
from datetime import datetime
import fnmatch
import subprocess
import time
import json


# set up
code_dir = "/home/nmuncy/compute/learn_mvpa"
work_dir = "/scratch/madlab/nate_vCAT"
sess_list = ["ses-S1"]
decon_type = "TENT"
decon_dict = {
    "loc": ["tf_loc_face.txt", "tf_loc_num.txt", "tf_loc_scene.txt"],
    "Study": ["tf_Study_fix.txt", "tf_Study_con.txt", "tf_Study_fbl.txt"],
}


def main():

    # set up stdout/err capture
    deriv_dir = os.path.join(work_dir, "derivatives")
    current_time = datetime.now()
    out_dir = os.path.join(
        deriv_dir, f'Slurm_out/TS3_{current_time.strftime("%H%M_%d-%m-%y")}'
    )
    os.makedirs(out_dir)

    # submit job for each subj/sess/phase
    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    # determine which subjs to run
    run_list = []
    for subj in subj_list:
        decon_list = list(decon_dict.keys())
        check_file1 = os.path.join(
            deriv_dir,
            subj,
            sess_list[0],
            f"{decon_list[0]}_single_stats_REML+tlrc.HEAD",
        )
        check_file2 = os.path.join(
            deriv_dir,
            subj,
            sess_list[0],
            f"{decon_list[1]}_single_stats_REML+tlrc.HEAD",
        )
        if not os.path.exists(check_file1) or not os.path.exists(check_file2):
            run_list.append(subj)

    # make batch list
    if len(run_list) > 10:
        batch_list = run_list[0:10]
    else:
        batch_list = run_list

    for subj in batch_list:
        for sess in sess_list:

            h_out = os.path.join(out_dir, f"out_{subj}_{sess}.txt")
            h_err = os.path.join(out_dir, f"err_{subj}_{sess}.txt")

            # write decon_dict to json in subj dir
            with open(
                os.path.join(deriv_dir, subj, sess, "decon_dict.json"), "w"
            ) as outfile:
                json.dump(decon_dict, outfile)

            sbatch_job = f"""
                sbatch \
                    -J "GP3{subj.split("-")[1]}" \
                    -t 50:00:00 \
                    --mem=4000 \
                    --ntasks-per-node=1 \
                    -p IB_44C_512G  \
                    -o {h_out} -e {h_err} \
                    --account iacc_madlab \
                    --qos pq_madlab \
                    --wrap="module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
                    python {code_dir}/gp_step3_decon.py \
                        {subj} \
                        {sess} \
                        {decon_type} \
                        {deriv_dir}"
            """
            sbatch_submit = subprocess.Popen(
                sbatch_job, shell=True, stdout=subprocess.PIPE
            )
            job_id = sbatch_submit.communicate()[0]
            print(job_id.decode("utf-8"))
            time.sleep(1)


if __name__ == "__main__":
    main()

# %%
