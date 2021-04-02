"""
Notes:

Wrapper script for ppi_job.py.

Update paths in "set up" section.

decon_type can be dmBLOCK, GAM, or 2GAM
"""

# %%
import os
from datetime import datetime
import fnmatch
import subprocess
import json
import time


# set up
code_dir = "/home/nmuncy/compute/RE_gPPI"
work_dir = "/scratch/madlab/nate_ppi"
sess_list = ["ses-S1"]
phase_list = ["Study"]
decon_type = "2GAM"
seed_dict = {"LHC": "-24 -12 -22"}
stim_dur = 2


def main():

    # set up stdout/err capture
    deriv_dir = os.path.join(work_dir, "derivatives")
    current_time = datetime.now()
    out_dir = os.path.join(
        deriv_dir, f'Slurm_out/PPI1_{current_time.strftime("%H%M_%d-%m-%y")}'
    )
    os.makedirs(out_dir)

    # submit job for each subj/sess/phase
    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    for subj in subj_list:
        for sess in sess_list:
            subj_dir = os.path.join(deriv_dir, subj, sess)
            for phase in phase_list:
                if not os.path.exists(
                    os.path.join(
                        subj_dir,
                        f"{phase}_{decon_type}_ppi_stats_REML+tlrc.HEAD",
                    )
                ):

                    # write json to avoid quotation issue
                    with open(os.path.join(subj_dir, "seed_dict.json"), "w") as outfile:
                        json.dump(seed_dict, outfile)

                    # Set stdout/err file
                    h_out = os.path.join(out_dir, f"out_{subj}_{sess}_{phase}.txt")
                    h_err = os.path.join(out_dir, f"err_{subj}_{sess}_{phase}.txt")

                    # submit command
                    sbatch_job = f"""
                        sbatch \
                        -J "PPI{subj.split("-")[1]}" -t 15:00:00 --mem=4000 --ntasks-per-node=1 \
                        -p IB_44C_512G  -o {h_out} -e {h_err} \
                        --account iacc_madlab --qos pq_madlab \
                        --wrap="/home/nmuncy/miniconda3/bin/python {code_dir}/ppi_step1_job.py \
                            {subj} {sess} {phase} {decon_type} {deriv_dir} {stim_dur}"
                    """

                    sbatch_submit = subprocess.Popen(
                        sbatch_job, shell=True, stdout=subprocess.PIPE
                    )
                    job_id = sbatch_submit.communicate()[0]
                    print(job_id)
                    time.sleep(1)


if __name__ == "__main__":
    main()

# %%
