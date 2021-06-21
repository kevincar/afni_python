import os
import time
from datetime import datetime
import subprocess
import json

# set up
code_dir = "/home/nmuncy/compute/emu_power"
parent_dir = "/scratch/madlab/emu_power"
atlas_dir = "/home/data/madlab/atlases/emu_template"
prior_dir = os.path.join(atlas_dir, "priors_FS")

phase = "test"
sess = "ses-S2"

# TODO update for multiple MVMs
# set beh_dict, key = WSVARS behavior, value = sub-brick
beh_dict = {"targHT": 9, "targMS": 11, "lureCR": 5, "lureFA": 7}

# set up glt comparisons: key = 1*list[0] -1*list[1]
glt_dict = {"Hit-Miss": ["targHT", "targMS"], "LCR-LFA": ["lureCR", "lureFA"]}


def main():

    # set up stdout/err capture
    current_time = datetime.now()
    out_dir = os.path.join(
        parent_dir,
        f'derivatives/Slurm_out/MVM_{current_time.strftime("%H%M_%d-%m-%y")}',
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    h_out = os.path.join(out_dir, "out_mvm.txt")
    h_err = os.path.join(out_dir, "err_mvm.txt")

    # set output directory, write dicts to jsons
    group_dir = os.path.join(parent_dir, "analyses")
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)

    with open(os.path.join(group_dir, "beh_dict.json"), "w") as outfile:
        json.dump(beh_dict, outfile)
    with open(os.path.join(group_dir, "glt_dict.json"), "w") as outfile:
        json.dump(glt_dict, outfile)

    sbatch_job = f"""
        sbatch \
            -J "PMVM" \
            -t 40:00:00 \
            --mem=4000 \
            --ntasks-per-node=1 \
            -p IB_44C_512G  \
            -o {h_out} -e {h_err} \
            --account iacc_madlab \
            --qos pq_madlab \
            --wrap="~/miniconda3/bin/python {code_dir}/mvm_step1_test.py \
                {sess} \
                {phase} \
                {group_dir} \
                {atlas_dir} \
                {prior_dir} \
                {parent_dir}"
    """
    sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
    job_id = sbatch_submit.communicate()[0]
    print(job_id.decode("utf-8"))
    time.sleep(1)


if __name__ == "__main__":
    main()
