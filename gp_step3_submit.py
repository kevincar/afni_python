"""
Notes:

Wrapper script for step3_decon.py.

Update paths in "set up" section.

decon_type can be dmBLOCK, GAM, 2GAM, or TENT
"""

import os
import time
import json
import fnmatch
import subprocess

from typing import List, Dict
from datetime import datetime
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import print_process_output

# set up
# code_dir = "/home/nmuncy/compute/learn_mvpa"
# work_dir = "/scratch/madlab/nate_vCAT"
# sess_list = ["ses-S1"]
# decon_type = "TENT"
# decon_dict = {
# "loc": ["tf_loc_face.txt", "tf_loc_num.txt", "tf_loc_scene.txt"],
# "Study": ["tf_Study_fix.txt", "tf_Study_con.txt", "tf_Study_fbl.txt"],
# }


def main():

    # Arguments for job submission
    parser: ArgumentParser = ArgumentParser(prog="gp_step3_submit")
    parser.add_argument("code_dir", help="Path to code repository")
    parser.add_argument("work_dir", help="Path to parent data directory")
    parser.add_argument("decon_type", help="Type of deconvolution to perform")
    parser.add_argument("decon_dict_path", help="Path to decon dict JSON")
    parser.add_argument("sess_list", nargs="+", help="List of sessions")

    # Gather arguments
    args: Namespace = parser.parse_args()
    code_dir: str = args.code_dir
    work_dir: str = args.work_dir
    decon_type: str = args.decon_type
    decon_dict_path: str = args.decon_dict_path
    sess_list: List[str] = args.sess_list

    # Load decon_dict
    decon_dict: Dict
    with open(decon_dict_path, "r") as fh:
        decon_dict = json.load(fh)

    # set up stdout/err capture
    deriv_dir: str = os.path.join(work_dir, "derivatives")
    current_time: datetime = datetime.now()
    timestamp: str = current_time.strftime("%H%M_%d-%m-%y")
    out_folder_name: str = f"TS3_{timestamp}"
    out_dir: str = os.path.join(deriv_dir, f"Slurm_out", out_folder_name)
    os.makedirs(out_dir)

    # submit job for each subj/sess/phase
    subj_list: List[str] = [
        x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")
    ]
    subj_list.sort()

    # determine which subjs to run
    run_list: List[str] = []
    for subj in subj_list:
        decon_list: List[str] = list(decon_dict.keys())
        check_file1: str = os.path.join(
            deriv_dir,
            subj,
            sess_list[0],
            f"{decon_list[0]}_single_stats_REML+tlrc.HEAD",
        )
        # check_file2: str = os.path.join(
        # deriv_dir,
        # subj,
        # sess_list[0],
        # f"{decon_list[1]}_single_stats_REML+tlrc.HEAD",
        # )
        if not os.path.exists(check_file1):
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

            # It's the same for all subjects...
            # write decon_dict to json in subj dir
            # with open(
            # os.path.join(deriv_dir, subj, sess, "decon_dict.json"), "w"
            # ) as outfile:
            # json.dump(decon_dict, outfile)

            script_name: str = f"{code_dir}/gp_step3_decon.py"
            sess = sess if sess != "" else "\"\""
            py_cmd: str = f"""
                python {script_name} \
                        {subj} \
                        {sess} \
                        {decon_type} \
                        {deriv_dir} \
                        {decon_dict_path}"""
            command: str = f"""
                module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
                {py_cmd}
                """

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
                    --wrap="{command}"
            """

            if os.environ.get("SLURM") is not None:
                sbatch_submit = subprocess.Popen(
                    sbatch_job, shell=True, stdout=subprocess.PIPE
                )
                job_id = sbatch_submit.communicate()[0]
                print(job_id.decode("utf-8"))
            else:
                proc = subprocess.Popen(py_cmd, shell=True, stdout=subprocess.PIPE)
                print_process_output(proc)
                proc.wait()
                if proc.returncode != 0:
                    raise Exception(f"Prof failed for subject {subj}")

            time.sleep(1)


if __name__ == "__main__":
    main()
