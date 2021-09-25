"""
Notes:

Wrapper script for step1_preproc.py.

Usage - update "set up" section.

phase_list = list of phases gathered within a single session.
    For example, if a study and then a test phase were both scanned
    during the same session, then phase_list = ["study", "test"]
"""
# %%
import os
import json
import time
import fnmatch
import subprocess

from typing import Dict, List
from datetime import datetime
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import print_process_output

# set up
# code_dir = "/home/nmuncy/compute/learn_mvpa"
# parent_dir = "/scratch/madlab/nate_vCAT"
# sess_dict = {"ses-S1": ["loc", "Study"]}
# blip_toggle = 0  # 1 = on, 0 = off


# %%
def main():

    # Argument Parser
    parser: ArgumentParser = ArgumentParser(prog="gp_step1_submit")
    parser.add_argument("-c", "--code", required=True)
    parser.add_argument("-p", "--parent", required=True)
    parser.add_argument("-s", "--sessiondatafile", required=True)
    parser.add_argument("-b", "--blip", type=int, default=0)
    args: Namespace = parser.parse_args()

    # Gather values
    code_dir: str = args.code
    parent_dir: str = args.parent
    session_file_path: str = args.sessiondatafile
    blip_toggle: int = args.blip
    sess_dict: Dict
    with open(session_file_path, "r") as fh:
        sess_dict = json.load(fh)

    # set up stdout/err capture
    work_dir: str = os.path.join(parent_dir, "derivatives")
    data_dir: str = os.path.join(parent_dir, "dset")
    slurm_dir: str = os.path.join(work_dir, "Slurm_out")
    current_time: datetime = datetime.now()
    time_str: str = current_time.strftime("%y-%m-%d_%H%M")
    out_dir: str = os.path.join(slurm_dir, f"TS1_{time_str}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # submit job for each subj/sess/phase
    subj_list: List = [x for x in os.listdir(data_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    # determine which subjs to run
    run_list: List = []
    sess_keys: List = list(sess_dict.keys())
    h_sess: str = sess_keys[0]
    for subj in subj_list:
        subj_dir: str = os.path.join(work_dir, subj)
        sess_dir: str = os.path.join(subj_dir, h_sess) if h_sess != "" else subj_dir

        # Check file
        check_file_name: str = f"run-1_{sess_dict[h_sess][0]}_scale+tlrc.HEAD"
        check_file_path: str = os.path.join(sess_dir, check_file_name)
        if not os.path.exists(check_file_path):
            run_list.append(subj)

    # make batch list
    batch_list: List
    if len(run_list) > 10:
        batch_list = run_list[0:10]
    else:
        batch_list = run_list

    for subj in batch_list:
        for sess in sess_dict:

            h_out: str = os.path.join(out_dir, f"out_{subj}_{sess}.txt")
            h_err: str = os.path.join(out_dir, f"err_{subj}_{sess}.txt")
            code_file_path: str = os.path.join(code_dir, "gp_step1_preproc.py")
            phases_str: str = " ".join(sess_dict[sess])
            sess = sess if sess != "" else '""'
            h_cmd: str = f"""
            python {code_file_path} \
                    {subj} \
                    {sess} \
                    {parent_dir} \
                    {blip_toggle} \
                    {phases_str}
            """

            if os.environ.get("SLURM") is not None:
                sbatch_job = f"""
                    sbatch \
                        -J "GP1{subj.split("-")[1]}" \
                        -t 10:00:00 \
                        --mem=4000 \
                        --ntasks-per-node=1 \
                        -p IB_44C_512G  \
                        -o {h_out} -e {h_err} \
                        --account iacc_madlab \
                        --qos pq_madlab \
                        --wrap="module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
                        python {code_dir}/gp_step1_preproc.py \
                            {subj} \
                            {sess} \
                            {parent_dir} \
                            {blip_toggle} \
                            {' '.join(sess_dict[sess])}"
                """
                sbatch_submit = subprocess.Popen(
                    sbatch_job, shell=True, stdout=subprocess.PIPE
                )
                job_id = sbatch_submit.communicate()[0]
                print(job_id.decode("utf-8"))
                time.sleep(1)
            else:
                proc: subprocess.Popen = subprocess.Popen(
                    h_cmd, shell=True, stdout=subprocess.PIPE
                )
                print_process_output(proc)
                proc.wait()
                if proc.returncode != 0:
                    raise Exception(f"Failed to run {subj}")


if __name__ == "__main__":
    main()
