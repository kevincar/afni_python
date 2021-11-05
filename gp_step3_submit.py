"""
Notes:

Wrapper script for step3_decon.py.

Update paths in "set up" section.

decon_type can be dmBLOCK, GAM, 2GAM, or TENT
"""

import os
import sys
import time
import json
import fnmatch
import subprocess

from typing import List, Dict, Optional
from datetime import datetime
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import print_process_output
from q import qsub

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
    parser.add_argument("-c", "--code", help="Path to code repository")
    parser.add_argument("-p", "--parent", help="Path to parent data directory")
    parser.add_argument("-t", "--type", help="Type of deconvolution to perform")
    parser.add_argument("-d", "--decondict", help="Path to decon dict JSON")
    parser.add_argument(
        "-n", "--batchnum", type=int,
        help="Number of subjects to include in a batch", default=10
    )
    parser.add_argument("sess_list", nargs="+", help="List of sessions")

    # Gather arguments
    args: Namespace = parser.parse_args()
    code_dir: str = args.code
    work_dir: str = args.parent
    decon_type: str = args.type
    decon_dict_path: str = args.decondict
    batch_size: int = args.batchnum
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
    if len(run_list) > batch_size:
        batch_list = run_list[0:batch_size]
    else:
        batch_list = run_list

    for subj in batch_list:
        for sess in sess_list:

            subj_num: int = int(subj.split("-")[1])
            job_name: str = f"GP3{subj_num}"
            time_limit: str = "50:00:00"
            memory: int = 4000
            h_out = os.path.join(out_dir, f"out_{subj}_{sess}.txt")
            h_err = os.path.join(out_dir, f"err_{subj}_{sess}.txt")
            code_file_path: str = os.path.join(code_dir, "gp_step3_decon.py")
            sess = sess if sess != "" else "\"\""
            h_cmd: str = f"{sys.executable} {code_file_path} {subj} {sess} {decon_type} {deriv_dir} {decon_dict_path}"

            # It's the same for all subjects...
            # write decon_dict to json in subj dir
            # with open(
            # os.path.join(deriv_dir, subj, sess, "decon_dict.json"), "w"
            # ) as outfile:
            # json.dump(decon_dict, outfile)

            command: str = f"""
                module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
                {h_cmd}
                """

            hpc: Optional[str] = os.environ.get("HPC")
            if hpc is not None:
                if hpc == "SLURM":
                    sbatch_job = f"""
                        sbatch \
                            -J "{job_name} "\
                            -t {time_limit} \
                            --mem={memory} \
                            --ntasks-per-node=1 \
                            -p IB_44C_512G  \
                            -o {h_out} -e {h_err} \
                            --account iacc_madlab \
                            --qos pq_madlab \
                            --wrap="{command}"
                    """
                    sbatch_submit = subprocess.Popen(
                        sbatch_job, shell=True, stdout=subprocess.PIPE
                    )
                    job_id = sbatch_submit.communicate()[0]
                    print(job_id.decode("utf-8"))
                elif hpc == "QSUB":
                    project_name: str = os.environ.get("PROJECT_NAME", "")
                    qsub(h_cmd, project_name=project_name, job_name=job_name,
                         time_limit=time_limit, memory=int(memory // 1000),
                         stdout_file=h_out, stderr_file=h_err
                         )
            else:
                proc = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                print_process_output(proc)
                proc.wait()
                if proc.returncode != 0:
                    raise Exception(f"Prof failed for subject {subj}")

            time.sleep(1)


if __name__ == "__main__":
    main()
