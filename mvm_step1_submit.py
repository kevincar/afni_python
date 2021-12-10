import os
import sys
import time
import json
import shutil
import subprocess

from q import qsub
from typing import Optional
from datetime import datetime
from argparse import ArgumentParser, Namespace

# set up
# code_dir = "/home/nmuncy/compute/emu_power"
# parent_dir = "/scratch/madlab/emu_power"
# atlas_dir = "/home/data/madlab/atlases/emu_template"
# prior_dir = os.path.join(atlas_dir, "priors_FS")

# phase = "test"
# sess = "ses-S2"

# TODO update for multiple MVMs
# set beh_dict, key = WSVARS behavior, value = sub-brick
# beh_dict = {"targHT": 9, "targMS": 11, "lureCR": 5, "lureFA": 7}

# set up glt comparisons: key = 1*list[0] -1*list[1]
# glt_dict = {"Hit-Miss": ["targHT", "targMS"], "LCR-LFA": ["lureCR", "lureFA"]}


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-c", "--code", default=os.environ.get("CODE"))
    parser.add_argument("-p", "--parent", default=os.environ.get("DATA"))
    parser.add_argument("-a", "--atlas", default=os.path.join(os.environ.get("DATA", ""), "atlas", "vold2_mni"))
    parser.add_argument("-b", "--prior", default=os.path.join(os.environ.get("DATA", ""), "atlas", "vold2_mni", "priors_ACT"))
    parser.add_argument("-e", "--beh", default=os.path.join(os.environ.get("DATA", ""), "docs", "beh_dict.json"))
    parser.add_argument("-g", "--glt", default=os.path.join(os.environ.get("DATA", ""), "docs", "glt_dict.json"))
    parser.add_argument("-f", "--phase", required=True)
    parser.add_argument("-s", "--session", required=True)

    args: Namespace = parser.parse_args()
    code_dir: str = args.code
    parent_dir: str = args.parent
    atlas_dir: str = args.atlas
    prior_dir: str = args.prior
    beh_dict_path: str = args.beh
    glt_dict_path: str = args.glt
    phase: str = args.phase
    sess: str = args.session

    # set up stdout/err capture
    current_time: datetime = datetime.now()
    time_str: str = current_time.strftime("%H%M_%d-%m-%y")
    out_dir: str = os.path.join(
        parent_dir, "derivatives", "Slurm_out", f"MVM_{time_str}"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    h_out = os.path.join(out_dir, "out_mvm.txt")
    h_err = os.path.join(out_dir, "err_mvm.txt")

    # set output directory, write dicts to jsons
    group_dir = os.path.join(parent_dir, "analyses")
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)

    beh_dict_file_name: str = "beh_dict.json"
    beh_dict_dst_file_path: str = os.path.join(group_dir, beh_dict_file_name)
    if not os.path.exists(beh_dict_dst_file_path):
        shutil.copy(beh_dict_path, beh_dict_dst_file_path)

    glt_dict_file_name: str = "glt_dict.json"
    glt_dict_dst_file_path: str = os.path.join(group_dir, glt_dict_file_name)
    if not os.path.exists(glt_dict_dst_file_path):
        shutil.copy(glt_dict_path, glt_dict_dst_file_path)

    script_path: str = os.path.join(code_dir, "mvm_step1_test.py")
    sess = sess if sess != "" else "\"\""
    cmd: str = (
        f"{sys.executable} {script_path} "
        + f"-s {sess} -f {phase} -g {group_dir} "
        + f"-a {atlas_dir} -r {prior_dir} -p {parent_dir}"
    )
    job_name: str = "PMVM"
    time_limit: str = "40:00:00"
    memory: int = 4000
    project_name: str = os.environ.get("PROJECT_NAME", "")
    hpc: Optional[str] = os.environ.get("HPC")
    if hpc is not None:
        if hpc == "SLURM":
            sbatch_job = f"""
                sbatch \
                    -J "{job_name}" \
                    -t {time_limit} \
                    --mem={memory} \
                    --ntasks-per-node=1 \
                    -p IB_44C_512G  \
                    -o {h_out} -e {h_err} \
                    --account {project_name} \
                    --qos pq_madlab \
                    --wrap="{cmd}"
            """
            sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
            job_id = sbatch_submit.communicate()[0]
            print(job_id.decode("utf-8"))
        elif hpc == "QSUB":
            qsub(cmd, project_name=project_name, job_name=job_name,
                 time_limit=time_limit, memory=int(memory // 1000),
                 stdout_file=h_out, stderr_file=h_err
                 )
    time.sleep(1)


if __name__ == "__main__":
    main()
