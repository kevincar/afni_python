"""Example
python ttest_submit.py      \
    -c $CODE                \
    -d $DATA/derivatives    \
    -g $DATA/group          \
    -p PheromoneOlfaction
"""
import os
import sys
import subprocess

import datetime as dt

from q import qsub
from argparse import ArgumentParser, Namespace
from typing import Optional, List


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(description="Submit a job to the cluster calculate a 3d t-test")
    parser.add_argument("-c", "--code", required=True, help="Path to the code directory")
    parser.add_argument("-d", "--data", required=True, help="Path to data derivatives directory")
    parser.add_argument("-g", "--group", required=True, help="Path to group analysis directory")
    parser.add_argument("-p", "--phase", required=True, help="Name of phase")

    args: Namespace = parser.parse_args()
    code_dir: str = args.code
    data_dir: str = args.data
    group_dir: str = args.group
    phase: str = args.phase

    script_name = "ttest_run.py"
    script_path = os.path.join(code_dir, script_name)
    command_parameters: List[str] = [
        f"{sys.executable} {script_path}",
        f"-d {data_dir}",
        f"-g {group_dir}",
        f"-p {phase}",
        "-e"
    ]
    cmd: str = " ".join(command_parameters)

    project_name: str = os.environ["PROJECT_NAME"]
    hpc: Optional[str] = os.environ.get("HPC")
    job_name: str = "TTEST"
    time_limit: str = "40:00:00"
    memory: int = 4000
    n_cores: int = 10

    out_dir: str = os.path.join(data_dir, "Slurm_out")
    current_time: dt.datetime = dt.datetime.now()
    time_str: str = current_time.strftime("%y-%m-%d_%H%M")
    out_dir_name: str = f"{job_name}_{time_str}"
    out_dir_path: str = os.path.join(out_dir, out_dir_name)
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    out_file_path: str = os.path.join(out_dir_path, "out_ttest.txt")
    err_file_path: str = os.path.join(out_dir_path, "err_ttest.txt")

    if hpc is not None:
        if hpc == "SLURM":
            sbatch_job = f"""
                sbatch \
                    -J "{job_name}" \
                    -t {time_limit} \
                    --mem={memory} \
                    --ntasks-per-node=1 \
                    -p IB_44C_512G  \
                    -o {out_file_path} -e {err_file_path} \
                    --account {project_name} \
                    --qos pq_madlab \
                    --wrap="{cmd}"
            """
            sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
            job_id = sbatch_submit.communicate()[0]
        elif hpc == "QSUB":
            qsub(cmd, project_name=project_name, job_name=job_name, cores=10,
                 time_limit=time_limit, memory=int(memory // 1000),
                 stdout_file=out_file_path, stderr_file=err_file_path
                 )
