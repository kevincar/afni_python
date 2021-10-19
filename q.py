"""Library of helper functions for the Grid Engine Compute system"""

import os
import shutil
import subprocess

from datetime import datetime
from typing import Optional, List


def qsub(
    cmd: str,
    project_name: str = os.environ["PROJECT_NAME"],
    job_name: Optional[str] = None,
    time_limit: Optional[str] = None,
    memory: Optional[int] = None,
    stdout_file: Optional[str] = None,
    stderr_file: Optional[str] = None,
):

    cur_time: datetime = datetime.now()
    time_str: str = datetime.strftime(cur_time, "%y%m%d%H%M%S")
    script_name: str = f"{job_name}{time_str}.sh"
    lines: List[str] = [
        "#!/bin/bash -l\n",
        f"#$ -P {project_name}",
        f"#$ -N {job_name}" if job_name else "",
        f"#$ -l h_rt={time_limit}" if time_limit else "",
        f"#$ -l mem_per_core={memory}G" if memory else "",
        f"#$ -o {stdout_file}" if stdout_file else "",
        f"#$ -e {stderr_file}" if stderr_file else "",
        "",
        "mb",
        f"{cmd}",
    ]

    script_content: str = "\n".join(lines)

    fh = open(script_name, "w")
    fh.write(script_content)
    fh.close()

    q_cmd: str = f"qsub -V {script_name}"

    q_proc: subprocess.CompletedProcess = subprocess.run(q_cmd.split(" "), text=True)
    print(q_proc.stdout)

    os.remove(script_name)

