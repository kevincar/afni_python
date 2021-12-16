"""Run the steps to perform student's T-Test calculations on subject beta
coefficients

Example:
python ttest_run.py                                 \
    -d /projectnb/pheromone-fmri/data/derivatives   \
    -g /projectnb/pheromone-fmri/data/group         \
    -p PheromoneOlfaction -e
"""
import os

from glob import glob
from argparse import ArgumentParser, Namespace
from typing import List
import subprocess
from subprocess import CompletedProcess


def generate_processing_script(
        derivatives_dir: str,
        group_dir: str,
        phase: str,
        labels: List[str]) -> str:
    """
    Generate the script to perform 3dTest

    Parameters
    ----------
    derivatives_dir : str
        The path to the directory containing subject folders with preprocessed
        volumes
    group_dir : str
        The path to the group directory where group analysis will be performed
    phase : str
        The name of the current phase for subjects to compute
    labels : List[str]
        A list of labels to compare. Really should only have a length of two

    Returns
    -------
    str
        The path to the generated script
    """
    script_name: str = "gen_group_command.py"
    test_name: str = "3dttest++"
    output_file_name: str = "paired_ttest_cmd.csh"
    set_names: str = "-".join(labels)
    test_output_prefix: str = f"{phase}_{set_names}_stat_ttest"
    subj_dirs_wildcard: str = os.path.join(derivatives_dir, "*")
    subj_ds_wildcard: str = os.path.join(subj_dirs_wildcard, f"{phase}*_stats_REML+tlrc.HEAD")
    datasets: List[str] = glob(subj_ds_wildcard)
    betas: List[str] = [f"{label}#0_Coef" for label in labels]
    mask_file_name: str = "Group_GM_intersect_mask+tlrc"
    mask_file_path: str = os.path.join(group_dir, mask_file_name)
    options: List[str] = [
        "-paired",
        f"-mask {mask_file_path}",
        "-CLUSTSIM"
    ]

    labels_param: str = " ".join(labels)
    betas_param: str = " ".join(betas)
    datasets_param: str = " ".join(datasets)
    output_file_path: str = os.path.join(group_dir, output_file_name)
    options_param: str = " ".join(options)
    params: List[str] = [
        script_name,
        f"-command {test_name}",
        f"-write_script {output_file_path}",
        f"-prefix {test_output_prefix}",
        f"-dsets {datasets_param}",
        f"-set_labels {labels_param}",
        f"-subs_betas {betas_param}",
        f"-options {options_param}"
    ]
    param_string: str = " ".join(params)

    proc: CompletedProcess = subprocess.run(param_string, shell=True, stdout=subprocess.PIPE, check=True)
    if proc.returncode != 0:
        raise Exception(f"{script_name} failed")

    return output_file_path


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(description="Generate 3dTtest processing script")
    parser.add_argument(
        "-d", "--data", required=True,
        help="The path to the derivatives directory that holds individual subject directories")
    parser.add_argument("-g", "--groupdir", required=True, help="Path to the group analysis directory")
    parser.add_argument("-p", "--phase", required=True, help="Phase name for the tests")
    parser.add_argument("-e", "--execute", default=False, action="store_const", const=True)
    parser.add_argument("-l", "--labels", required=True, nargs=2, help="Label pairs to compaire")

    args: Namespace = parser.parse_args()

    derivatives_dir: str = args.data
    group_dir: str = args.groupdir
    phase: str = args.phase
    go: str = args.execute
    labels: List[str] = args.labels
    proc_script: str = generate_processing_script(derivatives_dir, group_dir, phase, labels)

    if go:
        proc: CompletedProcess = subprocess.run(f"tcsh {proc_script}", shell=True, stdout=subprocess.PIPE, check=True, cwd=group_dir)
        if proc.returncode != 0:
            raise Exception("script failed")
