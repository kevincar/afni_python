"""Run the steps to perform student's T-Test calculations on subject beta
coefficients

Example:
python ttest_run.py                                 \
    -d /projectnb/pheromone-fmri/data/derivatives   \
    -g /projectnb/pheromone-fmri/data/group         \
    -p PheromoneOlfaction                           \
    -e                                              \
    -l P1 ENI                                       \
    -r $DATA/atlas/ho_ofc
"""
import os

from glob import glob
from argparse import ArgumentParser, Namespace
from typing import List, Optional
import subprocess
from subprocess import CompletedProcess
from afni import afni_3dmask_tool


def generate_roi_mask(group_dir: str, roi_path: str) -> str:
    """Generate a ROI mask

    Parameters
    ----------
    group_dir : str
        Group analysis directory
    roi_path : str
        The file path to the ROI mask NiFti file

    Returns
    -------
    str
        The file path to the resulting mask
    """
    gmi_mask_filename: str = "Group_GM_intersect_mask+tlrc"
    gmi_mask_filepath: str = os.path.join(group_dir, gmi_mask_filename)
    inputs: str = " ".join([
        gmi_mask_filepath,
        roi_path
    ])

    out_filename: str = "ROI_Group_GM"
    afni_3dmask_tool(input=inputs, frac=1.0, prefix=out_filename, cwd=group_dir)
    out_filepath: str = os.path.join(group_dir, out_filename)
    return out_filepath


def generate_processing_script(
        derivatives_dir: str,
        group_dir: str,
        phase: str,
        labels: List[str],
        roi: Optional[str]) -> str:
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
    roi : Optional[str]
        Enables the t-test to be performed using only data specified from
        a ROI mask. The ROI mask is first multlied by the group gray matter
        intersection mask before applying it to the analysis


    Returns
    -------
    str
        The path to the generated script
    """
    # Determine the Mask to use
    mask_file_name: str = "Group_GM_intersect_mask+tlrc"
    mask_file_path: str = os.path.join(group_dir, mask_file_name)

    # Check if this is ROI
    out_dir_name: str = "ttest"
    if roi is not None:
        roi_file_name: str = os.path.basename(roi)
        roi_file_base: str = roi_file_name.split(".")[0]
        out_dir_name += roi_file_base
        mask_file_path = generate_roi_mask(group_dir, roi)

    out_dir: str = os.path.join(group_dir, out_dir_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Setup script arguments
    script_name: str = "gen_group_command.py"
    test_name: str = "3dttest++"
    output_file_name: str = "paired_ttest_cmd.csh"
    set_names: str = "-".join(labels)
    test_output_prefix: str = f"{phase}_{set_names}_stat_ttest"
    subj_dirs_wildcard: str = os.path.join(derivatives_dir, "*")
    subj_ds_wildcard: str = os.path.join(subj_dirs_wildcard, f"{phase}*_stats_REML+tlrc.HEAD")
    datasets: List[str] = glob(subj_ds_wildcard)
    betas: List[str] = [f"{label}#0_Coef" for label in labels]
    options: List[str] = [
        "-paired",
        f"-mask {mask_file_path}",
        "-CLUSTSIM",
        "-ETAC"
    ]

    # Compute shell style parameters
    labels_param: str = " ".join(labels)
    betas_param: str = " ".join(betas)
    datasets_param: str = " ".join(datasets)
    output_file_path: str = os.path.join(out_dir, output_file_name)
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
    parser.add_argument("-r", "--roi", help="ROI mask for exploratory small-volume analysis")

    args: Namespace = parser.parse_args()

    derivatives_dir: str = args.data
    group_dir: str = args.groupdir
    phase: str = args.phase
    go: bool = args.execute
    labels: List[str] = args.labels
    roi_path: Optional[str] = args.roi
    proc_script: str = generate_processing_script(derivatives_dir, group_dir,
                                                  phase, labels, roi_path)

    if go:
        proc: CompletedProcess = subprocess.run(f"tcsh {proc_script}", shell=True, stdout=subprocess.PIPE, check=True, cwd=group_dir)
        if proc.returncode != 0:
            raise Exception("script failed")
