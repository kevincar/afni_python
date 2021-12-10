# %%
import os
import json
import fnmatch
import shutil
import subprocess
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import func_sbatch
from afni import (
    c3d,
    afni_3dcopy,
    afni_3dmean,
    afni_3dmask_tool,
    afni_3drefit,
    afni_3dresample
)


def gather_num_censored_trs(
        subj_list: List[str],
        deriv_dir: str,
        sess: str,
        phase: str) -> pd.DataFrame:
    """
    Gather the number of censored TRs and their proportions. This is used for
    determining subjects that are outliers.

    Paramount
    ---------
    subj_list : List[str]
        List of subject identifiers
    deriv_dir : str
        Data directory
    sess : str
        Session for analysis
    phase : str
        Phase for alaysis

    Returns
    -------
    pd.DataFrame
        Information on the subject, their censored TRs and fractions
    """
    info: pd.DataFrame = pd.DataFrame(columns=["Subject", "N Centers", "Proportion"])

    for subj in subj_list:
        summary_file_name: str = f"out_summary_{phase}.txt"
        summary_file_path: str = os.path.join(deriv_dir, subj, sess, summary_file_name)
        summary_data: pd.DataFrame = pd.read_csv(summary_file_path, sep=":", names=["Key", "Value"])
        data_values: np.ndarray = summary_data.values
        summary_dict: Dict = {row[0].strip(): row[1].strip() for row in data_values}

        centers: int = int(summary_dict["TRs censored"])
        fraction: float = float(summary_dict["censor fraction"])
        info = info.append({"Subject": subj, "N Centers": centers, "Proportion": fraction}, ignore_index=True)
        info = info.astype({"Subject": str, "N Centers": int, "Proportion": float})

    return info


def filter_outlier_subjects(
        subj_list: List[str],
        deriv_dir: str,
        sess: str,
        phase: str) -> List[str]:
    """
    Given a list of subjects to consider for analysis, remove those whose data
    demonstrate movement was to great for adequate analysis

    Parameters
    ----------
    subj_list : List [str]
        An list of subjects
    deriv_dir : str
        The path to the folder containing subject directories
    sess : str
        The name of the session to consider
    phase : str
        The name of the phase to consider

    Returns
    -------
    List[str]
        The filtered list of subjects
    """
    reference_subject: str = next(iter(subj_list))
    reference_directory: str = os.path.join(deriv_dir, reference_subject)
    decon_list: List[str] = [
        f[:f.find("_")] for f in os.listdir(reference_directory)
        if "stats_REML+tlrc.HEAD" in f
    ]

    result: List[str] = subj_list
    for decon_name in decon_list:
        subject_data_info: pd.DataFrame = gather_num_censored_trs(subj_list, deriv_dir, sess, phase)
        stats: pd.DataFrame = subject_data_info.describe()
        lower: pd.Series = stats.loc["25%"]
        med: pd.Series = stats.loc["50%"]
        upper: pd.Series = stats.loc["75%"]
        IQR: pd.Series = upper - lower
        IQR15: pd.Series = IQR * 1.5
        max_limit: pd.Series = med + IQR15
        center_max: float = max_limit["N Centers"]
        outlier_data: pd.DataFrame = subject_data_info[subject_data_info["N Centers"] > center_max]
        outlier_subjects_series: pd.Series = outlier_data["Subject"]
        outlier_subjects: List[str] = outlier_subjects_series.values.tolist()
        result = [s for s in result if s not in outlier_subjects]

    return result


def create_group_intersection_mask(
        threshold: float,
        group_dir: str,
        subj_list: List[str],
        deriv_dir: str,
        sess: str,
        phase: str):
    """
    Generate a group intersection mask

    Parameters
    ----------
    threshold : float
        The mask threshold
    group_dir : str
        The directory used for holding group analysis files
    subj_list : List[str]
        A list of subjects to include in the intersection mask
    deriv_dir : str
        Path to the working data directory
    sess : str
        The name of the current session being analyzed
    phase : str
        The name of the current phase being analyzed
    """
    out_file_name: str = "Group_intersect_mask.nii.gz"
    out_file_path: str = os.path.join(group_dir, out_file_name)

    if os.path.exists(out_file_path):
        return

    included_subj_list: List[str] = filter_outlier_subjects(subj_list, deriv_dir, sess, phase)
    mask_files: List[str] = [
        os.path.join(deriv_dir, subj, sess, "mask_epi_anat+tlrc")
        for subj in included_subj_list
        if os.path.exists(os.path.join(deriv_dir, subj, sess, "mask_epi_anat+tlrc.HEAD"))
    ]
    print(included_subj_list)

    prefix: str = "Group_intersect_mean.nii.gz"
    afni_3dmean(*mask_files, prefix=prefix, cwd=group_dir)
    afni_3dmask_tool(input=mask_files, frac=threshold, prefix=out_file_name, cwd=group_dir)


def create_gray_matter_input(prior_dir: str) -> str:
    """
    Generate a gray matter input

    Parameters
    ----------
    prior_dir : str
        Path to the location of prior template scans

    Returns
    -------
    str
        The path to the generated gray matter file
    """
    gm_target_file_path: str = os.path.join(prior_dir, "GM.nii.gz")
    if os.path.exists(gm_target_file_path):
        return gm_target_file_path

    prior_file_nums: List[int] = [2, 4]
    prior_file_names: List[str] = [f"Prior{n}.nii.gz" for n in prior_file_nums]
    prior_file_paths: List[str] = [os.path.join(prior_dir, fn) for fn in prior_file_names]
    c3d(prior_file_paths[0], prior_file_paths[1], add="", o=gm_target_file_path, cwd=prior_dir)
    return gm_target_file_path


def create_gm_intersection_mask(group_dir: str, ref_file: str, gm_file: str):
    """
    Combines the intersection and gray matter input

    Parameters
    ----------
    group_dir : str
        Location for group data analysis files
    ref_file : str
        A reference file to use for generating the mask
    gm_file : str
        A path to the gray matter file
    """
    out_dset_name: str = "Group_GM_intersect_mask+tlrc"
    out_file_name: str = f"{out_dset_name}.HEAD"
    out_file_path: str = os.path.join(group_dir, out_file_name)

    if os.path.exists(out_file_path):
        return

    prefix: str = "tmp_GM_mask.nii.gz"

    if not os.path.exists(os.path.join(group_dir, prefix)):
        afni_3dresample(master=ref_file, rmode="NN", input=gm_file, prefix=prefix, cwd=group_dir)
    c3d(prefix, "Group_intersect_mask.nii.gz", multiply="", o="tmp_GM_intersect_prob_mask.nii.gz", cwd=group_dir)
    c3d("tmp_GM_intersect_prob_mask.nii.gz", thresh="0.1 1 1 0", o="tmp_GM_intersect_mask.nii.gz", cwd=group_dir)
    afni_3dcopy("tmp_GM_intersect_mask.nii.gz", out_dset_name, cwd=group_dir)
    afni_3drefit(out_dset_name, space="MNI", cwd=group_dir)

    # cmds: List[str] = [
    # f"cd {group_dir}",
    # f"3dresample -master {ref_file} -rmode NN -input {gm_file} -prefix {prefix}",
    # f"c3d {prefix} Group_intersect_mask.nii.gz -multiply -o tmp_GM_intersect_prob_mask.nii.gz",
    # "c3d tmp_GM_intersect_prob_mask.nii.gz -thresh 0.1 1 1 0 -o tmp_GM_intersect_mask.nii.gz",
    # "3dcopy tmp_GM_intersect_mask.nii.gz Group_GM_intersect_mask+tlrc",
    # "3drefit -space MNI Group_GM_intersect_mask+tlrc"
    # ]
    # cmd: str = "\n".join(cmds)
    # gm_mask_proc: subprocess.Popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # gm_mask_proc.wait()
    # if gm_mask_proc.returncode != 0:
    # raise Exception("GM Intersection Mask failed")


def func_mask(subj_list, deriv_dir, sess, phase, atlas_dir, prior_dir, group_dir):

    # set ref file for resampling
    ref_file = os.path.join(deriv_dir, subj_list[1], sess, f"run-1_{phase}_scale+tlrc")

    create_group_intersection_mask(0.3, group_dir, subj_list, deriv_dir, sess, phase)
    gray_matter_file_path: str = create_gray_matter_input(prior_dir)
    print(f"GMFILE: {gray_matter_file_path}")
    create_gm_intersection_mask(group_dir, ref_file, gray_matter_file_path)


def build_data_table(
        subj_list: List[str],
        deriv_dir: str,
        beh_dict: Dict,
        phase: str,
        sess: str,
        bs_factors: List[str] = []) -> pd.DataFrame:
    """
    Given a list of subjects, a behavior dictionary, and an optional
    between-subjects factors list, return a data frame that will be passed to
    the 3dMVM command

    Parameters
    ----------
    subj_list : List[str]
        A list of subject identifiers
    deriv_dir : str
        Path to the subject directories
    beh_dict : Dict
        A dictionary where the keys are the behaviors, experimental conditions,
        or within-subject factors, and the values are the index for the
        functional data file
    phase : str
        The current phase
    sess : str
        The current session
    bs_factors : List[str]
        Between subject factors. This list should be the same length as
        `subj_list` if provided

    Returns
    -------
    pd.DataFrame
        A pandas dataframe of the data table needed by 3dMVM
    """
    has_bs_factors: bool = len(bs_factors) > 0
    headers: List[str] = ["Subj", "WSVARS", "InputFile"]
    if has_bs_factors:
        headers.insert(1, "BSVARS")

    data_table: pd.DataFrame = pd.DataFrame(columns=headers)
    for idx, subject_id in enumerate(subj_list):
        session_path: str = os.path.join(deriv_dir, subject_id, sess)
        for behavior, behavior_timepoint in beh_dict.items():
            input_file_name: str = f"{phase}_signle_stats_REML+tlrc[{behavior_timepoint}]"
            input_file_path: str = os.path.join(session_path, input_file_name)
            data_row: List[str] = [subject_id, behavior, input_file_path]

            if has_bs_factors:
                subject_bs_factor: str = bs_factors[idx]
                data_row.insert(1, subject_bs_factor)

            data_row_table: pd.DataFrame = pd.DataFrame([data_row], columns=data_table.columns)
            data_table = data_table.append(data_row_table)

    return data_table


def func_mvm(
        beh_dict: Dict,
        glt_dict: Dict,
        subj_list: List[str],
        sess: str,
        phase: str,
        group_dir: str,
        deriv_dir: str,
        bs_factors: List[str] = []):
    """
    Run a 3d Multivariate Analysis approach on the data based on within-subject
    and between-subject groups

    Parameters
    ----------
    beh_dict : Dict
        A dictionary where the keys indicate the behavioral or experimental
        treatment and the value is the time value in the analysis file
    glt_dict : Dict
        ?
    subj_list : List[str]
        A list of subject identifiers
    sess : str
        The session name
    phase : str
        The phase name
    group_dir : str
        The path to the group analysis directory
    deriv_dir : str
        The path to the derivatives directory that holds all subject
        directories
    bs_factors : List[str]
        An optional list of between-subject factors. This list should be the
        same length as `subj_list`
    """
    has_bs_factors: bool = len(bs_factors) < 1

    data_table: pd.DataFrame = build_data_table(subj_list, deriv_dir, beh_dict, sess, phase, bs_factors)
    data_table_arguments_list: List[str] = data_table.values.flatten().tolist()
    data_table_arguments: str = " ".join(data_table_arguments_list)
    data_table_headers: str = " ".join(data_table.columns.tolist())

    # General Linear T-Tests (GLT)
    glt_list = []
    for idx, test_name in enumerate(glt_dict):
        count: int = idx + 1
        glt_label_arg: str = f"-gltLabel {count} {test_name}"

        test_conditions: List = glt_dict[test_name]

        bs_vars: List[str] = []
        ws_vars: List[str] = test_conditions
        ws0: str = ws_vars[0]
        ws1: str = ws_vars[1]
        glt_code_ws: str = f"WSVARS: 1*{ws0} -1*{ws1}"
        glt_code_cmd: str = r"'{glt_code_ws}'"

        if has_bs_factors:
            bs_vars = test_conditions[0]
            bs0: str = bs_vars[0]
            bs1: str = bs_vars[1]
            glt_code_bs: str = f"BSVARS: 1*{bs0} -1*{bs1}"
            ws_vars = test_conditions[1]
            ws0 = ws_vars[0]
            if len(ws_vars) > 1:
                ws1 = ws_vars[1]
                glt_code_ws = f"WSVARS: 1*{ws0}"
            else:
                glt_code_ws = f"WSVARS: 1*{ws0} -1*{ws1}"
            glt_code_cmd = f"'{glt_code_bs} {glt_code_ws}'"
            glt_code_cmd

        glt_code_arg: str = "-gltCode {count} {glt_code_cmd}"
        glt_cmd: str = f"{glt_label_arg} {glt_code_arg}"
        glt_list.append(glt_cmd)

    n_glt_arguments: int = len(glt_list)
    glt_arguments: str = " ".join(glt_list)

    bsvars_arg: str = "'BSVARS'" if has_bs_factors else "1"

    h_cmd = f"""
        cd {group_dir}

        3dMVM \
            -prefix MVM \
            -jobs 10 \
            -mask Group_GM_intersect_mask+tlrc \
            -bsVars {bsvars_arg} \
            -wsVars 'WSVARS' \
            -num_glt {n_glt_arguments} \
            {glt_arguments} \
            -dataTable \
            {data_table_headers} \
            {data_table_arguments}
    """
    hpc: Optional[str] = os.environ.get("HPC")
    if hpc is not None:
        if hpc == "SLURM":
            func_sbatch(h_cmd, 2, 6, 10, "cMVM", group_dir)
        elif hpc == "QSUM":
            proc: subprocess.Popen = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            proc.wait()
            if proc.returncode != 0:
                raise Exception("Failed to run 3dMVM")

    print("MVM Done")


def func_acf(subj, subj_file, group_dir, acf_file):
    h_cmd = f"""
        cd {group_dir}

        3dFWHMx \
            -mask Group_GM_intersect_mask+tlrc \
            -input {subj_file} \
            -acf >> {acf_file}
    """
    hpc: Optional[str] = os.environ.get("HPC")
    if hpc is not None:
        if hpc == "SLURM":
            func_sbatch(h_cmd, 2, 4, 1, f"a{subj.split('-')[-1]}", group_dir)
        elif hpc == "QSUB":
            acf_proc: subprocess.Popen = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            acf_proc.wait()
            if acf_proc.returncode != 0:
                raise Exception("acf proc failed")


def func_clustSim(group_dir, acf_file, mc_file):

    df_acf = pd.read_csv(acf_file, sep=" ", header=None)
    df_acf = df_acf.dropna(axis=1)
    df_acf = df_acf.loc[(df_acf != 0).any(axis=1)]
    mean_list = list(df_acf.mean())

    h_cmd = f"""
        cd {group_dir}

        3dClustSim \
            -mask Group_GM_intersect_mask+tlrc \
            -LOTS \
            -iter 10000 \
            -acf {mean_list[0]} {mean_list[1]} {mean_list[2]} \
            > {mc_file}
    """
    hpc: Optional[str] = os.environ.get("HPC")
    if hpc is not None:
        if hpc == "SLURM":
            func_sbatch(h_cmd, 6, 4, 10, "mc", group_dir)
        elif hpc == "QSUM":
            cluster_proc: subprocess.Popen = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            cluster_proc.wait()
            if cluster_proc.returncode != 0:
                raise Exception("cluster proc failed")

    print("clusterStim Done")


def func_argparser() -> ArgumentParser:
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("-s", "--sess", required=True, help="Session")
    parser.add_argument("-f", "--phase", required=True, help="Phase")
    parser.add_argument("-g", "--group-path", required=True, help="Output Directory")
    parser.add_argument("-a", "--atlas-path", required=True, help="Location of atlas")
    parser.add_argument("-r", "--prior-path", required=True, help="Location of atlas priors")
    parser.add_argument("-p", "--parent-path", required=True, help="Location of Project Directory")
    parser.add_argument("-b", "--bs-factors-file", help="Whether to use Within Subject Factors")

    return parser


# %%
def main():

    # TODO update for multiple MVMs

    # get args
    args: Namespace = func_argparser().parse_args()
    sess: str = args.sess
    phase: str = args.phase
    group_dir: str = args.group_path
    atlas_dir: str = args.atlas_path
    prior_dir: str = args.prior_path
    parent_dir: str = args.parent_path
    bs_factors_file_path: Optional[str] = args.bs_factors_file

    # get/make paths, dicts
    deriv_dir: str = os.path.join(parent_dir, "derivatives")
    subj_list: List[str] = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]

    beh_dict_file_name: str = "beh_dict.json"
    glt_dict_file_name: str = "glt_dict.json"

    docs_dir: str = os.path.join(parent_dir, "docs")
    beh_dict_path: str = os.path.join(docs_dir, beh_dict_file_name)
    glt_dict_path: str = os.path.join(docs_dir, glt_dict_file_name)

    beh_dict_file_path: str = os.path.join(group_dir, beh_dict_file_name)
    if not os.path.exists(beh_dict_file_path):
        shutil.copy(beh_dict_path, beh_dict_file_path)

    glt_dict_file_path: str = os.path.join(group_dir, glt_dict_file_name)
    if not os.path.exists(glt_dict_file_path):
        shutil.copy(glt_dict_path, glt_dict_file_path)

    with open(beh_dict_file_path) as json_file:
        beh_dict = json.load(json_file)

    with open(glt_dict_file_path) as json_file:
        glt_dict = json.load(json_file)

    bs_factors: List[str] = []
    if bs_factors_file_path:
        bs_factors_df: pd.DataFrame = pd.read_csv(bs_factors_file_path, headers=None)
        bs_factors = bs_factors_df.values.flatten().tolist()

    """ make group gray matter intreset mask """
    mask_output_file: str = "Group_GM_intersect_mask+tlrc.HEAD"
    mask_output_file_path: str = os.path.join(group_dir, mask_output_file)
    if not os.path.exists(mask_output_file_path):
        func_mask(subj_list, deriv_dir, sess, phase, atlas_dir, prior_dir, group_dir)

    print("Mask done")

    """ run MVM """
    mvm_output_file_name: str = "MVM+tlrc.HEAD"
    mvm_output_file_path: str = os.path.join(group_dir, mvm_output_file_name)
    if not os.path.exists(mvm_output_file_path):
        func_mvm(beh_dict, glt_dict, subj_list, sess, phase, group_dir, deriv_dir, bs_factors)

    print("MVM Done")

    """ get subj acf estimates """
    # define, start file
    acf_file = os.path.join(group_dir, "ACF_subj_all.txt")
    # if not os.path.exists(acf_file):
    open(acf_file, "w").close()

    # if file is empty, run func_acf for e/subj
    acf_size = os.path.getsize(acf_file)
    if acf_size == 0:
        print("NICE")
        for subj in subj_list:
            subj_file = os.path.join(
                deriv_dir, subj, sess, f"{phase}_single_errts_REML+tlrc"
            )
            func_acf(subj, subj_file, group_dir, acf_file)

    print("ACF done")

    """ do clust simulations """
    mc_file = os.path.join(group_dir, "MC_thresholds.txt")
    if not os.path.exists(mc_file):
        open(mc_file, "a").close()

    mc_size = os.path.getsize(mc_file)
    if mc_size == 0:
        func_clustSim(group_dir, acf_file, mc_file)

    print("clustSim done")


if __name__ == "__main__":
    main()
