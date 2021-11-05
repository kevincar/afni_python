# %%
import os
import json
import fnmatch
import subprocess
import pandas as pd
from typing import List, Dict, Optional
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import func_sbatch


def filter_outlier_subects(
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
    decon_list: List[str] = [f for f in os.listdir(reference_directory) if "stats_REML+tlrc.HEAD" in f]
    subject_data_info: pd.DataFrame = pd.DataFrame(columns=["Subject", "N Centers", "Proper Center"])
    decon_list, subject_data_info
    return []


def func_mask(subj_list, deriv_dir, sess, phase, atlas_dir, prior_dir, group_dir):

    # set ref file for resampling
    ref_file = os.path.join(deriv_dir, subj_list[1], sess, f"run-1_{phase}_scale+tlrc")

    # make group intersection mask
    if not os.path.exists(os.path.join(group_dir, "Group_intersect_mask.nii.gz")):

        mask_list = []
        for subj in subj_list:
            mask_file = os.path.join(deriv_dir, subj, sess, "mask_epi_anat+tlrc")
            if os.path.exists(f"{mask_file}.HEAD"):
                mask_list.append(mask_file)

        threshold = 0.3
        h_cmd = f"""
            module load afni-20.2.06
            cd {group_dir}

            cp {atlas_dir}/emu_template* .
            3dMean -prefix Group_intersect_mean.nii.gz {" ".join(mask_list)}
            3dmask_tool \
                -input {" ".join(mask_list)} \
                -frac {threshold} \
                -prefix Group_intersect_mask.nii.gz
        """
        h_mask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_mask.wait()
        if h_mask.returncode != 0:
            raise Exception("Group intersection Mask failed")

    # make GM input
    gm_target_file_path: str = os.path.join(prior_dir, "GM.nii.gz")
    if not os.path.exists(gm_target_file_path):
        prior_file_nums: List[int] = [2, 4]
        prior_file_names: List[str] = [f"Prior{n}.nii.gz" for n in prior_file_nums]
        prior_file_paths: List[str] = [os.path.join(prior_dir, fn) for fn in prior_file_names]
        h_cmd = f"""
            c3 {prior_file_paths[0]} {prior_file_paths[1]} -add -o {gm_target_file_path}
        """
        h_gm = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_gm.wait()
        if h_gm.returncode != 0:
            raise Exception("Group intersection mask priors failed")

    # make GM intersection mask
    if not os.path.exists(os.path.join(group_dir, "Group_GM_intersect_mask+tlrc.HEAD")):
        h_cmd = f"""
            module load afni-20.2.06
            module load c3d-1.0.0-gcc-8.2.0
            cd {group_dir}

            3dresample \
                -master {ref_file} \
                -rmode NN \
                -input {gm_target_file_path} \
                -prefix tmp_GM_mask.nii.gz

            c3d \
                tmp_GM_mask.nii.gz Group_intersect_mask.nii.gz \
                -multiply \
                -o tmp_GM_intersect_prob_mask.nii.gz

            c3d \
                tmp_GM_intersect_prob_mask.nii.gz \
                -thresh 0.1 1 1 0 \
                -o tmp_GM_intersect_mask.nii.gz

            3dcopy tmp_GM_intersect_mask.nii.gz Group_GM_intersect_mask+tlrc
            3drefit -space MNI Group_GM_intersect_mask+tlrc

            if [ -f Group_GM_intersect_mask+tlrc.HEAD ]; then
                rm tmp_*
            fi
        """
        h_GMmask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_GMmask.wait()
        if h_GMmask.returncode != 0:
            raise Exception("GMMask failed")


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
        for behavior, behavior_timepoint in beh_dict:
            input_file_name: str = f"{phase}_signle_stats_REML+tlrc[{behavior_timepoint}]"
            input_file_path: str = os.path.join(session_path, input_file_name)
            data_row: List[str] = [subject_id, behavior, input_file_path]

            if has_bs_factors:
                subject_bs_factor: str = bs_factors[idx]
                data_row.insert(1, subject_bs_factor)

            data_table = data_table.append([data_row])

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
            ws1 = ws_vars[1]
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
        elif hpc == "QSUM":
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
    beh_dict_file_path: str = os.path.join(group_dir, beh_dict_file_name)
    with open(beh_dict_file_path) as json_file:
        beh_dict = json.load(json_file)

    glt_dict_file_name: str = "glt_dict.json"
    glt_dict_file_path: str = os.path.join(group_dir, glt_dict_file_name)
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

    """ run MVM """
    mvm_output_file_name: str = "MVM+tlrc.HEAD"
    mvm_output_file_path: str = os.path.join(group_dir, mvm_output_file_name)
    if not os.path.exists(mvm_output_file_path):
        func_mvm(beh_dict, glt_dict, subj_list, sess, phase, group_dir, deriv_dir, bs_factors)

    """ get subj acf estimates """
    # define, start file
    acf_file = os.path.join(group_dir, "ACF_subj_all.txt")
    if not os.path.exists(acf_file):
        open(acf_file, "a").close()

    # if file is empty, run func_acf for e/subj
    acf_size = os.path.getsize(acf_file)
    if acf_size == 0:
        for subj in subj_list:
            subj_file = os.path.join(
                deriv_dir, subj, sess, f"{phase}_single_errts_REML+tlrc"
            )
            func_acf(subj, subj_file, group_dir, acf_file)

    """ do clust simulations """
    mc_file = os.path.join(group_dir, "MC_thresholds.txt")
    if not os.path.exists(mc_file):
        open(mc_file, "a").close()

    mc_size = os.path.getsize(mc_file)
    if mc_size == 0:
        func_clustSim(group_dir, acf_file, mc_file)


if __name__ == "__main__":
    main()
