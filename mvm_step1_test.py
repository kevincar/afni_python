# %%
import os
import json
import fnmatch
import subprocess
import pandas as pd
from argparse import ArgumentParser
from gp_step1_preproc import func_sbatch


# %%
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

        h_cmd = f"""
            module load afni-20.2.06
            cd {group_dir}

            cp {atlas_dir}/emu_template* .
            3dMean -prefix Group_intersect_mean.nii.gz {" ".join(mask_list)}
            3dmask_tool \
                -input {" ".join(mask_list)} \
                -frac 1 \
                -prefix Group_intersect_mask.nii.gz
        """
        h_mask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_mask.wait()

    # make GM intersection mask
    if not os.path.exists(os.path.join(group_dir, "Group_GM_intersect_mask+tlrc.HEAD")):
        h_cmd = f"""
            module load afni-20.2.06
            module load c3d-1.0.0-gcc-8.2.0
            cd {group_dir}

            3dresample \
                -master {ref_file} \
                -rmode NN \
                -input {prior_dir}/GM.nii.gz \
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


def func_mvm(beh_dict, glt_dict, subj_list, sess, phase, group_dir, deriv_dir):

    data_table = []
    for beh in beh_dict:
        for subj in subj_list:
            data_table.append(subj)
            data_table.append(beh)
            h_file = os.path.join(
                deriv_dir,
                subj,
                sess,
                f"""'{phase}_single_stats_REML+tlrc[{beh_dict[beh]}]'""",
            )
            data_table.append(h_file)

    glt_list = []
    for count, test in enumerate(glt_dict):
        glt_list.append(f"-gltLabel {count + 1} {test}")
        glt_list.append(f"-gltCode {count + 1}")
        glt_list.append(f"'WSVARS: 1*{glt_dict[test][0]} -1*{glt_dict[test][1]}'")

    h_cmd = f"""
        cd {group_dir}

        3dMVM \
            -prefix MVM \
            -jobs 10 \
            -mask Group_GM_intersect_mask+tlrc \
            -bsVars 1 \
            -wsVars 'WSVARS' \
            -num_glt {len(list(glt_dict.keys()))} \
            {" ".join(glt_list)} \
            -dataTable \
            Subj WSVARS InputFile \
            {" ".join(data_table)}
    """
    func_sbatch(h_cmd, 2, 6, 10, "cMVM", group_dir)


def func_acf(subj, subj_file, group_dir, acf_file):
    h_cmd = f"""
        cd {group_dir}

        3dFWHMx \
            -mask Group_GM_intersect_mask+tlrc \
            -input {subj_file} \
            -acf >> {acf_file}
    """
    func_sbatch(h_cmd, 2, 4, 1, f"a{subj.split('-')[-1]}", group_dir)


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
    func_sbatch(h_cmd, 6, 4, 10, "mc", group_dir)


def func_argparser():
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("sess", help="Session")
    parser.add_argument("phase", help="Phase")
    parser.add_argument("group_dir", help="Output Directory")
    parser.add_argument("atlas_dir", help="Location of atlas")
    parser.add_argument("prior_dir", help="Location of atlas priors")
    parser.add_argument("parent_dir", help="Location of Project Directory")

    return parser


# %%
def main():

    # TODO update for multiple MVMs

    # get args
    args = func_argparser().parse_args()
    subj_list = args.subj_list
    sess = args.sess
    phase = args.phase
    group_dir = args.group_dir
    atlas_dir = args.atlas_dir
    prior_dir = args.prior_dir
    parent_dir = args.parent_dir

    # get/make paths, dicts
    deriv_dir = os.path.join(parent_dir, "derivatives")
    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]

    with open(os.path.join(group_dir, "beh_dict.json")) as json_file:
        beh_dict = json.load(json_file)

    with open(os.path.join(group_dir, "glt_dict.json")) as json_file:
        glt_dict = json.load(json_file)

    """ make group gray matter intreset mask """
    if not os.path.exists(os.path.join(group_dir, "Group_GM_intersect_mask+tlrc.HEAD")):
        func_mask(subj_list, deriv_dir, sess, phase, atlas_dir, prior_dir, group_dir)

    """ run MVM """
    if not os.path.exists(os.path.join(group_dir, "MVM+tlrc.HEAD")):
        func_mvm(beh_dict, glt_dict, subj_list, sess, phase, group_dir, deriv_dir)

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
