# %%
import os
import fnmatch
import subprocess
from gp_step0_dcm2nii import func_sbatch


# %%
def func_mask(subj_list, deriv_dir, sess, phase_list, atlas_dir, prior_dir, out_dir):

    # set ref file for resampling
    ref_file = os.path.join(
        deriv_dir, subj_list[1], sess, f"run-1_{phase_list[0]}_scale+tlrc"
    )

    # make group intersection mask
    if not os.path.exists(os.path.join(out_dir, "Group_intersect_mask.nii.gz")):

        mask_list = []
        for subj in subj_list:
            mask_file = os.path.join(deriv_dir, subj, sess, "mask_epi_anat+tlrc")
            if os.path.exists(f"{mask_file}.HEAD"):
                mask_list.append(mask_file)

        h_cmd = f"""
            module load afni-20.2.06
            cd {out_dir}

            cp {atlas_dir}/vold2_mni_brain* .
            3dMean -prefix Group_intersect_mean.nii.gz {" ".join(mask_list)}
            3dmask_tool \
                -input {" ".join(mask_list)} \
                -frac 1 \
                -prefix Group_intersect_mask.nii.gz
        """
        h_mask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_mask.wait()

    # make GM intersection mask
    if not os.path.exists(os.path.join(out_dir, "Group_GM_intersect_mask+tlrc.HEAD")):
        h_cmd = f"""
            module load afni-20.2.06
            module load c3d-1.0.0-gcc-8.2.0
            cd {out_dir}

            c3d \
                {prior_dir}/Prior2.nii.gz {prior_dir}/Prior4.nii.gz \
                -add \
                -o tmp_Prior_GM.nii.gz

            3dresample \
                -master {ref_file} \
                -rmode NN \
                -input tmp_Prior_GM.nii.gz \
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


# %%
def func_tentAvg(subj, sess, phase, deriv_dir):

    # make list of behaviors fo rphase
    time_dir = os.path.join(deriv_dir, subj, sess, "timing_files")
    beh_list = [
        x.split("_")[-1].split(".")[0]
        for x in os.listdir(time_dir)
        if fnmatch.fnmatch(x, f"tf_{phase}*")
    ]

    # make list of sub-bricks for behaviors
    #   exclude first, last, and fstat bricks
    beh_dict = {}
    ref_file = os.path.join(deriv_dir, subj, sess, f"{phase}_single_stats_REML+tlrc")
    for beh in beh_list:
        h_cmd = f"""
            module load afni-20.2.06
            3dinfo -subbrick_info {ref_file} |
                grep "{beh}" |
                awk '{{print $4}}' |
                sed 's/\#//'
        """
        h_list = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        (h_out, h_err) = h_list.communicate()
        h_list.wait()
        ind_list = h_out.decode("utf-8").replace("\n", " ").rstrip(" ").split()
        beh_dict[beh] = ind_list[1:-2]

    # combine face, scene sub-bricks in
    #   order to do a F+S > Num comparison
    comp_dict = {}
    comp_dict["num"] = beh_dict["num"]
    comp_dict["face"] = beh_dict["face"]
    comp_dict["scene"] = beh_dict["scene"]
    comp_dict["FS"] = beh_dict["face"] + beh_dict["scene"]

    # make mean of tents for each comp
    for comp in comp_dict:
        if not os.path.exists(
            os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_{comp}+tlrc.HEAD")
        ):
            h_cmd = f"""
                module load afni-20.2.06
                cd {os.path.join(deriv_dir, subj, sess)}

                3dTstat \
                    -prefix {phase}_tentAvg_{comp} \
                    {phase}_single_stats_REML+tlrc[{",".join(comp_dict[comp])}]
            """
            h_avg = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_avg.wait()


# %%
def func_etac(subj_list, out_dir, deriv_dir, sess, strA, strB, phase):

    # set up ETAC script
    list_A = []
    list_B = []
    for subj in subj_list:
        list_A.append(subj)
        list_A.append(
            os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_{strA}+tlrc")
        )
        list_B.append(subj)
        list_B.append(
            os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_{strB}+tlrc")
        )

    h_cmd = f"""
        module load afni-20.2.06
        cd {out_dir}

        # -ETAC_blur 4 6 8
        3dttest++ \\
            -paired \\
            -mask Group_GM_intersect_mask+tlrc \\
            -prefix {phase}_{strA}-{strB} \\
            -prefix_clustsim {phase}_{strA}-{strB}_clustsim \\
            -ETAC \\
            -ETAC_opt NN=2:sid=2:hpow=0:pthr=0.01,0.005,0.001:name=NN2 \\
            -setA A {" ".join(list_A)} \\
            -setB B {" ".join(list_B)}

        3dcopy {phase}_{strA}-{strB}_clustsim.NN2.ETACmask.global.2sid.5perc.nii.gz \
            FINAL_{phase}_{strA}-{strB}+tlrc
    """
    func_sbatch(h_cmd, 40, 6, 10, "vCATetac", out_dir)


def func_ttest(subj_list, out_dir, deriv_dir, sess, strA, strB, phase):

    # set up ETAC script
    list_A = []
    list_B = []
    for subj in subj_list:
        list_A.append(subj)
        list_A.append(
            os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_{strA}+tlrc")
        )
        list_B.append(subj)
        list_B.append(
            os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_{strB}+tlrc")
        )

    h_cmd = f"""
        module load afni-20.2.06
        cd {out_dir}

        3dttest++ \\
            -paired \\
            -mask Group_GM_intersect_mask+tlrc \\
            -prefix {phase}_{strA}-{strB}_tt \\
            -setA A {" ".join(list_A)} \\
            -setB B {" ".join(list_B)}
    """
    func_sbatch(h_cmd, 2, 4, 4, "vCATtt", out_dir)


# %%
def main():

    # set vars, lists
    phase_list = ["loc", "Study"]
    sess = "ses-S1"
    deriv_dir = "/scratch/madlab/nate_vCAT/derivatives"
    out_dir = "/scratch/madlab/nate_vCAT/analyses"
    atlas_dir = "/home/data/madlab/atlases/vold2_mni"
    prior_dir = os.path.join(atlas_dir, "priors_ACT")

    # group analysis dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get subjects
    subj_list = [x for x in os.listdir(deriv_dir) if fnmatch.fnmatch(x, "sub-*")]
    subj_list.sort()

    # make group gray matter intreset mask
    if not os.path.exists(os.path.join(out_dir, "Group_GM_intersect_mask+tlrc.HEAD")):
        func_mask(subj_list, deriv_dir, sess, phase_list, atlas_dir, prior_dir, out_dir)

    # make average tent for Face+Scene and Number of
    #   localizer task
    phase = phase_list[0]
    for subj in subj_list:
        file1 = os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_FS+tlrc.HEAD")
        file2 = os.path.join(deriv_dir, subj, sess, f"{phase}_tentAvg_num+tlrc.HEAD")
        if not os.path.exists(file1) or not os.path.exists(file2):
            func_tentAvg(subj, sess, phase, deriv_dir)

    # run etac - face vs scene
    if not os.path.exists(os.path.join(out_dir, "FINAL_loc_face-scene+tlrc.HEAD")):
        func_etac(subj_list, out_dir, deriv_dir, sess, "face", "scene", phase)

    # face + scene vs num
    if not os.path.exists(os.path.join(out_dir, "FINAL_loc_FS-num+tlrc.HEAD")):
        func_etac(subj_list, out_dir, deriv_dir, sess, "FS", "num", phase)

    # run ttest - face vs scene
    if not os.path.exists(os.path.join(out_dir, "FINAL_loc_face-scene_tt+tlrc.HEAD")):
        func_ttest(subj_list, out_dir, deriv_dir, sess, "face", "scene", phase)


if __name__ == "__main__":
    main()
