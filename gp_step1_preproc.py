# %%
"""
Notes

1) Script will generate named sbatch subprocesses named for each sub-step.
    Output for each sbatch step is written to work_dir and prepeneded
    with "sbatch_writeOut"

2) Pre-processing steps include copying data to derivatives, distortion correcting,
    volume registration, normalization, generating intersection masks, blurring, and
    scaling the data.

3) Written in Python 3.8, has afni and c3d dependencies.
"""

# %%
import os
import subprocess
import fnmatch
import math
from argparse import ArgumentParser
from gp_step0_dcm2nii import func_sbatch


# %%
# helper functions
def func_epi_list(phase, h_dir):
    h_list = [
        x.split("+")[0]
        for x in os.listdir(h_dir)
        if fnmatch.fnmatch(x, f"run-*_{phase}+orig.HEAD")
    ]
    h_list.sort()
    return h_list


def flatten_list(list_2d):
    flat_list = []
    for element in list_2d:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


# %%
# pipeline functions
def func_copy_data(subj, sess, work_dir, data_dir, phase_list):

    """
    Step 1: Copy data into work_dir

    1) Get func, anat, fmap data. Rename appropriately.
    """

    # struct
    if not os.path.exists(os.path.join(work_dir, "struct+orig.HEAD")):
        h_cmd = f"""
            module load afni-20.2.06
            3dcopy \
                {data_dir}/anat/{subj}_{sess}_T1w.nii.gz \
                {work_dir}/struct+orig
        """
        h_copy = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_copy.wait()

    # epi - keep runs sorted by phase
    for phase in phase_list:

        epiNii_list = [
            epi
            for epi in os.listdir(os.path.join(data_dir, "func"))
            if fnmatch.fnmatch(epi, f"*task-{phase}*.nii.gz")
        ]
        epiNii_list.sort()

        for h_count, h_file in enumerate(epiNii_list):
            run_count = 1 + h_count
            if not os.path.exists(f"{work_dir}/run-{run_count}_{phase}+orig.HEAD"):
                h_cmd = f"""
                    module load afni-20.2.06
                    3dcopy \
                        {data_dir}/func/{h_file} \
                        {work_dir}/run-{run_count}_{phase}+orig
                """
                h_copy = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                h_copy.wait()

    # fmap
    fmap_list = [
        x
        for x in os.listdir(os.path.join(data_dir, "fmap"))
        if fnmatch.fnmatch(x, "*.nii.gz")
    ]

    for fmap in fmap_list:
        h_dir = fmap.split("-")[-1].split("_")[0]
        if not os.path.exists(f"{work_dir}/blip_{h_dir}+orig.HEAD"):
            h_cmd = f"""
                module load afni-20.2.06
                3dcopy \
                    {data_dir}/fmap/{fmap} \
                    {work_dir}/blip_{h_dir}
            """
            h_copy = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_copy.wait()


def func_outliers(work_dir, phase_list, subj_num, out_thresh):

    """
    Step 2: Detect outliers voxels, blip correct

    1) Determine the proportion of voxels per volume that have outlier signal.
        Censor volumes that exceed limit.
    """

    for phase in phase_list:
        epi_list = func_epi_list(phase, work_dir)
        for run in epi_list:
            if not os.path.exists(os.path.join(work_dir, f"out.cen.{run}.1D")):

                # calc tr length
                h_cmd = f"""
                    module load afni-20.2.06
                    3dinfo -tr {work_dir}/{run}+orig
                """
                h_tr = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                h_len_tr = h_tr.communicate()[0]
                len_tr = h_len_tr.decode("utf-8").strip()

                # calc number of volumes
                h_cmd = f"""
                    module load afni-20.2.06
                    3dinfo -ntimes {work_dir}/{run}+orig
                """
                h_nvol = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                h_nvol_out = h_nvol.communicate()[0]
                num_tr = h_nvol_out.decode("utf-8").strip()

                # determine polort argument
                pol = 1 + math.ceil((float(len_tr) * float(num_tr)) / 150)

                # determine percentage outliers for e/volume
                h_cmd = f"""
                    module load afni-20.2.06
                    cd {work_dir}

                    3dToutcount \
                        -automask \
                        -fraction \
                        -polort {pol} \
                        -legendre {run}+orig > outcount.{run}.1D

                    1deval \
                        -a outcount.{run}.1D \
                        -expr '1-step(a-{out_thresh})' > out.cen.{run}.1D
                """
                h_outc = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                h_outc.wait()


def func_fmap_corr(work_dir, subj_num, phase_list):

    """
    Step 3: Blip correct data

    1) Calculate median of AP, PA files
    2) Compute midpoint between media files
    3) Apply warp to de-distort (unwarp) EPI data

    Note: If acq = LR, fmap = RL:
        base = LR, source = RL
        -pmNAMES RL LR
        unwarp LR with LR_WARP
    """

    # create median datasets and masks
    for h_dir in ["AP", "PA"]:
        if not os.path.exists(
            os.path.join(work_dir, f"tmp_med_masked_{h_dir}+orig.HEAD")
        ):
            h_cmd = f"""
                cd {work_dir}

                3dTstat \
                    -median \
                    -prefix tmp_med_{h_dir} \
                    blip_{h_dir}+orig

                3dAutomask \
                    -apply_prefix tmp_med_masked_{h_dir} \
                    tmp_med_{h_dir}+orig
            """
            func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}med", work_dir)

    # compute midpoint between fmaps
    if not os.path.exists(os.path.join(work_dir, "blip_warp_For_WARP+orig.HEAD")):
        h_cmd = f"""
            cd {work_dir}

            3dQwarp \
                -plusminus \
                -pmNAMES Rev For \
                -pblur 0.05 0.05 \
                -blur -1 -1 \
                -noweight \
                -minpatch 9 \
                -source tmp_med_masked_PA+orig \
                -base tmp_med_masked_AP+orig \
                -prefix blip_warp
        """
        func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}qwa", work_dir)

    # unwarp run data (de-distort), apply header
    for phase in phase_list:
        epi_list = func_epi_list(phase, work_dir)
        for run in epi_list:
            if not os.path.exists(os.path.join(work_dir, f"{run}_blip+orig.HEAD")):
                h_cmd = f"""
                    cd {work_dir}

                    3dNwarpApply \
                        -quintic \
                        -nwarp blip_warp_For_WARP+orig \
                        -source {run}+orig \
                        -prefix {run}_blip

                    3drefit \
                        -atrcopy blip_AP+orig \
                        IJK_TO_DICOM_REAL \
                        {run}_blip+orig
                """
                func_sbatch(h_cmd, 1, 2, 4, f"{subj_num}nwar", work_dir)


def func_vrbase(work_dir, phase_list, blip_tog):

    """
    Step 4: Make volreg base

    1) The volreg base (epi_vrBase) is the single volume in entire experiment
        with the smallest number of outlier volumes.

    Note: If data is not time shifted, do so before determining volreg_base.
        Also, it was easier to write a small bash script due to the multiple
        afni commands.
    """

    # combine all outcount.* into one master file (outcount_all.1D)
    #      1D to rule them all, and in the darkness bind them
    # make sure things are in order
    out_dict = {}
    for phase in phase_list:
        h_list = [
            os.path.join(work_dir, x)
            for x in os.listdir(work_dir)
            if fnmatch.fnmatch(x, f"outcount.*{phase}.1D")
        ]
        h_list.sort()
        out_dict[phase] = h_list
    out_list = flatten_list(list(out_dict.values()))

    out_all = os.path.join(work_dir, "outcount_all.1D")
    with open(out_all, "w") as outfile:
        for out_file in out_list:
            with open(out_file) as infile:
                outfile.write(infile.read())

    # all epi runs in experiment, same order as out_list!
    scan_dict = {}
    h_str = "_blip+orig" if blip_tog == 1 else "+orig"

    for phase in phase_list:
        h_list = [
            x.split(".")[0]
            for x in os.listdir(work_dir)
            if fnmatch.fnmatch(x, f"run-*{phase}{h_str}.HEAD")
        ]
        h_list.sort()
        scan_dict[phase] = h_list
    scan_list = flatten_list(list(scan_dict.values()))

    # make volume registration base
    if not os.path.exists(os.path.join(work_dir, "epi_vrBase+orig.HEAD")):

        # list of volume nums
        num_vols = []
        for scan in scan_list:
            h_cmd = f"""
                module load afni-20.2.06
                3dinfo -ntimes {work_dir}/{scan}
            """
            h_nvol = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_nvol.wait()
            h_nvol_out = h_nvol.communicate()[0]
            num_tr = h_nvol_out.decode("utf-8").strip()
            num_vols.append(num_tr)

        # determine index of min
        h_cmd = f"""
            module load afni-20.2.06
            3dTstat \
                -argmin \
                -prefix - \
                {work_dir}/outcount_all.1D\\'
        """
        h_mind = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_mind.wait()
        h_ind = h_mind.communicate()[0]
        ind_min = h_ind.decode("utf-8").strip()

        # determine min volume
        h_cmd = f"""
            module load afni-20.2.06
            1d_tool.py \
                -set_run_lengths {" ".join(num_vols)} \
                -index_to_run_tr {ind_min}
        """
        h_minV = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_minV.wait()
        h_min = h_minV.communicate()[0]
        min_runVol = h_min.decode("utf-8").strip().split()

        # determine run, volume
        #   account for 0 vs 1 indexing
        min_run = scan_list[int(min_runVol[0]) - 1]
        min_vol = int(min_runVol[1])

        # make epi volreg base
        h_cmd = f"""
            module load afni-20.2.06
            cd {work_dir}
            3dbucket \
                -prefix epi_vrBase \
                {min_run}"[{min_vol}]"
        """
        h_vrb = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_vrb.wait()


def func_register(atlas, work_dir, subj_num):

    """
    Step 5: Calculate normalization

    1) This step will perform the rigid alignments of T1->EPI
        and non-linear diffeomorphic of T1->Template.
    """

    if not os.path.exists(os.path.join(work_dir, "struct_ns+tlrc.HEAD")):
        h_cmd = f"""
            cd {work_dir}

            align_epi_anat.py \
                -anat2epi \
                -anat struct+orig \
                -save_skullstrip \
                -suffix _al_junk \
                -epi epi_vrBase+orig \
                -epi_base 0 \
                -epi_strip 3dAutomask \
                -cost lpc+ZZ \
                -giant_move \
                -check_flip \
                -volreg off \
                -tshift off

            auto_warp.py \
                -base {atlas} \
                -input struct_ns+orig \
                -skull_strip_input no

            3dbucket \
                -DAFNI_NIFTI_VIEW=tlrc \
                -prefix struct_ns \
                awpy/struct_ns.aw.nii*

            cp awpy/anat.un.aff.Xat.1D .
            cp awpy/anat.un.aff.qw_WARP.nii .
        """
        func_sbatch(h_cmd, 1, 4, 4, f"{subj_num}dif", work_dir)


def func_volreg_warp(work_dir, phase_list, subj_num, blip_tog):

    """
    Step 6: Warp EPI data to template space

    1) This step will perform the rigid alignments of T1-EPI (A)
        EPI-EPI base volume (B), and non-linear diffeomorphic of
        T1-Template (C). Note blip distortion map (D).

    2) Together, then, we will have T1-EPI (A), EPI-EPI base volume (B),
        and non-linear diffeomorphic of T1-Template (C) and possibly
        blip distortion map (D) matrices.

    3) It will then concatenate these warp matrices, and warp EPI data from
        raw/native space to template space via W=A'+B+C+D. Thus, only one
        interpolation of the epi data occurs.
    """

    scan_dict = {}
    h_str = "_blip+orig" if blip_tog == 1 else "+orig"

    for phase in phase_list:
        h_list = [
            x.split(".")[0]
            for x in os.listdir(work_dir)
            if fnmatch.fnmatch(x, f"run-*{phase}{h_str}.HEAD")
        ]
        h_list.sort()
        scan_dict[phase] = h_list
    scan_list = flatten_list(list(scan_dict.values()))

    for h_run in scan_list:

        # get run str
        run = h_run.split("_blip")[0] if blip_tog == 1 else h_run.split("+")[0]

        # Calculate volreg for e/run
        if not os.path.exists(os.path.join(work_dir, f"mat.{run}.vr.aff12.1D")):
            h_cmd = f"""
                cd {work_dir}

                3dvolreg \
                    -verbose \
                    -zpad 1 \
                    -base epi_vrBase+orig \
                    -1Dfile dfile.{run}.1D \
                    -prefix {run}_volreg \
                    -cubic \
                    -1Dmatrix_save mat.{run}.vr.aff12.1D \
                    {h_run}
            """
            func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}vre", work_dir)

        # set up, warp EPI to template
        if not os.path.exists(os.path.join(work_dir, f"{run}_warp+tlrc.HEAD")):

            # get grid size
            h_cmd = f"""
                module load afni-20.2.06
                3dinfo -di {work_dir}/{run}+orig
            """
            h_gs = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_gs_out = h_gs.communicate()[0]
            grid_size = h_gs_out.decode("utf-8").strip()

            # concatenate matrices
            h_cmd = f"""
                module load afni-20.2.06
                cd {work_dir}

                cat_matvec \
                    -ONELINE \
                    anat.un.aff.Xat.1D \
                    struct_al_junk_mat.aff12.1D -I \
                    mat.{run}.vr.aff12.1D > mat.{run}.warp.aff12.1D
            """
            h_cat = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_cat.wait()

            # warp epi, mask into template space
            nwarp_list = ["anat.un.aff.qw_WARP.nii", f"mat.{run}.warp.aff12.1D"]
            if blip_tog == 1:
                nwarp_list.append("blip_warp_For_WARP+orig")

            h_cmd = f"""
                cd {work_dir}

                3dNwarpApply \
                    -master struct_ns+tlrc \
                    -dxyz {grid_size} \
                    -source {run}+orig \
                    -nwarp '{" ".join(nwarp_list)}' \
                    -prefix {run}_warp
            """
            func_sbatch(h_cmd, 2, 4, 4, f"{subj_num}war", work_dir)

        # Update - don't waste computation time warping
        #   simple mask into template space. Just make
        #   mask from warped EPI data
        if not os.path.exists(os.path.join(work_dir, f"tmp_{run}_min+tlrc.HEAD")):
            h_cmd = f"""
                module load afni-20.2.06
                cd {work_dir}

                3dcalc \
                    -overwrite \
                    -a {run}_warp+tlrc \
                    -expr 1 \
                    -prefix tmp_{run}_mask

                3dTstat \
                    -min \
                    -prefix tmp_{run}_min \
                    tmp_{run}_mask+tlrc
            """
            h_mask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_mask.wait()


def func_clean_volreg(work_dir, phase_list, subj_num):

    """
    Step 7: Clean volreg data

    1) Censor potentially bad volumes to avoid biasing scale step.

    Note: This rarely has an effect.
    """

    # Determine minimum value, make mask
    #   beware the jabberwocky i.e. expanding braces in 3dMean
    for phase in phase_list:
        epi_list = func_epi_list(phase, work_dir)
        if not os.path.exists(os.path.join(work_dir, f"{phase}_minVal_mask+tlrc.HEAD")):
            h_cmd = f"""
                cd {work_dir}

                3dMean \
                    -datum short \
                    -prefix tmp_mean_{phase} \
                    tmp_run-{{1..{len(epi_list)}}}_{phase}_min+tlrc

                3dcalc \
                    -a tmp_mean_{phase}+tlrc \
                    -expr 'step(a-0.999)' \
                    -prefix {phase}_minVal_mask
            """
            func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}min", work_dir)

        # make clean data
        for run in epi_list:
            if not os.path.exists(
                os.path.join(work_dir, f"{run}_volreg_clean+tlrc.HEAD")
            ):
                h_cmd = f"""
                    cd {work_dir}
                    3dcalc \
                        -a {run}_warp+tlrc \
                        -b {phase}_minVal_mask+tlrc \
                        -expr 'a*b' \
                        -prefix {run}_volreg_clean
                """
                func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}cle", work_dir)


def func_blur(work_dir, subj_num, blur_mult):

    """
    Step 8: Blur EPI data

    1) Blur kernel is size = blur_multiplier * voxel dim,
        rounded up to nearest int. FWHM.
    """

    # Blur
    epi_list = [
        x.split("_vol")[0]
        for x in os.listdir(work_dir)
        if fnmatch.fnmatch(x, "*volreg_clean+tlrc.HEAD")
    ]

    for run in epi_list:
        if not os.path.exists(os.path.join(work_dir, f"{run}_blur+tlrc.HEAD")):

            # calc voxel dim i
            h_cmd = f"""
                module load afni-20.2.06
                3dinfo -di {work_dir}/{run}+orig
            """
            h_gs = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_gs_out = h_gs.communicate()[0]
            grid_size = h_gs_out.decode("utf-8").strip()
            blur_size = math.ceil(blur_mult * float(grid_size))

            # do blur
            h_cmd = f"""
                cd {work_dir}
                3dmerge \
                    -1blur_fwhm {blur_size} \
                    -doall \
                    -prefix {run}_blur \
                    {run}_volreg_clean+tlrc
            """
            func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}blur", work_dir)


def func_tiss_masks(work_dir, subj_num, atropos_dict, atropos_dir):

    """
    Step 9:  Make union and tissue masks

    1) Make a union mask, where sufficient signal exists for both T1w
        and T2*w at each voxel for analyses. Incorporated at the
        group-level analysis.

    2) Make tissue masks. The WM mask will be used later to derive
        nuissance regressors for the REML.
        Note: this references some custom masks, and is based in
            atropos rather than in AFNIs tiss seg protocol.
    """

    epi_list = [
        x.split("_vol")[0]
        for x in os.listdir(work_dir)
        if fnmatch.fnmatch(x, "*volreg_clean+tlrc.HEAD")
    ]

    # Make EPI-T1 union mask (mask_epi_anat)
    if not os.path.exists(os.path.join(work_dir, "mask_epi_anat+tlrc.HEAD")):

        for run in epi_list:
            if not os.path.exists(
                os.path.join(work_dir, f"tmp_mask.{run}_blur+tlrc.HEAD")
            ):
                h_cmd = f"""
                    module load afni-20.2.06
                    cd {work_dir}
                    3dAutomask -prefix tmp_mask.{run} {run}_blur+tlrc
                """
                h_mask = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                h_mask.wait()

        h_cmd = f"""
            cd {work_dir}

            3dmask_tool \
                -inputs tmp_mask.*+tlrc.HEAD \
                -union \
                -prefix tmp_mask_allRuns

            3dresample \
                -master tmp_mask_allRuns+tlrc \
                -input struct_ns+tlrc \
                -prefix tmp_anat_resamp

            3dmask_tool \
                -dilate_input 5 -5 \
                -fill_holes \
                -input tmp_anat_resamp+tlrc \
                -prefix tmp_mask_struct

            3dmask_tool \
                -input tmp_mask_allRuns+tlrc tmp_mask_struct+tlrc \
                -inter \
                -prefix mask_epi_anat

            3dABoverlap \
                -no_automask tmp_mask_allRuns+tlrc tmp_mask_struct+tlrc | \
                tee out.mask_ae_overlap.txt
        """
        func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}uni", work_dir)

    # Make tissue-class masks
    #   I like Atropos better than AFNI's way, so use those priors
    h_ref = f"{epi_list[0]}_blur+tlrc"

    for key in atropos_dict:
        h_tiss = atropos_dict[key]
        if not os.path.exists(
            os.path.join(work_dir, f"final_mask_{h_tiss}_eroded+tlrc.HEAD")
        ):
            h_cmd = f"""
                module load c3d-1.0.0-gcc-8.2.0
                cd {work_dir}

                c3d \
                    {atropos_dir}/Prior{key}.nii.gz \
                    -thresh 0.3 1 1 0 \
                    -o tmp_{h_tiss}_bin.nii.gz

                3dresample \
                    -master {h_ref} \
                    -rmode NN \
                    -input tmp_{h_tiss}_bin.nii.gz \
                    -prefix final_mask_{h_tiss}+tlrc

                3dmask_tool \
                    -input tmp_{h_tiss}_bin.nii.gz \
                    -dilate_input -1 \
                    -prefix tmp_mask_{h_tiss}_eroded

                3dresample \
                    -master {h_ref} \
                    -rmode NN \
                    -input tmp_mask_{h_tiss}_eroded+orig \
                    -prefix final_mask_{h_tiss}_eroded
            """
            func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}atr", work_dir)


def func_scale(work_dir, phase_list, subj_num):

    """
    Step 10: Scale data

    1) Data is scaled by mean signal
    """

    for phase in phase_list:
        epi_list = func_epi_list(phase, work_dir)
        for run in epi_list:
            if not os.path.exists(os.path.join(work_dir, f"{run}_scale+tlrc.HEAD")):
                h_cmd = f"""
                    cd {work_dir}

                    3dTstat -prefix tmp_tstat_{run} {run}_blur+tlrc

                    3dcalc -a {run}_blur+tlrc \
                        -b tmp_tstat_{run}+tlrc \
                        -c {phase}_minVal_mask+tlrc \
                        -expr 'c * min(200, a/b*100)*step(a)*step(b)' \
                        -prefix {run}_scale
                """
                func_sbatch(h_cmd, 1, 1, 1, f"{subj_num}scale", work_dir)


def func_argparser():
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("h_sub", help="Subject ID")
    parser.add_argument("h_ses", help="Session")
    parser.add_argument("h_par", help="Parent Directory")
    parser.add_argument("h_bt", help="Blip Toggle")
    parser.add_argument("h_phl", nargs="+", help="Phase List")
    return parser


# %%
def main():

    # """ For testing """
    # subj = "sub-005"
    # sess = "ses-S1"
    # phase_list = ["loc", "Study"]
    # blip_tog = 0

    # par_dir = "/scratch/madlab/nate_vCAT"
    # data_dir = os.path.join(par_dir, "dset", subj, sess)
    # work_dir = os.path.join(par_dir, "derivatives", subj, sess)

    """ Get passed arguments """
    args = func_argparser().parse_args()
    subj = args.h_sub
    sess = args.h_ses
    par_dir = args.h_par
    phase_list = args.h_phl
    blip_tog = args.h_bt

    """ Make some vars """
    data_dir = os.path.join(par_dir, "dset", subj, sess)
    work_dir = os.path.join(par_dir, "derivatives", subj, sess)
    subj_num = subj.split("-")[1]

    """ Set up deriv """
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    """ Copy dset niftis to derivatives """
    check_func = os.path.join(work_dir, f"run-1_{phase_list[0]}+orig.HEAD")
    if not os.path.exists(check_func):
        func_copy_data(subj, sess, work_dir, data_dir, phase_list)

    """ Determine outlier voxels per volume """
    out_thresh = 0.1
    if not os.path.exists(os.path.join(work_dir, f"out.cen.run-1_{phase_list[0]}.1D")):
        func_outliers(work_dir, phase_list, subj_num, out_thresh)

    """ fmap correct data """
    if blip_tog == 1:
        check_blip = os.path.join(work_dir, f"run-1_{phase_list[0]}_blip+orig.HEAD")
        if not os.path.exists(check_blip):
            func_fmap_corr(work_dir, subj_num, phase_list)

    """ Make volume registration base """
    if not os.path.exists(os.path.join(work_dir, "epi_vrBase+orig.HEAD")):
        func_vrbase(work_dir, phase_list, blip_tog)

    """ Calculate normalization vectors """
    atlas_dir = "/home/data/madlab/atlases/vold2_mni"
    atlas = os.path.join(atlas_dir, "vold2_mni_brain+tlrc")
    check_diffeo = os.path.join(work_dir, "anat.un.aff.qw_WARP.nii")
    if not os.path.exists(check_diffeo):
        func_register(atlas, work_dir, subj_num)

    """ Volreg, warp epi to template space """
    check_warp = os.path.join(work_dir, f"run-1_{phase_list[0]}_warp+tlrc.HEAD")
    if not os.path.exists(check_warp):
        func_volreg_warp(work_dir, phase_list, subj_num, blip_tog)

    """ Clean warped data """
    check_clean = os.path.join(
        work_dir, f"run-1_{phase_list[0]}_volreg_clean+tlrc.HEAD"
    )
    if not os.path.exists(check_clean):
        func_clean_volreg(work_dir, phase_list, subj_num)

    """ Blur data """
    blur_mult = 1.5
    check_blur = os.path.join(work_dir, f"run-1_{phase_list[0]}_blur+tlrc.HEAD")
    if not os.path.exists(check_blur):
        func_blur(work_dir, subj_num, blur_mult)

    """ Make tissue masks """
    atropos_dict = {1: "CSF", 2: "GMc", 3: "WM", 4: "GMs"}
    atropos_dir = os.path.join(atlas_dir, "priors_ACT")
    check_tiss = os.path.join(work_dir, "final_mask_CSF_eroded+tlrc.HEAD")
    if not os.path.exists(check_tiss):
        func_tiss_masks(work_dir, subj_num, atropos_dict, atropos_dir)

    """ Scale data """
    check_scale = os.path.join(work_dir, f"run-1_{phase_list[0]}_scale+tlrc.HEAD")
    if not os.path.exists(check_scale):
        func_scale(work_dir, phase_list, subj_num)


if __name__ == "__main__":
    main()

# %%
