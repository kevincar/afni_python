"""
Notes

Timing files must be named "tf_phase_behavior.txt"
    e.g. tf_vCAT_Hit.txt, or tf_test_FA.txt

If using dmBLOCK or TENT options, duration should be married
    to start time.

Decon base models = GAM, 2GAM, dmBLOCK, TENT
    GAM, 2GAM base models not using duration - just use
    BLOCK!

TODO:
    1) Update to be robust against empty timing files
        Idea - break function if tmp_num != float
        then have func_job adjust based on exit status
        of function?
"""

# %%
import os
import fnmatch
import subprocess
import re
import json
from argparse import ArgumentParser
from gp_step0_dcm2nii import func_sbatch


# %%
def func_decon(
    run_files, tf_dict, cen_file, h_phase, h_type, h_desc, work_dir, dmn_list, drv_list
):

    # build epi list for -input
    in_files = []
    for fil in run_files:
        in_files.append(f"{fil.split('.')[0]}")

    # build censor arguments
    reg_base = []
    for cmot, mot in enumerate(dmn_list):
        reg_base.append(f"-ortvec {mot} mot_dmn_run{cmot + 1}")

    for cmot, mot in enumerate(drv_list):
        reg_base.append(f"-ortvec {mot} mot_drv_run{cmot + 1}")

    # determine tr
    h_cmd = f"module load afni-20.2.06 \n 3dinfo -tr {work_dir}/{run_files[0]}"
    h_tr = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    h_len_tr = h_tr.communicate()[0]
    len_tr = float(h_len_tr.decode("utf-8").strip())

    # determine, build behavior regressors
    reg_beh = []
    for c_beh, beh in enumerate(tf_dict):
        if h_type == "dmBLOCK":
            reg_beh.append(f"-stim_times_AM1 {c_beh + 1} {tf_dict[beh]} 'dmBLOCK(1)'")
            reg_beh.append(f"-stim_label {c_beh + 1} {beh}")

        elif h_type == "GAM":
            reg_beh.append(f"-stim_times {c_beh + 1} {tf_dict[beh]} 'GAM'")
            reg_beh.append(f"-stim_label {c_beh + 1} {beh}")

        elif h_type == "2GAM":
            reg_beh.append(
                f"-stim_times {c_beh + 1} {tf_dict[beh]} 'TWOGAMpw(4,5,0.2,12,7)'"
            )
            reg_beh.append(f"-stim_label {c_beh + 1} {beh}")

        elif h_type == "TENT":

            # extract duration, account for no behavior in 1st run
            tmp_str = tf_dict[beh].replace("tf", "dur")
            dur_file = open(os.path.join(work_dir, "timing_files", tmp_str)).readlines()

            if "*" not in dur_file[0]:
                tent_len = round(12 + float(dur_file[0]))
            else:
                with open(os.path.join(work_dir, "timing_files", tmp_str)) as f:
                    for line in f:
                        s = re.search(r"\d+", line)
                        if s:
                            tmp_num = s.string.split("\t")[0]
                tent_len = round(12 + float(tmp_num))
            tent_args = ["0", str(tent_len), str(round(tent_len / len_tr))]

            reg_beh.append(
                f"""-stim_times {c_beh + 1} timing_files/{tf_dict[beh]} 'TENT({",".join(tent_args)})'"""
            )
            reg_beh.append(f"-stim_label {c_beh + 1} {beh}")

    # set output str
    h_out = f"{h_phase}_{h_desc}"

    # build full decon command
    cmd_decon = f"""
        3dDeconvolve \\
            -x1D_stop \\
            -GOFORIT \\
            -input {" ".join(in_files)} \\
            -censor {cen_file} \\
            {" ".join(reg_base)} \\
            -polort A \\
            -float \\
            -local_times \\
            -num_stimts {len(tf_dict.keys())} \\
            {" ".join(reg_beh)} \\
            -jobs 1 \\
            -x1D X.{h_out}.xmat.1D \\
            -xjpeg X.{h_out}.jpg \\
            -x1D_uncensored X.{h_out}.nocensor.xmat.1D \\
            -bucket {h_out}_stats \\
            -cbucket {h_out}_cbucket \\
            -errts {h_out}_errts
    """
    print(cmd_decon)
    return cmd_decon


# %%
def func_job(phase, decon_type, work_dir, sub_num, time_files):

    # # For testing
    # subj = "sub-031"
    # sess = "ses-S1"
    # phase = "Study"
    # decon_type = "TENT"
    # sub_num = subj.split("-")[1]
    # par_dir = "/scratch/madlab/nate_vCAT"
    # work_dir = os.path.join(par_dir, "derivatives", subj, sess)
    # # with open(os.path.join(work_dir, "decon_dict.json")) as json_file:
    # #     decon_dict = json.load(json_file)
    # time_files = decon_dict[phase]

    # %%
    """
    Step 1: Make motion regressors

    Creates motion, demean, and derivative files. Demeaned
        are the ones used in the deconvolution.

    Censor file is combination of 2 things:
        1) Censors based on >0.3 rot/translation relative to previous
            volume. Previous volume censored also.
        2) Which volumes had >10% outlier voxels.
    """

    # make list of pre-processed epi files
    run_list = [
        x
        for x in os.listdir(work_dir)
        if fnmatch.fnmatch(x, f"*{phase}_scale+tlrc.HEAD")
    ]
    num_run = len(run_list)
    run_list.sort()

    # build motion, censor files
    h_cmd = f"""
        cd {work_dir}

        cat dfile.run-*_{phase}.1D > dfile_rall_{phase}.1D

        # make motion files
        1d_tool.py \
            -infile dfile_rall_{phase}.1D \
            -set_nruns {num_run} \
            -demean \
            -write \
            motion_demean_{phase}.1D

        1d_tool.py \
            -infile dfile_rall_{phase}.1D \
            -set_nruns {num_run} \
            -derivative \
            -demean \
            -write \
            motion_deriv_{phase}.1D

        # split into runs
        1d_tool.py \
            -infile motion_demean_{phase}.1D \
            -set_nruns {num_run} \
            -split_into_pad_runs \
            mot_demean_{phase}

        1d_tool.py \
            -infile motion_deriv_{phase}.1D \
            -set_nruns {num_run} \
            -split_into_pad_runs \
            mot_deriv_{phase}

        # make censor file
        1d_tool.py \
            -infile dfile_rall_{phase}.1D \
            -set_nruns {num_run} \
            -show_censor_count \
            -censor_prev_TR \
            -censor_motion 0.3 \
            motion_{phase}

        cat out.cen.run-*{phase}.1D > outcount_censor_{phase}.1D

        1deval \
            -a motion_{phase}_censor.1D \
            -b outcount_censor_{phase}.1D \
            -expr "a*b" > censor_{phase}_combined.1D
    """
    if not os.path.exists(os.path.join(work_dir, f"censor_{phase}_combined.1D")):
        func_sbatch(h_cmd, 1, 1, 1, f"{sub_num}mot", work_dir)

    # %%
    """
    Step 2: Deconvolve

    Uses 3dDeconvolve to generate matrix. 3dREMLfit is then used
        to do a GLS with an ARMA function.

    White matter time series is used as a nuissance regressor.

    Deconvolve script written for review.

    Base models include pmBLOCK, GAM, and TWOGAMpw.
    """

    # Get motion files
    dmn_list = [
        x
        for x in os.listdir(work_dir)
        if fnmatch.fnmatch(x, f"mot_demean_{phase}.*.1D")
    ]
    dmn_list.sort()

    drv_list = [
        x for x in os.listdir(work_dir) if fnmatch.fnmatch(x, f"mot_deriv_{phase}.*.1D")
    ]
    drv_list.sort()

    if type(time_files) == list:

        desc = "single"  # single is just a place holder

        # make timing file dictionary
        tf_dict = {}
        for tf in time_files:
            beh = tf.split("_")[-1].split(".")[0]
            tf_dict[beh] = tf

        # write decon script (for review)
        decon_script = os.path.join(work_dir, f"decon_{phase}_{desc}.sh")
        with open(decon_script, "w") as script:
            script.write(
                func_decon(
                    run_list,
                    tf_dict,
                    f"censor_{phase}_combined.1D",
                    phase,
                    decon_type,
                    desc,
                    work_dir,
                    dmn_list,
                    drv_list,
                )
            )

    elif type(time_files) == dict:
        for desc in time_files:
            # desc = "BE"

            tf_dict = {}
            for tf in time_files[desc]:
                beh = tf.split("_")[-1].split(".")[0]
                tf_dict[beh] = tf

            decon_script = os.path.join(work_dir, f"decon_{phase}_{desc}.sh")
            with open(decon_script, "w") as script:
                script.write(
                    func_decon(
                        run_list,
                        tf_dict,
                        f"censor_{phase}_combined.1D",
                        phase,
                        decon_type,
                        desc,
                        work_dir,
                        dmn_list,
                        drv_list,
                    )
                )

    # %%
    # gather scripts of phase
    script_list = [
        x for x in os.listdir(work_dir) if fnmatch.fnmatch(x, f"decon_{phase}*.sh")
    ]

    # run decon script to generate matrices
    for dcn_script in script_list:
        h_cmd = f"""
            cd {work_dir}
            source {os.path.join(work_dir, dcn_script)}
        """
        func_sbatch(h_cmd, 1, 1, 1, f"{sub_num}dcn", work_dir)

    # %%
    # generate WM timeseries
    if not os.path.exists(os.path.join(work_dir, f"{phase}_WMe_rall+tlrc.HEAD")):
        h_cmd = f"""
            cd {work_dir}

            3dTcat -prefix tmp_allRuns_{phase} run-*{phase}_scale+tlrc.HEAD

            3dcalc \
                -a tmp_allRuns_{phase}+tlrc \
                -b final_mask_WM_eroded+tlrc \
                -expr 'a*bool(b)' \
                -datum float \
                -prefix tmp_allRuns_{phase}_WMe

            3dmerge \
                -1blur_fwhm 20 \
                -doall \
                -prefix {phase}_WMe_rall \
                tmp_allRuns_{phase}_WMe+tlrc
        """
        func_sbatch(h_cmd, 1, 4, 1, f"{sub_num}wts", work_dir)

    # run REML
    if type(time_files) == list:
        desc = "single"
        if not os.path.exists(
            os.path.join(work_dir, f"{phase}_{desc}_stats_REML+tlrc.HEAD")
        ):
            h_cmd = f"""
                cd {work_dir}
                tcsh -x {phase}_{desc}_stats.REML_cmd -dsort {phase}_WMe_rall+tlrc
            """
            func_sbatch(h_cmd, 4, 4, 6, f"{sub_num}rml", work_dir)
    elif type(time_files) == dict:
        for desc in time_files:
            if not os.path.exists(
                os.path.join(work_dir, f"{phase}_{desc}_stats_REML+tlrc.HEAD")
            ):
                h_cmd = f"""
                    cd {work_dir}
                    tcsh -x {phase}_{desc}_stats.REML_cmd -dsort {phase}_WMe_rall+tlrc
                """
                func_sbatch(h_cmd, 4, 4, 6, f"{sub_num}rml", work_dir)


# receive arguments
# parser.add_argument("h_phl", nargs="+", help="Phase List")
def func_argparser():
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("pars_subj", help="Subject ID")
    parser.add_argument("pars_sess", help="Session")
    parser.add_argument("pars_type", help="Decon Type")
    parser.add_argument("pars_dir", help="Derivatives Directory")
    return parser


def main():

    args = func_argparser().parse_args()
    main_work_dir = os.path.join(args.pars_dir, args.pars_subj, args.pars_sess)
    main_sub_num = args.pars_subj.split("-")[1]

    # get time dict
    with open(os.path.join(main_work_dir, "decon_dict.json")) as json_file:
        decon_dict = json.load(json_file)

    # submit job for each phase
    for main_phase in decon_dict:
        func_job(
            main_phase,
            args.pars_type,
            main_work_dir,
            main_sub_num,
            decon_dict[main_phase],
        )


if __name__ == "__main__":
    main()
