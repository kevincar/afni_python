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
def func_write_decon(
    run_list, tf_dict, cen_file, phase, decon_type, desc, work_dir, dmn_list, drv_list
):
    """
    Notes: This function generates a 3dDeconvolve command.
        It supports GAM, 2GAM, TENT, and dmBLOCK basis functions.
        TENT does not currently include duration.

    """

    # build censor arguments
    reg_base = []
    for cmot, mot in enumerate(dmn_list):
        reg_base.append(f"-ortvec {mot} mot_dmn_run{cmot + 1}")

    for cmot, mot in enumerate(drv_list):
        reg_base.append(f"-ortvec {mot} mot_drv_run{cmot + 1}")

    # determine tr
    h_cmd = f"""
        module load afni-20.2.06
        3dinfo -tr {work_dir}/{run_list[0]}
    """
    h_tr = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    h_len_tr = h_tr.communicate()[0]
    len_tr = float(h_len_tr.decode("utf-8").strip())

    # determine, build behavior regressors
    switch_dict = {
        "dmBLOCK": "'dmBLOCK(1)'",
        "GAM": "'GAM'",
        "2GAM": "'TWOGAMpw(4,5,0.2,12,7)'",
    }

    reg_beh = []
    for c_beh, beh in enumerate(tf_dict):
        if decon_type == "dmBLOCK" or decon_type == "GAM" or decon_type == "2GAM":

            # add stim_time info, order is
            #   -stim_times 1 tf_beh.txt basisFunction
            reg_beh.append("-stim_times_AM1")
            reg_beh.append(f"{c_beh + 1}")
            reg_beh.append(f"timing_files/{tf_dict[beh]}")
            reg_beh.append(switch_dict[decon_type])

            # add stim_label info, order is
            #   -stim_label 1 beh
            reg_beh.append("-stim_label")
            reg_beh.append(f"{c_beh + 1}")
            reg_beh.append(beh)

        elif decon_type == "TENT":

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

            # stim_time
            reg_beh.append("-stim_times")
            reg_beh.append(f"{c_beh + 1}")
            reg_beh.append(f"timing_files/{tf_dict[beh]}")
            reg_beh.append(f"""'TENT({",".join(tent_args)})'""")

            # stim_label
            reg_beh.append("-stim_label")
            reg_beh.append(f"{c_beh + 1}")
            reg_beh.append(beh)

    # set output str
    h_out = f"{phase}_{desc}"

    # build full decon command
    cmd_decon = f"""
        3dDeconvolve \\
            -x1D_stop \\
            -GOFORIT \\
            -input {" ".join(run_list)} \\
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


def func_motion(work_dir, phase, sub_num):

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
    if not os.path.exists(os.path.join(work_dir, f"censor_{phase}_combined.1D")):
        h_cmd = f"""
            module load afni-20.2.06
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

            cat out.cen.run-*{phase}.1D > outcount_{phase}_censor.1D

            1deval \
                -a motion_{phase}_censor.1D \
                -b outcount_{phase}_censor.1D \
                -expr "a*b" > censor_{phase}_combined.1D
        """
        h_mot = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_mot.wait()


def func_decon(work_dir, phase, time_files, decon_type, sub_num):

    """
    Step 2: Generate decon matrix

    Deconvolve script written for review.
    """

    # make list of pre-processed epi files
    run_list = [
        x.split(".")[0]
        for x in os.listdir(work_dir)
        if fnmatch.fnmatch(x, f"*{phase}_scale+tlrc.HEAD")
    ]
    run_list.sort()

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

    # write decon script for each phase of session
    #   desc = "single" is a place holder for when a session
    #   only has a single deconvolution
    if type(time_files) == list:

        desc = "single"

        # make timing file dictionary
        tf_dict = {}
        for tf in time_files:
            beh = tf.split("_")[-1].split(".")[0]
            tf_dict[beh] = tf

        # write decon script (for review)
        decon_script = os.path.join(work_dir, f"decon_{phase}_{desc}.sh")
        with open(decon_script, "w") as script:
            script.write(
                func_write_decon(
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

            tf_dict = {}
            for tf in time_files[desc]:
                beh = tf.split("_")[-1].split(".")[0]
                tf_dict[beh] = tf

            decon_script = os.path.join(work_dir, f"decon_{phase}_{desc}.sh")
            with open(decon_script, "w") as script:
                script.write(
                    func_write_decon(
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

    # gather scripts of phase
    script_list = [
        x for x in os.listdir(work_dir) if fnmatch.fnmatch(x, f"decon_{phase}*.sh")
    ]

    # run decon script to generate matrices
    for dcn_script in script_list:
        h_cmd = f"""
            module load afni-20.2.06
            cd {work_dir}
            source {os.path.join(work_dir, dcn_script)}
        """
        h_dcn = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_dcn.wait()


def func_reml(work_dir, phase, sub_num, time_files):

    """
    Step 3: Deconvolve

    3dREMLfit is used to do a GLS with an ARMA function.

    White matter time series is used as a nuissance regressor.
    """

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

    # run REML for each phase of session
    if type(time_files) == list:
        desc = "single"
        if not os.path.exists(
            os.path.join(work_dir, f"{phase}_{desc}_stats_REML+tlrc.HEAD")
        ):
            h_cmd = f"""
                cd {work_dir}
                tcsh \
                    -x {phase}_{desc}_stats.REML_cmd \
                    -dsort {phase}_WMe_rall+tlrc \
                    -GOFORIT
            """
            func_sbatch(h_cmd, 25, 4, 6, f"{sub_num}rml", work_dir)

    elif type(time_files) == dict:
        for desc in time_files:
            if not os.path.exists(
                os.path.join(work_dir, f"{phase}_{desc}_stats_REML+tlrc.HEAD")
            ):
                h_cmd = f"""
                    cd {work_dir}
                    tcsh \
                        -x {phase}_{desc}_stats.REML_cmd \
                        -dsort {phase}_WMe_rall+tlrc \
                        -GOFORIT
                """
                func_sbatch(h_cmd, 25, 4, 6, f"{sub_num}rml", work_dir)


# %%
# receive arguments
def func_argparser():
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("pars_subj", help="Subject ID")
    parser.add_argument("pars_sess", help="Session")
    parser.add_argument("pars_type", help="Decon Type")
    parser.add_argument("pars_dir", help="Derivatives Directory")
    return parser


def main():

    # capture passed args
    args = func_argparser().parse_args()
    subj = args.pars_subj
    sess = args.pars_sess
    decon_type = args.pars_type
    deriv_dir = args.pars_dir

    # set up
    work_dir = os.path.join(deriv_dir, subj, sess)
    sub_num = subj.split("-")[1]

    """ Get time dict """
    with open(os.path.join(work_dir, "decon_dict.json")) as json_file:
        decon_dict = json.load(json_file)

    """ Submit job for each phase """
    for phase in decon_dict:

        """ Make motion files """
        if not os.path.exists(os.path.join(work_dir, f"censor_{phase}_combined.1D")):
            func_motion(work_dir, phase, sub_num)

        """ Generate decon matrices """
        time_files = decon_dict[phase]
        decon_check = os.path.join(work_dir, f"X.{phase}_single.jpg")
        if not os.path.exists(decon_check):
            func_decon(work_dir, phase, time_files, decon_type, sub_num)

        """ Do Decon """
        reml_check = os.path.join(work_dir, f"{phase}_single_stats_REML+tlrc.HEAD")
        if not os.path.exists(reml_check):
            func_reml(work_dir, phase, sub_num, time_files)


if __name__ == "__main__":
    main()

# %%
