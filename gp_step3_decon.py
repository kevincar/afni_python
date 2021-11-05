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
import re
import json
import fnmatch
import subprocess
from typing import Dict, List
from argparse import ArgumentParser, Namespace
from gp_step0_dcm2nii import func_sbatch
from gp_step1_preproc import print_process_output


def func_write_decon(
    run_list, tf_dict, cen_file, phase, decon_type, desc, work_dir, dmn_list, drv_list
):
    """
    Notes: This function generates a 3dDeconvolve command.
        It supports GAM, 2GAM, TENT, and dmBLOCK basis functions.
        TENT does not currently include duration.

    Parameters
    ----------
    run_list : List[str]
        A list of run files for a given block and subject
    tf_dict : Dict
        dictionary where keys are the behaviors and the values are the
        corresponding timing file names
    cen_file : ?
        ?
    phase : ?
        ?
    decon_type : ?
        ?
    desc : ?
        ?
    work_dir : str
        ?
    dmn_list : ?
        ?
    drv_list : ?
        ?
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
    switch_dict: Dict[str, List] = {
        "dmBLOCK": ["'dmBLOCK(1)'", "-stim_times_AM1"],
        "GAM": ["'GAM'", "-stim_times"],
        "2GAM": ["'TWOGAMpw(4,5,0.2,12,7)'", "-stim_times"],
        "BLOCK": [lambda d: f"'BLOCK({d}, 1)'", "-stim_times"],
    }

    reg_beh = []
    for c_beh, beh in enumerate(tf_dict):
        if decon_type in ["dmBLOCK", "GAM", "2GAM", "BLOCK"]:

            switch_data: List = switch_dict[decon_type]
            switch_cmd: str = switch_data[1]
            switch_rmodel: str = (
                switch_data[0] if decon_type != "BLOCK" else switch_data[0](5)
            )
            beh_count: int = c_beh + 1
            timing_file_name: str = tf_dict[beh]
            timing_file_path: str = os.path.join("timing_files", timing_file_name)

            # add stim_time info, order is
            #   -stim_times 1 tf_beh.txt basisFunction
            reg_beh.append(switch_cmd)
            reg_beh.append(f"{beh_count}")
            reg_beh.append(timing_file_path)
            reg_beh.append(switch_rmodel)

            # add stim_label info, order is
            #   -stim_label 1 beh
            reg_beh.append("-stim_label")
            reg_beh.append(f"{beh_count}")
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
            echo "motion_demean"
            1d_tool.py \
                -infile dfile_rall_{phase}.1D \
                -set_nruns {num_run} \
                -demean \
                -write \
                motion_demean_{phase}.1D || (echo "motion_demean failed"; exit 1)

            echo "motion_deriv"
            1d_tool.py \
                -infile dfile_rall_{phase}.1D \
                -set_nruns {num_run} \
                -derivative \
                -demean \
                -write \
                motion_deriv_{phase}.1D || (echo "motion_deriv failed"; exit 1)

            echo "mot_demean"
            # split into runs
            1d_tool.py \
                -infile motion_demean_{phase}.1D \
                -set_nruns {num_run} \
                -split_into_pad_runs \
                mot_demean_{phase} || (echo "mot_demeain failed" && exit 1)

            echo "mot_deriv"
            1d_tool.py \
                -infile motion_deriv_{phase}.1D \
                -set_nruns {num_run} \
                -split_into_pad_runs \
                mot_deriv_{phase} || (echo "mot_deriv failed" && exit 1)

            # make censor file
            1d_tool.py \
                -infile dfile_rall_{phase}.1D \
                -set_nruns {num_run} \
                -show_censor_count \
                -censor_prev_TR \
                -censor_motion 0.3 \
                motion_{phase} || (echo "motion_phase failed" && exit 1)

            cat out.cen.run-*{phase}.1D > outcount_{phase}_censor.1D

            1deval \
                -a motion_{phase}_censor.1D \
                -b outcount_{phase}_censor.1D \
                -expr "a*b" > censor_{phase}_combined.1D
        """
        h_mot = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        print_process_output(h_mot)
        h_mot.wait()
        if h_mot.returncode != 0:
            raise Exception(f"func_motion failed for subject {sub_num}")


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
    print(drv_list)

    # write decon script for each phase of session
    #   desc = "single" is a place holder for when a session
    #   only has a single deconvolution
    if type(time_files) == list:
        desc = "single"

        # # make timing file dictionary
        tf_dict: Dict[str, str] = {}
        for tf in time_files:
            print(tf)
            tf_name: str = os.path.splitext(tf)[0]
            beh: str = tf_name.split("_")[-1]
            tf_dict[beh] = tf

        # write decon script (for review)
        decon_script_name: str = f"decon_{phase}_{desc}.sh"
        decon_script: str = os.path.join(work_dir, decon_script_name)
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
    else:
        raise Exception(
            f"time_file of type {type(time_files)} unsupported. Must be dict or list"
        )

    # gather scripts of phase
    script_list = [
      x for x in os.listdir(work_dir) if fnmatch.fnmatch(x, f"decon_{phase}*.sh")
    ]

    # run decon script to generate matrices
    for dcn_script in script_list:
        dcn_script_path: str = os.path.join(work_dir, dcn_script)
        h_cmd = f"""
        module load afni-20.2.06
        cd {work_dir}
        source {dcn_script_path}
        """
        h_dcn = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        print_process_output(h_dcn)
        h_dcn.wait()
        if h_dcn.returncode != 0:
            raise Exception(f"dcn_script {dcn_script} failed")


def func_reml(work_dir, phase, sub_num, time_files):

    """
    Step 3: Deconvolve

    3dREMLfit is used to do a GLS with an ARMA function.

    White matter time series is used as a nuissance regressor.
    """

    # generate WM timeseries
    WM_signal_file_name: str = f"{phase}_WMe_rall+tlrc.HEAD"
    WM_signal_file_path: str = os.path.join(work_dir, WM_signal_file_name)
    if not os.path.exists(WM_signal_file_path):
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
        if os.environ.get("SLURM") is not None:
            func_sbatch(h_cmd, 1, 4, 1, f"{sub_num}wts", work_dir)
        else:
            proc_wm = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            print_process_output(proc_wm)
            proc_wm.wait()
            if proc_wm.returncode != 0:
                raise Exception("WM timeseries failed")

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
            if os.environ.get("SLURM") is not None:
                func_sbatch(h_cmd, 25, 4, 6, f"{sub_num}rml", work_dir)
            else:
                proc_reml = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
                print_process_output(proc_reml)
                proc_reml.wait()
                if proc_reml.returncode != 0:
                    raise Exception("REML failed")

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

def generate_outputs():
    """
    generate relavant output files for the deconvolution process
    """

# %%
# receive arguments
def func_argparser():
    parser = ArgumentParser("Receive Bash args from wrapper")
    parser.add_argument("pars_subj", help="Subject ID")
    parser.add_argument("pars_sess", help="Session")
    parser.add_argument("pars_type", help="Decon Type")
    parser.add_argument("pars_dir", help="Derivatives Directory")
    parser.add_argument("decon_dict_path", help="Path to decon dict")
    return parser


def main():

    # capture passed args
    args: Namespace = func_argparser().parse_args()
    subj: str = args.pars_subj
    sess: str = args.pars_sess
    decon_type: str = args.pars_type
    deriv_dir: str = args.pars_dir
    decon_dict_path: str = args.decon_dict_path

    # set up
    subj_dir: str = os.path.join(deriv_dir, subj)
    work_dir: str = os.path.join(subj_dir, sess)
    sub_num: int = int(subj.split("-")[1])

    # """ Get time dict """
    decon_dict: Dict
    with open(decon_dict_path, "r") as json_file:
        decon_dict = json.load(json_file)

    # """ Submit job for each phase """
    for phase in decon_dict:

        """Make motion files"""
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
