"""
Notes
"""

# %%
import os
import sys
import json
import fnmatch
import subprocess
import pandas as pd
import numpy as np


# %%
def func_lenRun(subj_dir, phase):

    """
    Returns a dictionary that contains TR length,
        number of volumes in the first run, number
        of runs in phase, and the length (sec) of a run.
    """

    # determine number of volumes
    h_cmd = f"""
        module load afni-20.2.06
        3dinfo -ntimes {subj_dir}/run-1_{phase}_scale+tlrc
    """
    h_nvol = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    num_nvol = int(h_nvol.communicate()[0].decode("utf-8").strip())

    # determine TR len
    h_cmd = f"""
        module load afni-20.2.06
        3dinfo -tr {subj_dir}/run-1_{phase}_scale+tlrc
    """
    h_tr = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    len_tr = float(h_tr.communicate()[0].decode("utf-8").strip())

    # determine number of runs
    num_runs = len(
        [
            x
            for x in os.listdir(subj_dir)
            if fnmatch.fnmatch(x, f"*{phase}*scale+tlrc.HEAD")
        ]
    )

    # determine length of e/run
    len_run = float(num_nvol * len_tr)

    tmp_dict = {
        "LenTR": len_tr,
        "NumVol": num_nvol,
        "NumRun": num_runs,
        "LenRun": len_run,
    }
    return tmp_dict


def func_timing(
    beh_list,
    hdr_dict,
    subj_dir,
    phase,
    dcn_str,
):
    """
    Mines timing files (tf_phase_foo.txt) to construct
        (a) category file, used for training, and
        (b) matrix, used for verification.

    Files have one row per volume of phase, and category ints
        are 1-indexed from train_dict behaviors (0 = baseline)
    """

    # convert timing files to 1D in volume time
    timing_dir = os.path.join(subj_dir, "timing_files")

    for beh in beh_list:

        # determine behavior duration, get 1st number in column 0
        df_dur = pd.read_csv(
            os.path.join(timing_dir, f"dur_{phase}_{beh}.txt"), header=None
        )
        beh_dur = df_dur[0][1]

        # make 1D file per run
        h_cmd = f"""
            module load afni-20.2.06
            timing_tool.py \
                -timing {timing_dir}/tf_{phase}_{beh}.txt \
                -tr {hdr_dict["LenTR"]} \
                -stim_dur {beh_dur} \
                -run_len {hdr_dict["LenRun"]} \
                -timing_to_1D \
                {subj_dir}/tmp_tf_{phase}_{beh}.1D
        """
        h_spl = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_spl.wait()

    # make attribute df - combine columns
    tf_list = []
    for beh in beh_list:
        h_tf = f"tmp_tf_{phase}_{beh}.1D"
        if os.path.exists(os.path.join(subj_dir, h_tf)):
            tf_list.append(h_tf)

    df_att = pd.concat(
        [
            pd.read_csv(
                os.path.join(subj_dir, item),
                names=[item.split("_")[3].split(".")[0]],
            )
            for item in tf_list
        ],
        axis=1,
    )
    column_list = list(df_att)

    # new attribute (att) column, fill with column name,
    #   remove rows/vols w/multiple behaviors, fill
    #   NaN with "base"
    for beh in beh_list:
        df_att.loc[df_att[beh] == 1, "att"] = beh

    for row, col in enumerate(df_att["att"]):
        if df_att[column_list].iloc[row].sum() > 1:
            df_att.at[row, "att"] = "base"
    df_att = df_att.replace(np.nan, "base", regex=True)

    # add category (cat) column for afni, 0 = base
    #   update - censor 0 via 9999
    if "base" not in beh_list:
        beh_list.insert(0, "base")
    for cat, beh in enumerate(beh_list):
        df_att.loc[df_att["att"] == beh, "cat"] = cat
    df_att["cat"] = df_att["cat"].replace([0], 9999)
    df_att["cat"] = df_att["cat"].astype(int)

    # write categories, and matrix (for checking)
    h_out = os.path.join(subj_dir, f"MVPA_{dcn_str}_categories.txt")
    np.savetxt(h_out, df_att["cat"].values, fmt="%s", delimiter=" ")
    h_df = os.path.join(subj_dir, f"MVPA_{dcn_str}_matrix.txt")
    np.savetxt(h_df, df_att.values, fmt="%s", delimiter=" ")


def func_detrend(subj_dir, dcn_str, beh_list, hdr_dict):

    """
    Extracts non-baseline sub-bricks from decon_cbucket_REML+tlrc to
        generate MVPA_phase_foo+tlrc
    """

    # get relevant column numbers
    #   read design matrix, find column labels,
    #   then look for column label in train_dict
    #   to get a brick_list of only effects of interest
    #
    #   will produce appropriate sub-bricks for cbucket file
    brick_list = []
    with open(os.path.join(subj_dir, f"X.{dcn_str}.xmat.1D")) as f:
        h_file = f.readlines()
        for line in h_file:
            if line.__contains__("ColumnLabels"):
                col_list = line.split('"')[1].replace(" ", "").split(";")
                for i, j in enumerate(col_list):
                    if j.split("#")[0] in beh_list:
                        brick_list.append(f"{str(i)}")

    # determine correct number of sub-bricks
    #   decon adds an extra for model
    h_cmd = f"""
            module load afni-20.2.06
            3dinfo -nv {subj_dir}/{dcn_str}_cbucket_REML+tlrc
        """
    h_len = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    len_wrong = h_len.communicate()[0].decode("utf-8").strip()
    len_right = int(len_wrong) - 2

    # make detrended data
    #   only contains tent data of non-pol/mot
    #   sub-bricks
    h_cmd = f"""
        module load afni-20.2.06
        cd {subj_dir}

        3dTcat \
            -prefix tmp_{dcn_str}_cbucket \
            -tr {hdr_dict["LenTR"]} \
            "{dcn_str}_cbucket_REML+tlrc[0..{len_right}]"

        3dSynthesize \
            -prefix MVPA_{dcn_str} \
            -matrix X.{dcn_str}.xmat.1D \
            -cbucket tmp_{dcn_str}_cbucket+tlrc \
            -select {" ".join(brick_list)} \
            -cenfill nbhr
    """
    h_cl = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    h_cl.wait()


# %%
def main():

    """
    Make category files and detrended EPI data.

    Output will be titled MVPA_foo
    """

    # # For Testing
    # work_dir = "/scratch/madlab/nate_vCAT/derivatives"
    # subj = "sub-005"
    # sess = "ses-S1"
    # train_dict = {"loc": ["face", "scene"]}

    # Receive args
    subj = str(sys.argv[1])
    sess = str(sys.argv[2])
    work_dir = str(sys.argv[3])

    # start work
    subj_dir = os.path.join(work_dir, subj, sess)

    # Get train dict, determine phase, beh/timing files
    with open(os.path.join(subj_dir, "train_dict.json")) as json_file:
        train_dict = json.load(json_file)
    phase = list(train_dict.keys())[0]
    beh_list = train_dict[phase]

    # get header info
    hdr_dict = func_lenRun(
        subj_dir,
        phase,
    )

    # set out str
    dcn_str = f"{phase}_single"

    # make category file:
    #   vector of definitions for training
    #   1 = beh_list[0], 2 = beh_list[1],
    #   9999 = other times
    if not os.path.exists(os.path.join(subj_dir, f"MVPA_{dcn_str}_categories.txt")):
        func_timing(
            beh_list,
            hdr_dict,
            subj_dir,
            phase,
            dcn_str,
        )

    # make detrended data
    if not os.path.exists(os.path.join(subj_dir, f"MVPA_{dcn_str}+tlrc.HEAD")):
        func_detrend(subj_dir, dcn_str, beh_list, hdr_dict)


if __name__ == "__main__":
    main()
