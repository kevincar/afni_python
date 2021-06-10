"""
Notes

Updated to use individual subject training models.
"""

# %%
import os
import numpy as np
import pandas as pd
import sys
import json
import subprocess

from mvpa_step1_setup import func_lenRun
from mvpa_step1_setup import func_timing
from mvpa_step1_setup import func_detrend


def func_test(dcn_str, subj_dir, mask):

    """
    Generates a test file consisting of only relevant volumes.
        This does not change the output, but I like it and it will
        be useful when splitting the phase into runs.

    Then, the classifier attempts to predict those volumes. The output files
        are categorical (class), and afni writes stats to stderr.
    """

    # Get only test volumes
    if not os.path.exists(os.path.join(subj_dir, f"MVPA_{dcn_str}_test+tlrc.HEAD")):

        # make new category file for only relevant volumes
        #   will be used for ROC
        df = pd.read_csv(
            os.path.join(subj_dir, f"MVPA_{dcn_str}_categories.txt"),
            sep=" ",
            header=None,
        )
        df.columns = ["cat"]
        df_new = df[df["cat"] != 9999]
        df_out = os.path.join(subj_dir, f"MVPA_{dcn_str}_cat_updated.txt")
        np.savetxt(df_out, df_new["cat"].values, fmt="%s", delimiter=" ")

        # make file of only relevant volumes                <---- NOTICE!
        #   this references the index of df_new, which
        #   is in volume/TR time, to extract only the
        #   volumes corresponding to the behaviors of
        #   test_dict[Foo][Test].
        #
        #   input file is output of func_detrend
        df_list = df_new.index.values.tolist()
        brick_list = [str(x) for x in df_list]
        h_cmd = f"""
            module load afni-20.2.06
            cd {subj_dir}
            3dTcat \
                -prefix MVPA_{dcn_str}_test \
                MVPA_{dcn_str}+tlrc[{",".join(brick_list)}]
        """
        h_tcat = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_tcat.wait()

    # Test
    if not os.path.exists(os.path.join(subj_dir, f"MVPA_pred_{dcn_str}.1D")):
        h_cmd = f"""
            module load afni-20.2.06
            cd {subj_dir}

            3dsvm \
                -testvol MVPA_{dcn_str}_test+tlrc \
                -model MVPA_train+tlrc \
                -mask {mask} \
                -testlabels MVPA_{dcn_str}_cat_updated.txt \
                -predictions MVPA_pred_{dcn_str} \
                -classout \
                2> MVPA_pred_{dcn_str}_acc.txt
        """
        h_test = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_test.wait()


# %%
def main():

    # # For testing
    # subj_dir = "/scratch/madlab/nate_vCAT/derivatives/sub-005/ses-S1"
    # test_dict = {"Study": {"Decon": ["fbl"], "Test": ["fblf", "fbls"]}}

    subj_dir = str(sys.argv[1])
    with open(os.path.join(subj_dir, "test_dict.json")) as json_file:
        test_dict = json.load(json_file)
    phase = list(test_dict.keys())[0]

    # get header info
    hdr_dict = func_lenRun(subj_dir, phase)

    # make test files
    test_list = test_dict[phase]["Test"]
    dcn_str = f"{phase}_single"
    func_timing(test_list, hdr_dict, subj_dir, phase, dcn_str)

    # make clean data for testing
    decon_list = test_dict[phase]["Decon"]
    if not os.path.exists(os.path.join(subj_dir, f"MVPA_{dcn_str}+tlrc.HEAD")):
        func_detrend(subj_dir, dcn_str, decon_list, hdr_dict)

    # test classifier
    mask = "/scratch/madlab/nate_vCAT/analyses/Group_GM_intersect_mask+tlrc"
    func_test(dcn_str, subj_dir, mask)


if __name__ == "__main__":
    main()
