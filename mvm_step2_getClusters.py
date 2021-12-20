# %%
import os
import json
import math
import subprocess

from typing import Dict
from argparse import ArgumentParser, Namespace


def func_getClusters(grp_dir, mvm_dict):

    # get K threshold from p < .001, NN=1, 2sided
    h_cmd = f"""
        head -n 136 {grp_dir}/MC_thresholds.txt | \
            tail -n 1 | \
            awk '{{print $7}}'
    """
    h_thr = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
    thr_num = math.ceil(float(h_thr.communicate()[0].decode("utf-8")))

    # extract sig clusters
    for comp in mvm_dict:
        h_cmd = f"""
            module load afni-20.2.06
            cd {grp_dir}

            3dClusterize \
                -nosum \
                -1Dformat \
                -inset MVM+tlrc \
                -idat {mvm_dict[comp][0]} \
                -ithr {mvm_dict[comp][1]} \
                -NN 1 \
                -clust_nvox {thr_num} \
                -bisided p=0.001 \
                -pref_map Clust_{comp} \
                > Table_{comp}.txt
        """
        if not os.path.exists(os.path.join(grp_dir, f"Clust_{comp}+tlrc.HEAD")):
            h_clust = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_clust.wait()


def main():

    # Setup argument
    parser: ArgumentParser = ArgumentParser(prog="getClusers")
    parser.add_argument("-g", "--group", required=True, help="Path to the group analysis directory")
    parser.add_argument("-d", "--data", required=True, help="Path to the data directory")

    # Capture arguments
    args: Namespace = parser.parse_args()
    group_dir: str = args.group
    data_dir: str = args.data
    docs_dir: str = os.path.join(data_dir, "docs")

    # Create the mvm_dict using the glt_dict
    glt_dict_filename: str = "glt_dict.json"
    glt_dict_filepath: str = os.path.join(docs_dir, glt_dict_filename)

    glt_dict: Dict
    with open(glt_dict_filepath, "r") as fh:
        glt_dict = json.load(fh)

    start_brik: int = 4
    cur_brik: int = start_brik
    mvm_dict: Dict = {}
    for comparison in glt_dict:
        mvm_dict.update({
            comparison: [cur_brik, cur_brik + 1]
        })
        cur_brik += 2

    # TODO update for multiple MVMs
    # grp_dir = "/scratch/madlab/emu_power/analyses"
    # mvm_dict = {"Hit-Miss": [2, 3], "CR-FA": [4, 5]}
    func_getClusters(group_dir, mvm_dict)


if __name__ == "__main__":
    main()
