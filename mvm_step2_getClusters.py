# %%
import os
import math
import subprocess


# %%
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
                -bisided -3.4372 3.4372 \
                -pref_map Clust_{comp} \
                > Table_{comp}.txt
        """
        if not os.path.exists(os.path.join(grp_dir, f"Clust_{comp}+tlrc.HEAD")):
            h_clust = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
            h_clust.wait()


# %%
def main():

    # TODO update for multiple MVMs
    grp_dir = "/scratch/madlab/emu_power/analyses"
    mvm_dict = {"Hit-Miss": [2, 3], "CR-FA": [4, 5]}
    func_getClusters(grp_dir, mvm_dict)


if __name__ == "__main__":
    main()
