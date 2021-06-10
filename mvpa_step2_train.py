"""
Notes

SVM train on each subject.
"""

# %%
import os
import sys
import subprocess


# %%
def main():

    subj_dir = str(sys.argv[1])
    train_str = str(sys.argv[2])
    group_dir = str(sys.argv[3])

    # Train on voxels identified by ETAC (Face vs Scene)
    mask = os.path.join(group_dir, "FINAL_loc_face-scene+tlrc")

    if not os.path.exists(os.path.join(subj_dir, "MVPA_train+tlrc.HEAD")):
        h_cmd = f"""
            module load afni-20.2.06
            cd {subj_dir}

            3dsvm \
                -trainvol {train_str}+tlrc \
                -trainlabels {train_str}_categories.txt \
                -mask {mask} \
                -model MVPA_train \
                > MVPA_train_acc.txt 2>&1
        """
        h_trn = subprocess.Popen(h_cmd, shell=True, stdout=subprocess.PIPE)
        h_trn.wait()


if __name__ == "__main__":
    main()

# %%
