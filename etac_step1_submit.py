import os
from datetime import datetime
import subprocess

# set up
code_dir = "/home/nmuncy/compute/learn_mvpa"
parent_dir = "/scratch/madlab/nate_vCAT"


def main():

    # set up stdout/err capture
    current_time = datetime.now()
    out_dir = os.path.join(
        parent_dir,
        f'derivatives/Slurm_out/ET1_{current_time.strftime("%H%M_%d-%m-%y")}',
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    h_out = os.path.join(out_dir, "out_etac.txt")
    h_err = os.path.join(out_dir, "err_etac.txt")

    sbatch_job = f"""
        sbatch \
            -J "ETACp" -t 45:00:00 --mem=4000 --ntasks-per-node=1 \
            -p IB_44C_512G  -o {h_out} -e {h_err} \
            --account iacc_madlab --qos pq_madlab \
            --wrap="module load python-3.7.0-gcc-8.2.0-joh2xyk \n \
            python {code_dir}/etac_step1_test.py"
    """
    sbatch_submit = subprocess.Popen(sbatch_job, shell=True, stdout=subprocess.PIPE)
    job_id = sbatch_submit.communicate()[0]
    print(job_id.decode("utf-8"))


if __name__ == "__main__":
    main()
