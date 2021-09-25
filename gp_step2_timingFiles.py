import os
import json
import shutil
import subprocess
import pandas as pd

from typing import List, Dict
from argparse import ArgumentParser, Namespace
from gp_step1_preproc import print_process_output


def main():
    # Setup the argument parser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("subj", help="Subject ID")
    parser.add_argument("parDir", help="Parent Directory Path")
    parser.add_argument(
        "-d", "--decon_dict", help="Path to deconvolution dictionary", required=True
    )

    # Parse the arguments
    args: Namespace = parser.parse_args()
    subj: str = args.subj
    par_dir: str = args.parDir
    decon_dict_path: str = args.decon_dict

    # Setup working variables
    subj_id: int = int(subj.split("-")[-1])
    deriv_dir: str = os.path.join(par_dir, "derivatives")
    work_dir: str = os.path.join(deriv_dir, subj)
    stimulus_dir: str = os.path.join(par_dir, "stimuli", "stim_vectors")
    timing_dir: str = os.path.join(work_dir, "timing_files")
    docs_dir: str = os.path.join(par_dir, "docs")

    # Process the stimulus code
    stimulus_order_file_name: str = "stim_order.txt"
    stimulus_order_file_path: str = os.path.join(docs_dir, stimulus_order_file_name)
    stimulus_order_data: pd.DataFrame = pd.read_csv(
        stimulus_order_file_path, sep=" ", names=["subject", "order"]
    )
    filtered_stimulus_order_data: pd.DataFrame = stimulus_order_data.query(
        f"subject == '{subj_id}'"
    )
    stim_order: int = filtered_stimulus_order_data["order"].iloc[0]
    stim_map: Dict = {132: 1, 231: 2, 123: 3, 213: 4, 312: 5, 321: 6}

    # Load the file prefix based on the deconvolution file
    decon_dict: Dict
    with open(decon_dict_path, "r") as fh:
        decon_dict = json.load(fh)

    phases: List[str] = list(decon_dict.keys())
    phase: str = phases[0]
    file_prefix: str = f"tf_{phase}"
    time_file_names: List[str] = decon_dict[phase]
    behaviors: List[str] = [
        os.path.splitext(time_file_name)[0].split("_")[-1]
        for time_file_name in time_file_names
    ]

    result_file_name: str = time_file_names[0]
    result_file_path: str = os.path.join(timing_dir, result_file_name)
    if not os.path.exists(result_file_path):
        os.mkdir(timing_dir)

        # Copy over the appropriate stimulus vector file
        vect_file_id: int = stim_map[stim_order]
        vect_file_name: str = f"BeVect{vect_file_id}.txt"
        vect_file_path: str = os.path.join(stimulus_dir, vect_file_name)
        dst_file_name: str = "tmp_behVect.txt"
        dst_path: str = os.path.join(timing_dir, dst_file_name)
        shutil.copy(vect_file_path, dst_path)

        # make timing files
        command: str = f"""
        make_stim_times.py \
                -files {dst_path} \
                -prefix {file_prefix} \
                -tr 5 \
                -nruns 3 \
                -nt 78 \
                -offset 0
        """
        proc: subprocess.Popen = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE
        )
        print_process_output(proc)
        proc.wait()
        if proc.returncode != 0:
            raise Exception("make_stim filed")

        os.remove(dst_path)

        # Resulting files will have double digit numbers at their end. Replace
        # these with the behabiors
        # Rename time files to include behavior in the name
        for idx, beh in enumerate(behaviors):
            file_no: int = idx + 1
            idx_str: str = f"{file_no:02}"
            src_beh_file_name: str = f"{file_prefix}.{idx_str}.1D"
            dst_beh_file_name: str = f"{file_prefix}_{beh}.1D"
            src_beh_file_path: str = os.path.join(timing_dir, src_beh_file_name)
            dst_beh_file_path: str = os.path.join(timing_dir, dst_beh_file_name)
            shutil.move(src_beh_file_path, dst_beh_file_path)

        if not os.path.exists(result_file_path):
            raise Exception("Problem with making timing files")


if __name__ == "__main__":
    main()
