import numpy as np

from typing import List, Optional
from subprocess import run, CompletedProcess


def run_tool(tool: str, *inputs, **args) -> CompletedProcess:
    """
    Run a selected tool from the AFNI toolbax

    Parameters
    ----------
    tool : str
        The name of the tool to run
    *inputs : List[str]
        A list of strings to use as input arguments or files
    **args : Dict
        A dictionary of keyword arguments to pass to the tool

    Returns
    -------
    CompletedProcess
        The attributes associated with the process
    """
    cwd: Optional[str] = args.get("cwd")
    if cwd is not None:
        del args["cwd"]

    cmd_components: List[str] = []

    # Executable
    cmd_components.append(tool)

    # Arguments
    keyword_argument_pairs: List[List] = [
        [
            f"-{key}",
            value if type(value) is not list else " ".join(value)
        ] for key, value in args.items()
    ]
    keyword_arguments: List[str] = np.array(keyword_argument_pairs).flatten().tolist()
    cmd_components.extend(keyword_arguments)

    # Inputs
    cmd_components.extend(inputs)

    cmd_str: str = " ".join(cmd_components)

    proc: CompletedProcess = run(cmd_str, cwd=cwd, shell=True, check=True, capture_output=True, text=True)
    return proc
