from typing import List
from .cmd import run_tool


def afni_3dcopy(src: str, dst: str, **kwargs):
    """
    3dcopy

    Parameters
    ----------
    src : str
        Source data set
    dst : str
        Destination data set
    """
    inputs: List[str] = [src, dst]
    run_tool("3dcopy", *inputs, **kwargs)
