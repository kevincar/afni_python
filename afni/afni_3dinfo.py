from typing import Any
from subprocess import CompletedProcess
from .cmd import run_tool


def afni_3dinfo(*inputs, **kwargs) -> Any:
    """3dinfo
    """
    proc: CompletedProcess = run_tool("3dinfo", *inputs, **kwargs)
    return proc.stdout
