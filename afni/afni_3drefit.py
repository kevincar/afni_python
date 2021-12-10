from .cmd import run_tool


def afni_3drefit(*inputs, **kwargs):
    """
    3drefit
    """
    run_tool("3drefit", *inputs, **kwargs)
