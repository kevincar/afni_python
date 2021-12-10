from .cmd import run_tool


def afni_3dresample(*inputs, **kwargs):
    """
    3dresample
    """
    run_tool("3dresample", *inputs, **kwargs)
