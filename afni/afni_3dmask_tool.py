from .cmd import run_tool


def afni_3dmask_tool(*inputs, **args):
    """
    combine masks
    """

    run_tool("3dmask_tool", *inputs, **args)
