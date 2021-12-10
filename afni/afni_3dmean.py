from .cmd import run_tool


def afni_3dmean(*inputs, **args):
    """
    voxel-by-voxel average of all datasets in the inputs

    Returns
    -------
    str
        The output prefix
    """
    run_tool("3dMean", *inputs, **args)
