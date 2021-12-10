from typing import Optional, List
from subprocess import run, CompletedProcess, PIPE


def c3d(*inputs, **kwargs):
    """
    Convert3D
    """
    cwd: Optional[str] = kwargs.get("cwd")
    if cwd is not None:
        del kwargs["cwd"]

    command_elements: List[str] = []

    # Executable
    command_elements.append("c3d")

    # Inputs
    command_elements.extend(inputs)

    # Keyword arguments
    for key, value in kwargs.items():
        command_elements.extend([
            f"-{key}",
            value if type(value) is not list else " ".join(value)
        ])

    command: str = " ".join(command_elements)

    proc: CompletedProcess = run(
        command,
        cwd=cwd,
        shell=True,
        check=True,
        stdout=PIPE
    )
    proc
