# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd
import subprocess
import os
import pathlib


def get_include_options():
    """Get the source code include path for HOOMD's include files."""
    current_module_path = pathlib.Path(hoomd.__file__).parent.resolve()
    build_module_path = (pathlib.Path(hoomd.version.build_dir)
                         / 'hoomd').resolve()

    # use the source directory if this module is in the build directory
    if current_module_path == build_module_path:
        hoomd_include_path = pathlib.Path(hoomd.version.source_dir)
    else:
        # otherwise, use the installation directory
        hoomd_include_path = current_module_path / 'include'

    return ['-I', str(hoomd_include_path.resolve())]


def to_llvm_ir(code, clang_exec):
    r"""Helper function to compile the provided code into an executable.

    Args:
        code (`str`): C++ code to compile
        clang_exec (`str`): The Clang executable to use
        fn (`str`, **optional**): If provided, the code will be written to a
        file.
    """
    hoomd_include_path = os.path.dirname(hoomd.__file__) + '/include'

    cmd = [
        clang_exec, '-O3', '--std=c++14', '-DHOOMD_LLVMJIT_BUILD', '-I',
        hoomd_include_path, '-S', '-emit-llvm', '-x', 'c++', '-o', '-', '-'
    ]
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    # pass C++ function to stdin
    output = p.communicate(code.encode('utf-8'))
    llvm_ir = output[0].decode()

    if p.returncode != 0:
        raise RuntimeError(f"Error initializing potential: {output[1]}")

    return llvm_ir


def get_gpu_compilation_settings(gpu):
    """Helper function to set CUDA libraries for GPU execution."""
    includes = [
        "-I" + os.path.dirname(hoomd.__file__) + '/include',
        "-I" + os.path.dirname(hoomd.__file__)
        + '/include/hoomd/extern/HIP/include',
        "-I" + hoomd.version.cuda_include_path,
    ]

    # compile JIT code for the current device
    compute_major, compute_minor = gpu.compute_capability
    return {
        "includes": includes,
        "cuda_devrt_lib_path": hoomd.version.cuda_devrt_library,
        "max_arch": compute_major * 10 + compute_minor
    }
