import os


def _find_cuda_runtime(start):
    """Walk up from start looking for a <dir>/cuda_runtime/bin folder.

    Works whether this file lives at the project root (normal case) or was
    copied into the venv's site-packages by scripts/get_cuda_runtime.ps1.
    """
    d = os.path.abspath(start)
    while True:
        candidate = os.path.join(d, "cuda_runtime", "bin")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _setup_path():
    """Prepend the project-local CUDA/cuDNN runtime DLL dir to PATH.

    TensorFlow 2.10 loads its CUDA DLLs (cudart64_110.dll, cudnn64_8.dll,
    ...) via PATH lookup. We keep those DLLs in cuda_runtime/bin (populated
    by scripts/get_cuda_runtime.ps1) so no system-wide CUDA install is
    needed. This module is auto-imported at interpreter startup because it
    is installed into the venv's site-packages.
    """
    dll_dir = _find_cuda_runtime(os.path.dirname(os.path.abspath(__file__)))
    if dll_dir and dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


_setup_path()
