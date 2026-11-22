"""Runtime contract shared by the runner, GPU workers and evidence importer."""
import re

TRITON_VERSION = "3.9.0"
UPSTREAM_COMMIT = "b7ddcf86c095de79527ba3098503e73158bb94bd"


def matches_triton(version):
    return isinstance(version, str) and version.partition("+")[0] == TRITON_VERSION


def require_triton():
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError("Triton 3.9.0 is required in the selected SEAM .venv") from exc
    if not matches_triton(triton.__version__):
        raise RuntimeError(f"Expected Triton {TRITON_VERSION}, imported {triton.__version__} from {triton.__file__}")
    return triton.__version__


def toolkit_release(banner):
    # Toolkit, torch.version.cuda and the driver's supported CUDA are separate fields.
    match = re.search(r"release\s+(\d+)\.(\d+)", banner)
    return tuple(map(int, match.groups())) if match else ()
