"""
Custom installer for ComfyUI_HeartMuLa.

Installs only the packages not already provided by ComfyUI.

On systems where fsspec >= 2026 is present we install with --no-deps to
avoid the pip resolver downgrading fsspec (torchtune -> datasets requires
fsspec <= 2025.10.0, which would break other tools on that system).
On a clean ComfyUI install fsspec will be at a compatible version so the
full resolver runs normally.
"""
import importlib.metadata
import subprocess
import sys


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def get_version(package: str) -> tuple:
    """Return the installed version as a tuple of ints, or (0,) if not found."""
    try:
        ver = importlib.metadata.version(package)
        return tuple(int(x) for x in ver.split(".")[:2])
    except importlib.metadata.PackageNotFoundError:
        return (0,)


def install() -> None:
    # These are not in ComfyUI and have no problematic transitive deps —
    # always safe to install normally.
    # imageio-ffmpeg ships a self-contained ffmpeg binary so the transcription
    # pipeline never needs a system ffmpeg on PATH.
    pip("soundfile", "bitsandbytes", "imageio-ffmpeg", "demucs")

    # vector-quantize-pytorch needs einx which is not in ComfyUI — install
    # with deps.
    pip("vector-quantize-pytorch")

    # torchao is a hard import inside torchtune/__init__.py — must be present.
    pip("torchao")

    # torchtune: its datasets transitive dep requires fsspec <= 2025.10.0.
    # If the system already has fsspec >= 2026, let the resolver skip deps
    # to avoid a harmful downgrade.  All torchtune runtime deps we actually
    # use are already present on such a system (they came with whatever
    # installed fsspec 2026 in the first place).
    if get_version("fsspec") >= (2026, 0):
        pip("--no-deps", "torchtune>=0.4.0")
    else:
        pip("torchtune>=0.4.0")


if __name__ == "__main__":
    install()
