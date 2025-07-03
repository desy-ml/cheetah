import os
from pathlib import Path

import requests
import trimesh
from tqdm import tqdm

import cheetah

# This follows the way PyTorch and Torchvision download model weights and store them in
# a cache directory, which is typically something like `~/.cache/cheetah`.
# See https://github.com/pytorch/pytorch/blob/main/torch/hub.py and
# https://github.com/pytorch/vision/blob/main/torchvision/models/_api.py for reference.


ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
ENV_CHEETAH_CACHE_DIR = "CHEETAH_CACHE_DIR"


def get_cheetah_cache_dir() -> Path:
    """Get the path to the Cheetah cache directory."""
    system_cache_dir = Path(
        os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR)
    ).expanduser()
    return system_cache_dir / "cheetah"


def get_repository_raw_url() -> str:
    """
    Get base URL for raw files in the Cheetah repository.

    Points to the correct version of the repository based on the current Cheetah
    version. Points to the `master` branch if a development version is used. (Signified
    by "-dev" in the version string.)
    """
    version_tag = (
        f"v{cheetah.__version__}" if "-dev" not in cheetah.__version__ else "master"
    )
    return f"https://raw.githubusercontent.com/desy-ml/cheetah/{version_tag}"


def download_url_to_file(
    source_url: str, destination_path: Path, show_progress: bool = True
) -> None:
    """
    Download a file from a URL to a local path.

    :param source_url: The URL to download the file from.
    :param destination_path: The local path where the file should be saved.
    :param show_progress: If True, show a progress bar during the download.
    """
    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    # Get the total file size from headers
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB

    with open(destination_path, "wb") as f, tqdm(
        total=total_size, disable=not show_progress, unit="iB", unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            pbar.update(len(chunk))


def load_3d_asset(name: str, show_download_progress: bool = True) -> trimesh.Trimesh:
    """
    Get a 3D asset by file name.

    :param name: The name of the asset file (e.g., "Quadrupole.glb").
    :param show_download_progress: If True and the asset is not cached, show a progress
        bar during the download.
    :return: A trimesh.Trimesh object representing the 3D asset.
    """
    assets_dir = get_cheetah_cache_dir() / "assets" / "3d"
    asset_path = assets_dir / name

    if not asset_path.exists():
        asset_url = f"{get_repository_raw_url()}/assets/3d/{name}"
        download_url_to_file(
            asset_url, asset_path, show_progress=show_download_progress
        )

    return trimesh.load_mesh(asset_path, file_type="glb")
