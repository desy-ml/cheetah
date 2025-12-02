import os
from pathlib import Path

import requests
import trimesh
from tqdm import tqdm

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


def get_repository_raw_url(owner: str, repository: str, branch_or_tag: str) -> str:
    """
    Get the base URL for the root of the raw content of a GitHub repository.

    :param owner: The owner of the GitHub repository. Can be a user or an organisation.
    :param repository: The name of the GitHub repository.
    :param branch_or_tag: The branch or tag name for the version of the repository.
    :return: The base URL for the raw content of the repository.
    """
    return f"https://raw.githubusercontent.com/{owner}/{repository}/{branch_or_tag}"


def get_latest_release_tag(owner: str, repository: str) -> str:
    """
    Get the latest release tag of a GitHub repository.

    :param owner: The owner of the GitHub repository. Can be a user or an organisation.
    :param repository: The name of the GitHub repository.
    :return: The latest release tag of the repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repository}/releases/latest"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if request failed
    return response.json()["tag_name"]


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

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open(destination_path, "wb") as f, tqdm(
        total=total_size,
        disable=not show_progress,
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {source_url.split('/')[-1]}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            pbar.update(len(chunk))


def load_3d_asset(
    name: str, show_download_progress: bool = True
) -> trimesh.Trimesh | None:
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
        # latest_release_tag = get_latest_release_tag(
        #     owner="desy-ml", repository="3d-assets"
        # )
        asset_repository_url = get_repository_raw_url(
            owner="desy-ml", repository="3d-assets", branch_or_tag="v1.0.2"
        )
        asset_url = f"{asset_repository_url}/{name}"
        try:
            download_url_to_file(
                asset_url, asset_path, show_progress=show_download_progress
            )
        except requests.HTTPError:
            return None

    return trimesh.load_mesh(asset_path, file_type="glb")
