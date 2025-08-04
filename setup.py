from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def get_version():
    version_file_path = this_directory / "cheetah" / "_version.py"
    version_file_content = version_file_path.read_text()
    for line in version_file_content.splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="cheetah-accelerator",
    version=get_version(),
    author="Jan Kaiser, Chenran Xu and Christian Hespe",
    author_email="jan.kaiser@desy.de",
    url="https://github.com/desy-ml/cheetah",
    description=(
        "Fast and differentiable particle accelerator optics simulation for"
        " reinforcement learning and optimisation applications."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[package for package in find_packages() if package.startswith("cheetah")],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.5",
        "numpy>=1.21.3",
        "scipy>=1.7.2",
        "torch[opt-einsum]>=2.3",
    ],
    extras_require={
        "openpmd": ["openpmd-beamphysics"],
        "3d-visualization": ["trimesh>=4.4.0", "requests>=2.32.2", "tqdm>=4.66.0"],
    },
)
