from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cheetah-accelerator",
    version="0.7.4",
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
        "3d-visualization": ["trimesh>=4.0.0"],
    },
)
