from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cheetah-accelerator",
    version="0.5.14",
    author="Jan Kaiser & Oliver Stein",
    author_email="jan.kaiser@desy.de",
    url="https://github.com/desy-ml/cheetah",
    description="Fast particle accelerator optics simulation for reinforcement learning and optimisation applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["cheetah"],
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "ocelot @ git+https://github.com/ocelot-collab/ocelot.git@21.12.1",
        "scipy",
    ],
)
