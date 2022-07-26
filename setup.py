from setuptools import setup


setup(
    name="cheetah",
    version="0.5.13",
    author="Jan Kaiser & Oliver Stein",
    author_email="jan.kaiser@desy.de",
    description="Fast particle accelerator optics simulation for reinforcement learning and optimisation applications.",
    url="https://github.com/desy-ml/cheetah",
    packages=["cheetah"],
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "ocelot @ git+https://github.com/ocelot-collab/ocelot.git@21.12.1",
        "scipy"
    ]
)
