from setuptools import setup

setup(name="cheetah",
      version="0.4.0",
      author="Jan Kaiser & Oliver Stein",
      author_email="jan.kaiser@desy.de | oliver.stein@desy.de",
      description="Fast particle accelerator optics simulation for reinforcement learning and optimisation applications.",
      url="https://github.com/desy-ml/cheetah",
      install_requires=["numpy>=1.18.5",
                        "ocelot>=20.11.2",
                        "scipy>=1.6.0"])
