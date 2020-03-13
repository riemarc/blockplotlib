import setuptools

description = ("Matplotlib-based library for block diagram creation.")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="blockplotlib",
    version="2020.1a",
    author="Marcus Riesmeier",
    author_email="marcus.riesmeier@umit.com",
    license="BSD 3-Clause License",
    description=description,
    long_description=description,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
)

