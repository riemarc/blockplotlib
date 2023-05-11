import setuptools
import os

description = "Matplotlib-based library for the creation of block diagrams"

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()

long_description = read_file("readme.md")
requirements = read_file("requirements.txt").splitlines()

setuptools.setup(
    name="blockplotlib",
    version="2023.1.4",
    url="https://github.com/riemarc/blockplotlib",
    author="Marcus Riesmeier",
    author_email="gluehen-sierren-0c@icloud.com",
    license="BSD 3-Clause License",
    description=description,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
)

