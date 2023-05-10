[![name](https://img.shields.io/pypi/v/blockplotlib?label=pypi%20package)](https://pypi.org/project/blockplotlib)
[![name](https://img.shields.io/pypi/dm/blockplotlib)](https://pypi.org/project/blockplotlib)

# Blockplotlib

Matplotlib-based library for the creation of block diagrams.
Besides this core functionality, blockplotlib provides a workflow
to include all kinds of matplotlib figures with the correct font size
into latex documents. While latex is also often used to render text
inside figures/block diagrams, blockplotlib (as matplotlib)
depends not on latex.

The functionality of the library is limited to the features
demonstrated in the examples, but can be extended quite easily.

## Examples

How to draw a simple control loop is illustrated with the
example `blockplotlib/examples/control_loop_one_dof.py`.

[]()

A more complex example is the block diagram of the
so-called hyperbolic observer canonical form:
`blockplotlib/examples/hyperbolic_ocf.py`

[]()

The inclusion of matplotlib figures with the correct
font size into a latex document can be adapted
from the example under `blockplotlib/examples/tex_article`.
It basically relies on two measurements of the height of a string.
One measurement is performed in the latex preamble and the
other one in the respective matplotlib figure using
a blockplotlib function.

More example can be found under `blockplotlib/examples/`.

## Installation

One way to use blockplotlib is to install it via
pip (`pip install blockplotlib`).
Another way to use it is to copy the file `blockplotlib.py` in the folder
of the Python script from which it is to be imported.