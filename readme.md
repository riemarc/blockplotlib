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
demonstrated in the examples but can be extended quite easily.

## Examples

How to draw a simple control loop is illustrated with the
example `blockplotlib/examples/control_loop_one_dof.py`.

![control_loop_one_dof](https://github.com/riemarc/blockplotlib/assets/18379817/7be3ba47-7e22-4965-98a3-467b6b47969b)
[control_loop_one_dof.pdf](https://github.com/riemarc/blockplotlib/files/11445397/control_loop_one_dof.pdf)

A more complex example is the block diagram of the
so-called hyperbolic observer canonical form:
`blockplotlib/examples/hyperbolic_ocf.py`

![hyperbolic_ocf](https://github.com/riemarc/blockplotlib/assets/18379817/5148f97a-12b8-4c1f-af6f-6d4b741f22c1)
[hyperbolic_ocf.pdf](https://github.com/riemarc/blockplotlib/files/11445461/hyperbolic_ocf.pdf)

The inclusion of matplotlib figures with the correct
font size into a latex document can be adapted
from the example under `blockplotlib/examples/tex_article`.
It basically relies on two measurements of the height of a string.
One measurement is performed in the latex preamble and the
other one in the respective matplotlib figure using
a blockplotlib function.

More examples can be found under `blockplotlib/examples/`.

## Installation

One way to use blockplotlib is to install it via
pip (`pip install blockplotlib`).
Another way to use it is to copy the file `blockplotlib.py` in the folder
of the Python script from which it is to be imported.
