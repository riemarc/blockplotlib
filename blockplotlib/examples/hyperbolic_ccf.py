from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True

from blockplotlib import *

text_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=0.7)
update_bpl_params(mpl_tpath_kws=text_path_kws, rp_block_height=3.2)


class TransportSystemBlock(RectangleBlock):
    def __init__(self, pos, var_text, params=None):
        if params is None:
            params = bpl_params.copy()

        load = self.get_load()
        mech = self.get_mechanics()
        mech.place_patch(load, "n")
        mech.place_text(var_text, "n")

        super().__init__(pos, params=params)
        self.place_patch(mech, "m", kind2="all")

    def get_load(self):
        path_data = np.array([
            [0., 0.], [0., 0.], [1.24280106, 0.94325026],
            [1.61251232, 0.95352058], [1.98222083, 0.96379082],
            [2.20055516, 0.13050694], [2.66461654, 0.07402118],
            [3.12867565, 0.0175349], [4.42427526, 0.6155791],
            [4.88705001, 0.6053105], [5.34982754, 0.59504106], [6., 0.],
            [6., 0.]])
        path_codes = [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        return PathPatch(Path(path_data, path_codes), facecolor="gray")

    def get_mechanics(self):
        circ1 = CircleBlock((0, 0), params=cp(cp_block_radius=0.5))
        circ2 = CircleBlock((6, 0), params=cp(cp_block_radius=0.5))
        offset = np.array([0, bpl_params["cp_block_stroke_width"] / 2])
        node1n = circ1.get_anchor("n") - offset
        node1s = circ1.get_anchor("s") + offset
        node2n = circ2.get_anchor("n") - offset
        node2s = circ2.get_anchor("s") + offset
        ed_params = cp(ap_block_line_width=bpl_params["cp_block_stroke_width"])
        l1 = Line(node1n, node2n, "m", params=ed_params)
        l2 = Line(node1s, node2s, "m", params=ed_params)

        return circ1 + circ2 + l1 + l2


patch = TransportSystemBlock((0, 0), r"$\bar x_{n+1}(\theta, t)$")

place_patches(workspace=locals())
show()
