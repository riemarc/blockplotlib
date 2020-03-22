from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True

from blockplotlib import *

update_bpl_params(rp_block_height=3.2)


class TransportSystemBlock(RectangleBlock):
    def __init__(self, pos, var_text, params=None):
        if params is None:
            params = bpl_params.copy()

        load = self.get_load()
        mech = self.get_mechanics()
        mech.place_patch(load, "n")
        mech.place_text(var_text, "n", params=params, pad_xy=(0.6, -0.1))

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


sum_width = bpl_params["cp_block_radius"] * 2
t_sys_width = bpl_params["rp_block_width"]
int_height = bpl_params["rp_block_height"]
int_width = 2
g_width = 2.2
g_height = 2.2
dx = 3
dy = 3

grid = dict(
    t_sys=(0, 0),
    sum=(-(t_sys_width / 2 + dx + sum_width / 2), 0),
    input=(-(t_sys_width / 2 + 2 * dx + sum_width), 0),
    # sum bottom corner
    sbc=(-(t_sys_width / 2 + dx + sum_width / 2),
         -2 * dy - int_height / 2 - sum_width / 2),
    # integrals
    int1=(t_sys_width / 2 + dx + 0.5 * int_width, 0),
    int2=(t_sys_width / 2 + 2 * dx + 1.5 * int_width, 0),
    int3=(t_sys_width / 2 + 4 * dx + 3.5 * int_width, 0),
    # cross over
    co1=(t_sys_width / 2 + 1.5 * dx + int_width, 0),
    co2=(t_sys_width / 2 + 2.5 * dx + 2 * int_width, 0),
    co3=(t_sys_width / 2 + 3.5 * dx + 3 * int_width, 0),
    # line over top/bottom
    lot1=(t_sys_width / 2 + 3 * dx + 2 * int_width, 0),
    lot2=(t_sys_width / 2 + 3 * dx + 3 * int_width, 0),
    lob1=(t_sys_width / 2 + 3 * dx + 2 * int_width,
          -2 * dy - int_height / 2 - sum_width / 2),
    lob2=(t_sys_width / 2 + 3 * dx + 3 * int_width,
          -2 * dy - int_height / 2 - sum_width / 2),
    # output top corner
    otc=(t_sys_width / 2 + 4.5 * dx + 4 * int_width, 0),
    # output bottom corner
    obc=(t_sys_width / 2 + 4.5 * dx + 4 * int_width,
         -2 * dy - int_height / 2 - sum_width / 2),
    # gains
    g0=(0, -dy - int_height / 2),
    g1=(t_sys_width / 2 + 1.5 * dx + int_width, -dy - int_height / 2),
    g2=(t_sys_width / 2 + 2.5 * dx + 2 * int_width, -dy - int_height / 2),
    g3=(t_sys_width / 2 + 3.5 * dx + 3 * int_width, -dy - int_height / 2),
    g4=(t_sys_width / 2 + 4.5 * dx + 4 * int_width, -dy - int_height / 2),
    # sums
    s0=(0, -2 * dy - int_height / 2 - sum_width / 2),
    s1=(t_sys_width / 2 + 1.5 * dx + int_width,
        -2 * dy - int_height / 2 - sum_width / 2),
    s2=(t_sys_width / 2 + 2.5 * dx + 2 * int_width,
        -2 * dy - int_height / 2 - sum_width / 2),
    s3=(t_sys_width / 2 + 3.5 * dx + 3 * int_width,
        -2 * dy - int_height / 2 - sum_width / 2),
)

t_sys_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=0.8)
t_sys_params = cp(mpl_tpath_kws=t_sys_path_kws)
t_sys = TransportSystemBlock(grid["t_sys"], r"$\bar x_{n+1}(\theta, t)$",
                             params=t_sys_params)
sum = CircleBlock(grid["sum"])
inp = Node(grid["input"])
int_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=1.5)
int_params = cp(rp_block_width=int_width, mpl_tpath_kws=int_path_kws)
int1 = RectangleBlock(grid["int1"], r"$\int$", params=int_params)
int2 = RectangleBlock(grid["int2"], r"$\int$", params=int_params)
int3 = RectangleBlock(grid["int3"], r"$\int$", params=int_params)

co1 = Crossover(grid["co1"])
co2 = Crossover(grid["co2"])
co3 = Crossover(grid["co3"])
lot1 = Node(grid["lot1"])
lot2 = Node(grid["lot2"])
lob1 = Node(grid["lob1"])
lob2 = Node(grid["lob2"])
otc = Corner(grid["otc"])
obc = Corner(grid["obc"])
sbc = Corner(grid["sbc"])

g0_params = t_sys_params.copy()
g0_params.update(rp_block_width=11, rp_block_height=g_height,
                 mpl_tpath_kws=bpl_params["mpl_tpath_kws"])
g0 = RectangleBlock(grid["g0"], r"$a^*_{n+1} = \gamma(\theta) + "
                                r"\sum_{i=1}^m\gamma_i\delta_{\theta_i}$",
                    params=g0_params)
g_params = cp(rp_block_width=g_width, rp_block_height=g_height)
g1 = RectangleBlock(grid["g1"], r"$a_n$", params=g_params)
g2 = RectangleBlock(grid["g2"], r"$a_{n-1}$", params=g_params)
g3 = RectangleBlock(grid["g3"], r"$a_2$", params=g_params)
g4 = RectangleBlock(grid["g4"], r"$a_1$", params=g_params)
s0 = CircleBlock(grid["s0"])
s1 = CircleBlock(grid["s1"])
s2 = CircleBlock(grid["s2"])
s3 = CircleBlock(grid["s3"])

a1 = Arrow(inp, sum, "e")
a2 = Arrow(sum, t_sys, "e")
a3 = Arrow(t_sys, int1, "e")
a4 = Arrow(int1, int2, "e")
l1 = Line(int2, lot1, "e")
l2 = Line(lot1, lot2, "m", type="---")
a5 = Arrow(lot2, int3, "e")
a6 = CompoundPatch([
    Line(int3, otc, "e", "m"), otc,
    Arrow(otc, g4, "m", "n")
])
a7 = CompoundPatch([
    Line(g4, obc, "s", "m"),
    obc,
    Arrow(obc, s3, "m", "e")
])
a8a = Arrow(co3, g3, "m", "n")
a8 = Arrow(g3, s3, "s")
l4 = Line(s3, lob2, "w")
lob = Line(lob1, lob2, "m", type="---")
a9 = Arrow(lob1, s2, "w")
a10a = Arrow(co2, g2, "m", "n")
a10 = Arrow(g2, s2, "s")
a11 = Arrow(s2, s1, "w")
a12a = Arrow(co1, g1, "m", "n")
a12 = Arrow(g1, s1, "s")
a13 = Arrow(s1, s0, "w")
a14a = Arrow(t_sys, g0, "s")
a14 = Arrow(g0, s0, "s")
a15 = CompoundPatch([
    Line(s0, sbc, "w", "m"),
    sbc,
    Arrow(sbc, sum, "m", "s")
])

place_patches(workspace=locals())
show()
