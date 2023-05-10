from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True

from blockplotlib import *
from blockplotlib import Line as BplLine

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

        return PathPatch(Path(path_data, path_codes), facecolor="gray",
                         capstyle='round')

    def get_mechanics(self):
        circ1 = CircleBlock((0, 0), params=cp(cp_block_radius=0.5))
        circ2 = CircleBlock((6, 0), params=cp(cp_block_radius=0.5))
        offset = np.array([0, bpl_params["cp_block_stroke_width"] / 2])
        node1n = circ1.get_anchor("n") - offset
        node1s = circ1.get_anchor("s") + offset
        node2n = circ2.get_anchor("n") - offset
        node2s = circ2.get_anchor("s") + offset
        ed_params = cp(ap_block_line_width=bpl_params["cp_block_stroke_width"])
        l1 = BplLine(node1n, node2n, "m", params=ed_params)
        l2 = BplLine(node1s, node2s, "m", params=ed_params)

        return circ1 + circ2 + l1 + l2


class Line(BplLine):
    def __init__(self, it1, it2, loc1, loc2=None, type="-", params=None):
        super().__init__(it1, it2, loc1, loc2=loc2, type=type, params=params)

        if not (type == "--" or type == "---"):
            if not isinstance(it1, PatchGroup):
                it1 = Node(it1)

            a1 = it1.get_anchor(loc1)
            corner = Corner(a1)
            new_patch = CompoundPatch([self, corner])
            self.geo_patches[0].set_xy(new_patch.geo_patches[0].get_xy())


class Arrow(Line):
    def __init__(self, it1, it2, loc1, loc2=None, type="->", params=None):
        super().__init__(it1, it2, loc1, loc2, type, params)


sum_width = bpl_params["cp_block_radius"] * 2
t_sys_width = bpl_params["rp_block_width"]
int_height = bpl_params["rp_block_height"]
int_width = 2
g_width = 2.3
g_height = 2.2
dx = 1.9
ddx = 1.6 * dx / 2
dy = 8
ddy = (dy - sum_width / 2 - g_width) / 2

grid = dict(
    t_sys=(0, 0),
    output=(+(t_sys_width / 2 + 1.5 * dx), 0),
    # system bottom node
    sbn=(0, -dy),
    # ints
    int3=(-(t_sys_width + int_width) / 2 - 1.5 * dx, 0),
)
grid.update(dict(
    int2=(grid["int3"][0] - 2 * ddx - int_width - sum_width, 0)
))
grid.update(dict(
    int1=(grid["int2"][0] - 2 * dx - 2 * ddx - int_width - 2 * sum_width, 0)
))
# corners
grid.update(dict(
    # corner top left
    co_tl=(grid["int1"][0] - int_width / 2 - ddx, 0),
    # corner top right
    co_tr=(grid["output"][0] + dx, 0),
    # corner bottom right
    co_br=(grid["output"][0], -dy),
))
grid.update(dict(
    # corner bottom left
    co_bl=(grid["co_tl"][0], -dy),
))
# sums
grid.update(dict(
    s1=(grid["int1"][0] + int_width / 2 + ddx + sum_width / 2, 0),
    s2=(grid["int2"][0] - int_width / 2 - ddx - sum_width / 2, 0),
    s3=(grid["int3"][0] - int_width / 2 - ddx - sum_width / 2, 0),
))
# gains
grid.update(dict(
    g1=(grid["co_tl"][0], -sum_width / 2 - ddy - g_width / 2)
))
grid.update(dict(
    g2=(grid["s1"][0], grid["g1"][1]),
    g3=(grid["s2"][0], grid["g1"][1]),
    g4=(grid["s3"][0], grid["g1"][1]),
    g0=(0, grid["g1"][1])
))
# cross over
grid.update(dict(
    co1=(grid["s1"][0], -dy),
    co2=(grid["s2"][0], -dy),
    co3=(grid["s3"][0], -dy),
    co4=(0, -dy),
))
# dotted helper
eps = 0.3
grid.update(dict(
    # dotted helper top
    dht1=(grid["s1"][0] + sum_width / 2 + dx * eps, 0),
    dht2=(grid["s2"][0] - sum_width / 2 - dx * eps * 1.5, 0),
))
grid.update(dict(
    # dotted helper bottom
    dhb1=(grid["dht1"][0], -dy),
    dhb2=(grid["dht2"][0], -dy)
))

t_sys_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=0.8)
t_sys_params = cp(mpl_tpath_kws=t_sys_path_kws)
t_sys = TransportSystemBlock(grid["t_sys"], r"$\xi_{N+1}(\theta, t)$",
                             params=t_sys_params)
int_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=1.5)
int_params = cp(rp_block_width=int_width, mpl_tpath_kws=int_path_kws)
int1 = RectangleBlock(grid["int1"], r"$\int$", params=int_params)
int2 = RectangleBlock(grid["int2"], r"$\int$", params=int_params)
int3 = RectangleBlock(grid["int3"], r"$\int$", params=int_params)

co1 = Crossover(grid["co1"])
co2 = Crossover(grid["co2"])
co3 = Crossover(grid["co3"])
co4 = Crossover(grid["co4"])
co5 = Crossover(grid["output"])
co_tl = Corner(grid["co_tl"])
co_tr = Corner(grid["co_tr"])
co_bl = Corner(grid["co_bl"])
co_br = Corner(grid["co_br"])
dht1 = Node(grid["dht1"])
dht2 = Node(grid["dht2"])
dhb1 = Node(grid["dhb1"])
dhb2 = Node(grid["dhb2"])

g0_params = t_sys_params.copy()
g0_params.update(rp_block_width=12, rp_block_height=g_height,
                 mpl_tpath_kws=bpl_params["mpl_tpath_kws"])
g0 = RectangleBlock(grid["g0"], r"$a_{N+1} = \tilde{a}_{N+1}(\theta) + "
                                r"\sum_{i=1}^m\mathring{a}_i\delta_{\theta_i}$",
                    params=g0_params)
g_params = cp(rp_block_width=g_width, rp_block_height=g_height)
g1 = RectangleBlock(grid["g1"], r"$\tilde{a}_1$", params=g_params)
g2 = RectangleBlock(grid["g2"], r"$\tilde{a}_2$", params=g_params)
g3 = RectangleBlock(grid["g3"], r"$\tilde{a}_{N-1}$", params=g_params)
g4 = RectangleBlock(grid["g4"], r"$\tilde{a}_N$", params=g_params)
s1 = CircleBlock(grid["s1"])
s2 = CircleBlock(grid["s2"])
s3 = CircleBlock(grid["s3"])

a1 = Arrow(co_tl, int1, "e")
a1.place_text(r"$-$", "s", pad_xy=(0.2, -0.3))
a2 = Arrow(int1, s1, "e")
a2.place_text(r"$\xi_1(t)$", loc1="w", loc2="w", pad_xy=(0.5, 1.2))
a3 = Arrow(s2, int2, "e")
a4 = Arrow(int2, s3, "e")
a4.place_text(r"$\xi_{N-1}(t)$", loc1="w", loc2="w", pad_xy=(0.5, 1.2))
a5 = Arrow(s3, int3, "e")
a6 = Arrow(int3, t_sys, "e")
a6.place_text(r"$\xi_{N}(t)$", loc1="w", loc2="w", pad_xy=(0.5, 1.2))
a7 = Arrow(t_sys, co_tr, "e")
a7.place_text(r"$y(t)$", loc1="m", loc2="m", pad_xy=(0, 1.2))
ldt1 = Line(s1, dht1, "e")
ldt2 = Line(dht1, dht2, "m", type="---")
ldt3 = Arrow(dht2, s2, "e")
l1 = Line(co5, co_br, "m", "n")
l2 = Line(co_br, co4, "m")
l3 = Line(co4, co3, "m")
l4 = Line(co2, co3, "m")
ldb1 = Line(co1, dhb1, "e")
ldb2 = Line(dhb1, dhb2, "m", type="---")
ldb3 = Line(dhb2, co2, "m")
l6 = Line(co1, co_bl, "m", "e")
a8 = Arrow(co_bl, g1, "m", "s")
l7 = Line(g1, co_tl, "n", "m")
a9 = Arrow(co1, g2, "m", "s")
a10 = Arrow(co2, g3, "m", "s")
a12 = Arrow(co3, g4, "m", "s")
a13 = Arrow(co4, g0, "m", "s")
a9b = Arrow(g2, s1, "n", "s")
a9b.place_text(r"$-$", "e",
               pad_xy=(0.3, a9b.get_geo_extents().height / 2 - 0.3))
a10b = Arrow(g3, s2, "n", "s")
a10b.place_text(r"$-$", "e",
                pad_xy=(0.3, a10b.get_geo_extents().height / 2 - 0.3))
a12b = Arrow(g4, s3, "n", "s")
a12b.place_text(r"$-$", "e",
                pad_xy=(0.3, a12b.get_geo_extents().height / 2 - 0.3))
a13b = Arrow(g0, t_sys, "n", "s")
a13b.place_text(r"$-$", "e",
                pad_xy=(0.3, a13b.get_geo_extents().height / 2 - 0.3))
cp1 = CompoundPatch([l6, co_bl, a8])
cp2 = CompoundPatch([l7, co_tl, a1])
cp3 = CompoundPatch([l1, co_br, l2])

place_patches(workspace=locals())
save_figure()
show()
