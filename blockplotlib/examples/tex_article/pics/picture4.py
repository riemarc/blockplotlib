import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.serif'] = ["Computer Modern"]

from blockplotlib import *

dx, dy = 4, 4
x1, x2, x3, x4, x5 = 0, dx, 3 * dx, 4.5 * dx, 5.3 * dx
y1, y2 = 0, -dy
grid = dict(
    setpoint=(x1, y1),
    sum=(x2, y1),
    system=(x3, y1),
    so_cross=(x4, y1),
    output=(x5, y1),
    xk_corner=(x4, y2),
    gain=(x3, y2),
    ks_corner=(dx, y2)
)

system = RectangleBlock(grid[r"system"], r"$\dot x = A x + B u$")
gain = RectangleBlock(grid[r"gain"], r"$K$", params=cp(rp_block_width=3))
sum = CircleBlock(grid["sum"])
setpoint = Node(grid["setpoint"])
output = Node(grid["output"])
so_cross = Crossover(grid["so_cross"])
xk_corner = Corner(grid["xk_corner"])
ks_corner = Corner(grid["ks_corner"])

a1 = Arrow(setpoint, sum, "e")
a2 = CompoundPatch([
    Arrow(sum, system, "e"),
    Corner(sum.get_anchor("e"))])
l1 = Line(system, so_cross, "e", "m")
a3 = Arrow(so_cross, output, "m")
a4 = CompoundPatch([
    Line(so_cross, xk_corner, "m"),
    xk_corner,
    Arrow(xk_corner, gain, "m", "e")])
a5 = Arrow(ks_corner, sum, "m", "s")
a5.place_text(r"$-$", "w", pad_xy=(-0.3, a5.get_geo_extents().height / 2 - 0.2))
a55 = CompoundPatch([
    Line(gain, ks_corner, "w", "m"),
    ks_corner,
    a5])

a1.place_text(r"$w$", "n", pad_xy=(0, .2))
a2.place_text(r"$u$", "n", pad_xy=(0, .2))
a3.place_text(r"$x$", "n", pad_xy=(0, .2))

place_patches(workspace=locals())

save_figure()
write_bpl_tex_file()
show()
