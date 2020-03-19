import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

from blockplotlib import *

dx, dy = 4, 4
x0, x1, x2, x3, x4, x5 = -dx, 0, dx, 2.9 * dx, 4.5 * dx, 5.3 * dx
y0, y1, y2 = 0, 0, -dy
grid = dict(
    setpoint=(x0, y0),
    filter=(x1, y1),
    sum=(x2, y1),
    system=(x3, y1),
    so_cross=(x4, y1),
    output=(x5, y1),
    xk_corner=(x4, y2),
    gain=(x3, y2),
    ks_corner=(dx, y2)
)

filter = RectangleBlock(grid[r"filter"], r"$V$", params=cp(rp_block_width=3))
system = RectangleBlock(grid[r"system"], r"$\dot x = A x + B u$")
gain = RectangleBlock(grid[r"gain"], r"$K$", params=cp(rp_block_width=3))
sum = CircleBlock(grid["sum"])
setpoint = Node(grid["setpoint"])
output = Node(grid["output"])
so_cross = Crossover(grid["so_cross"])
xk_corner = Corner(grid["xk_corner"])
ks_corner = Corner(grid["ks_corner"])

a0 = Arrow(setpoint, filter, "e")
a1 = Arrow(filter, sum, "e")
a2 = Arrow(sum, system, "e")
l1 = Line(system, so_cross, "e", "m")
a3 = Arrow(so_cross, output, "m")
a4 = CompoundPatch([
    Line(so_cross, xk_corner, "m"),
    xk_corner,
    Arrow(xk_corner, gain, "m", "e")])
a5 = Arrow(ks_corner, sum, "m", "s")
a5.place_text(r"$-$", "e", pad_xy=(-0.3, a5.get_geo_extents().height / 2 - 0.2))
a55 = CompoundPatch([
    Line(gain, ks_corner, "w", "m"),
    ks_corner,
    a5])

a0.place_text(r"$w$", "s", pad_xy=(0, .2))
a2.place_text(r"$u$", "s", pad_xy=(0, .2))
a3.place_text(r"$x$", "s", pad_xy=(0, .2))

place_patches(workspace=locals())
plt.axis("off")
plt.axis("equal")
plt.show()
