import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from blockplotlib import *


plt.figure(figsize=(5, 5), facecolor="white")
plt.axis("off")

dx = 4
dy = 4
grid = dict(
    setpoint=(0, 0),
    sum=(dx, 0),
    system=(dx * 3, 0),
    output=(dx * 5.7, 0),
    so_cross=(dx * 4.9, 0),
    xk_corner=(dx * 4.9, -dy),
    gain=(dx * 3, -dy),
    ks_corner=(dx, -dy)
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

a1.place_text(r"$w$", "s", pad_xy=(0, .2))
a2.place_text(r"$u$", "s", pad_xy=(0, .2))
a3.place_text(r"$x$", "s", pad_xy=(0, .2))

place_patches(workspace=locals())

plt.axis("equal")
plt.show()
