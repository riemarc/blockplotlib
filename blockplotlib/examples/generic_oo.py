import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
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

a_s_s = Arrow(setpoint, sum, "e")
a_s_sy = Arrow(sum, system, "e")
l_sy_c = Line(system, so_cross, "e", "m")
a_c_o = Arrow(so_cross, output, "e")
l_c_x = Line(so_cross, xk_corner, "m")
a_x_g = Arrow(xk_corner, gain, "m", "e")
l_g_k = Line(gain, ks_corner, "w", "m")
a_k_s = Arrow(ks_corner, sum, "m", "s")

place_patches(workspace=locals())

plt.axis("equal")
plt.show()
