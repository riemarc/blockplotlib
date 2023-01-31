from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

from blockplotlib import *

update_bpl_params(
    rp_block_width=10,
    rp_block_height=3,
    ap_block_tip_length=0.8,
    ap_block_line_width=0.12,
)

dx, dy = 20, 10
grid = dict(
    # distributed parameter system
    dps=(0, 0),
    # lumped parameter system
    lps=(0, -dy),
    # distributed control law
    dcl=(dx, 0),
    # lumped control law
    lcl=(dx, -dy),
    # half way crossing
    hwc=(dx / 3.4, 0),
    # arrow end point
    aep=(dx - bpl_params["rp_block_width"] / 6,
         -dy + bpl_params["rp_block_height"] / 2),
    # late lumping text anchor
    llta=(dx * 1.4, dy * 0.4),
    # early lumping text anchor
    elta=(-dx * 0.4, -dy * 1.4),
)

dps = RectangleBlock(grid["dps"], r"\begin{center} system with \\"
                                  r"distributed parameters\end{center}")
lps = RectangleBlock(grid["lps"], r"\begin{center} system with \\"
                                  r"lumped parameters\end{center}")
dcl = RectangleBlock(grid["dcl"], r"\begin{center} distributed \\"
                                  r" control law\end{center}")
lcl = RectangleBlock(grid["lcl"], r"\begin{center} lumped \\"
                                  r" control law\end{center}")
pad = 0.5
a1 = Arrow(dps, dcl, "e").place_text(r"design", "n", pad_xy=(0, pad))
a2 = Arrow(dps, lps, "s").place_text(r"approximation", "w", pad_xy=(-pad, 0))
a3 = Arrow(lps, lcl, "e").place_text(r"design", "s", pad_xy=(0, -pad))
a4 = Arrow(dcl, lcl, "s").place_text(r"approximation", "e", pad_xy=(pad, 0))

ed_bpl_params = bpl_params.copy()
ed_bpl_params["mpl_tpath_kws"].update(size=1.2)
llta = Node(grid["llta"])
llta.place_text(
    r"\begin{flushright}late lumping\end{flushright", "w",
    params=ed_bpl_params)
llta.place_text(
    r"\begin{flushright}\underline{direct} late lumping\end{flushright", "w",
    params=ed_bpl_params)
elta = Node(grid["elta"], r"\begin{flushleft}early lumping\end{flushleft", "e")

ed_bpl_params = bpl_params.copy()
ed_bpl_params["cop_block_radius"] = 0.25
hwc = Crossover(grid["hwc"], params=ed_bpl_params)
aep = Node(grid["aep"])
a5 = Arrow(hwc, aep, "m")
a4.place_text(r"\underline{approximation}", "w", pad_xy=(-dx / 2.6, 0))

hh = bpl_params["rp_block_height"] / 2 - bpl_params["rp_block_stroke_width"]
wh = bpl_params["rp_block_width"] / 2 - bpl_params["rp_block_stroke_width"]
tria1 = Polygon(
    [(dx - wh, -dy + hh), (dx + wh, -dy + hh), (dx + wh, -dy - hh)], lw=0.5)
tria2 = Polygon(
    [(dx - wh, -dy + hh), (dx + wh, -dy - hh), (dx - wh, -dy - hh)], lw=0)

set_color([a1, a4, a5, llta, hwc, tria1], "tab:green")
set_color([a2, a3, elta, tria2], "tab:blue")
set_alpha([tria1, tria2], 0.5)
tria1.set_zorder(0)
tria2.set_zorder(0)

place_patches(workspace=locals())
# show()

hide_patches([llta.txt_patches[-2]])
save_figure(stem="picture1")

hide_patches([llta.txt_patches[-2]], False)
hide_patches([a4.txt_patches[-1], a5, llta.txt_patches[-1], hwc])
save_figure(stem="picture2")

hide_patches(get_patches(locals()))
hide_patches([dps, lcl], False)
save_figure(stem="picture")

show()
