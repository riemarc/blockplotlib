import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Wedge, Circle, Rectangle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
import numpy as np


usetex = True
matplotlib.rcParams['text.usetex'] = usetex
rp_width = 8
rp_height = 2
cp_radius = rp_height / 8
block_strocke_width = cp_radius / 4
cp_width = cp_radius / 2
label_size = rp_height / 2
arrow_length = 0.45
arrow_lw = block_strocke_width / 2
rect_path_kwargs = dict()
rect_patch_kwargs = dict(facecolor="black", edgecolor=None, linewidth=None)
text_path_kwargs = dict(usetex=usetex, size=label_size)
text_patch_kwargs = dict(facecolor="black", edgecolor=None, linewidth=None)
circle_patch_kwargs = dict(facecolor="black", edgecolor=None, linewidth=None)


def get_anchor(it, loc):
    c = it.get_extents()
    if loc == "w":
        anchor = (c.x0, (c.y1 + c.y0) / 2)
    elif loc == "e":
        anchor = (c.x1, (c.y1 + c.y0) / 2)
    elif loc == "n":
        anchor = ((c.x1 + c.x0) / 2, c.y1)
    elif loc == "s":
        anchor = ((c.x1 + c.x0) / 2, c.y0)
    elif loc == "m":
        anchor = ((c.x1 + c.x0) / 2, (c.y1 + c.y0) / 2)
    else:
        raise ValueError()

    return np.array(anchor)


opposite_loc = {"w": "e", "e": "w", "n": "s", "s": "n", "m":"m"}


def align_patch1_to_patch2(p1, loc1, p2, loc2=None, pad_xy=(0, 0), axis=None):
    if axis is None: axis = plt.gca()
    if loc2 is None: loc2 = opposite_loc[loc1]

    anchor1 = get_anchor(p1, loc1)
    anchor2 = get_anchor(p2, loc2)
    shift = anchor2 - anchor1 + np.array(pad_xy).flatten()
    p1.set_transform(Affine2D().translate(shift[0], shift[1]) +
                     axis.transData)


def get_arrow_patch(anchor1, anchor2, type, d_angle=0.3, length=arrow_length,
                    width=arrow_lw):
    vect = anchor1 - anchor2
    angle = np.arctan2(vect[1], vect[0])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    tang = np.array([np.cos(angle), np.sin(angle)])

    if type == "->":
        arrow_p1 = anchor2 + np.array([np.cos(angle + d_angle),
                                       np.sin(angle + d_angle)]) * length
        arrow_p2 = anchor2 + np.array([np.cos(angle - d_angle),
                                       np.sin(angle - d_angle)]) * length
        arrow_edge = normal * (2 * arrow_length * np.sin(d_angle) - arrow_lw) / 2
        arrow_p11 = arrow_p1 - arrow_edge
        arrow_p22 = arrow_p2 + arrow_edge
        arrow_line = tang * (np.linalg.norm(vect) - arrow_length * np.cos(d_angle))
        arrow_p111 = arrow_p11 + arrow_line
        arrow_p222 = arrow_p22 + arrow_line

        verts = [anchor2, arrow_p1, arrow_p11, arrow_p111, arrow_p222, arrow_p22, arrow_p2, anchor2, anchor2]
        codes = [Path.MOVETO] + [Path.LINETO] * 7 + [Path.CLOSEPOLY]

    elif type == "-":
        p1 = anchor1 + normal * arrow_lw / 2
        p2 = anchor2 + normal * arrow_lw / 2
        p3 = anchor2 - normal * arrow_lw / 2
        p4 = anchor1 - normal * arrow_lw / 2
        verts = [p1, p2, p3, p4, p1, p1]
        codes = [Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY]

    else:
        raise NotImplementedError()

    arrow_path = Path(verts, codes)
    arrow_patch = PathPatch(arrow_path, facecolor="black")

    return arrow_patch


def get_text(label, tp_kws=None, tpc_kws=None):
    if tp_kws is None: tp_kws = text_path_kwargs
    if tpc_kws is None: tpc_kws = text_patch_kwargs

    text_path = TextPath((0, 0), label, **tp_kws)
    text_patch = PathPatch(text_path, **tpc_kws)

    return text_patch


def get_rectangle(pos, label=None, width=rp_width, height=rp_height,
                  stroke_width=block_strocke_width, rp_kws=None,
                  rpc_kws=None, tp_kws=None, tpc_kws=None):
    if rp_kws is None: rp_kws = rect_path_kwargs
    if rpc_kws is None: rpc_kws = rect_patch_kwargs

    pos = np.array(pos)
    dx = np.array([width / 2, 0])
    dxi = np.array([width / 2 - stroke_width, 0])
    dy = np.array([0, height / 2])
    dyi = np.array([0, height / 2 - stroke_width])
    verts = [
        pos - dx - dy, pos - dx + dy, pos + dx + dy, pos + dx - dy,
        pos - dx - dy, pos - dxi - dyi, pos + dxi - dyi, pos + dxi + dyi,
        pos - dxi + dyi, pos - dxi - dyi, pos - dx - dy, pos - dx - dy]
    codes = ([Path.MOVETO] + [Path.LINETO] * 4 + [Path.MOVETO] +
             [Path.LINETO] * 4 + [Path.MOVETO] + [Path.CLOSEPOLY])
    rect_path = Path(verts, codes, **rp_kws)
    rect_patch = PathPatch(rect_path, **rpc_kws)

    if not label:
        return rect_patch

    text_patch = get_text(label, tp_kws, tpc_kws)
    align_patch1_to_patch2(text_patch, "m", rect_patch)

    return rect_patch, text_patch


def get_circle(pos, label=None, radius=cp_radius, width=cp_width, cp_kws=None,
               tp_kws=None, tpc_kws=None):
    if cp_kws is None: cp_kws = circle_patch_kwargs
    if width is None: width = radius

    if radius == 0:
        circle_patch = Circle(pos, radius, **cp_kws)
    else:
        circle_patch = Wedge(pos, radius, 0, 360, width, **cp_kws)

    if not label:
        return circle_patch,

    text_patch = get_text(label, tp_kws, tpc_kws)
    align_patch1_to_patch2(text_patch, "m", circle_patch)

    return circle_patch, text_patch


def get_arrow(it1, loc1, it2, loc2, type="->"):
    a1 = get_anchor(it1[0], loc1)
    a2 = get_anchor(it2[0], loc2)

    return get_arrow_patch(a1, a2, type)


def place_patches(axis=None, patches=None):
    if axis is None:
        axis = plt.gca()

    def add_patch(patch):
        if patch not in axis.patches:
            axis.add_patch(patch)

    allowed_patches = (Rectangle, PathPatch, Circle, Wedge)

    if patches is None:
        for v in globals().values():
            if isinstance(v, allowed_patches):
                add_patch(v)
            elif isinstance(v, tuple):
                for vv in v:
                    if not isinstance(vv, allowed_patches):
                        break
                else:
                    for p in v:
                        add_patch(p)

    else:
        for patch in patches:
            add_patch(patch)



if __name__ == "__main__":

    plt.figure(figsize=(5,5), facecolor="white")
    plt.axis("off")

    b1 = get_rectangle((0, 0), r"\bf{hello axis} $a\mapsto b$")
    b2 = get_rectangle((0, 4), r"\bf{hello axis} $a - b$")
    b3 = get_rectangle((0, -4), r"\bf{hello axis} $a\rightarrow b$")
    c1 = get_arrow(b1, "n", b2, "s")
    c2 = get_arrow(b3, "n", b1, "s")

    b4 = get_rectangle((12, 0), r"\bf{hello gaxis} $a\mapsto b$")
    b5 = get_rectangle((12, 4), r"\begin{center}\bf{hello axis} \\ $a - b$\end{center}", height=rp_height*1.5)
    b6 = get_rectangle((12, -4), r"\bf{hello axis} $a\rightarrow b$")
    c3 = get_arrow(b1, "e", b4, "w")
    c4 = get_arrow(b1, "e", b5, "w")
    c5 = get_arrow(b1, "e", b5, "s")
    c6 = get_arrow(b1, "e", b6, "w")
    c7 = get_arrow(b1, "e", b6, "n")
    c8 = get_arrow(b2, "e", b5, "w")

    u1 = get_circle((24, 0), width=None)
    u2 = get_circle((24, -4), r"$\Phi(t)=e^{A t}$", radius=rp_height * 1.3)
    u3 = get_circle((28, 0), radius=0)
    u4 = get_circle((24, 4), radius=0)
    c9 = get_arrow(b4, "e", u1, "w", type="-")
    c10 = get_arrow(u1, "e", u3, "w")
    c11 = get_arrow(u1, "s", u2, "n")
    c12 = get_arrow(b5, "e", u4, "w", type="-")
    c13 = get_arrow(u4, "s", u1, "n", type="-")

    t1 = get_text(r"$y(t)$")
    align_patch1_to_patch2(t1, "s", c9)
    place_patches()

    plt.axis("equal")
    plt.show()
