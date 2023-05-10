"""
see readme file at https://github.com/riemarc/blockplotlib
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import (
    Wedge, Circle, PathPatch, Patch, Polygon as PolygonPatch, Rectangle)
from matplotlib.text import TextPath, Text
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union


__all__ = ["bpl_params", "opposite_loc", "get_anchor", "PatchGroup",
           "RectangleBlock", "change_params",
           "place_patches", "CircleBlock", "cp", "Corner", "Node", "Line",
           "Arrow", "Crossover", "CompoundPatch", "save_figure",
           "write_bpl_tex_file", "show", "update_bpl_params", "set_alpha",
           "set_color", "get_patches", "get_mpl_patches", "hide_patches",
           "get_name_from_sys_argv"]


patch_kws = dict(
    facecolor="black",
    edgecolor="black",
    linewidth=0,
    snap=False,
    aa=True)
text_path_kws = dict(usetex=mpl.rcParams['text.usetex'], size=1)
bpl_params = dict(
    # rp ~ rectangle patch
    rp_block_width=8,
    rp_block_height=2,
    rp_block_stroke_width=0.125,
    # cp ~ circle patch
    cp_block_radius=0.5,
    cp_block_stroke_width=0.125,
    # cop ~ cross over patch
    cop_block_radius=0.16,
    # ap ~ arrow patch / line patch
    ap_block_tip_angle=0.4,
    ap_block_tip_length=0.45,
    ap_block_line_width=0.08,
    ap_block_seg_len=0.3,
    ap_block_seg_num=3,
    # rpath_kws ~ rectangle path kwargs
    mpl_rpath_kws=dict(),
    # rpatch_kws ~ rectangle patch kwargs
    mpl_rpatch_kws=patch_kws.copy(),
    # cpatch_kws ~ circle patch kwargs
    mpl_cpatch_kws=patch_kws.copy(),
    # apatch_kws ~ arrow patch kwargs
    mpl_apatch_kws=patch_kws.copy(),
    # tpatch_kws ~ text patch kwargs
    mpl_tpatch_kws=patch_kws.copy(),
    # tpath_kws ~ text path kwargs
    mpl_tpath_kws=text_path_kws.copy(),
)
_bpl_params = bpl_params.copy()
opposite_loc = {"w": "e", "e": "w", "n": "s", "s": "n", "m": "m"}


def reset_bpl_params():
    bpl_params.update(_bpl_params.copy())


def update_bpl_params(**kwargs):
    bpl_params.update(kwargs)


def change_params(**kwargs):
    params = bpl_params.copy()
    params.update(kwargs)

    return params


def cp(**kwargs):
    return change_params(**kwargs)


def show(*args, **kwargs):
    if "clipped" in kwargs:
        clipped = kwargs.pop("clipped")
    else:
        clipped = True
    if "latexmk" not in sys.argv:
        if not clipped:
            return plt.show(*args, **kwargs)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.margins(0)
        return plt.show(*args, **kwargs)


def get_patches(workspace):
    patches = list()

    for v in workspace.values():
        if isinstance(v, allowed_patches):
            patches.append(v)

    return patches


def get_mpl_patches(patches=None, workspace=None):
    if patches is None:
        if workspace is None:
            raise ValueError
        else:
            patches = get_patches(workspace)

    pts = list()
    for pt in patches:
        if isinstance(pt, PatchGroup):
            pts += pt.get_patches()
        else:
            pts.append(pt)

    return pts


def hide_patches(patches, b=True):
    for patch in get_mpl_patches(patches):
        patch.set_visible(not b)


def set_alpha(patches, alpha):
    for pt in get_mpl_patches(patches):
        pt.set_alpha(alpha)


def set_color(patches, color):
    for pt in get_mpl_patches(patches):
        pt.set_color(color)


def place_patches(patches=None, workspace=None, axis=None):
    if axis is None:
        axis = plt.gca()

    for patch in get_mpl_patches(patches, workspace):
        if patch not in axis.patches:
            axis.add_patch(patch)


def remove_patches(patches, axis=None):
    if axis is None:
        axis = plt.gca()

    for patch in patches:
        if patch in axis.patches:
            axis.add


def get_anchor(bbox, loc):
    if loc == "w":
        anchor = (bbox.x0, (bbox.y1 + bbox.y0) / 2)
    elif loc == "e":
        anchor = (bbox.x1, (bbox.y1 + bbox.y0) / 2)
    elif loc == "n":
        anchor = ((bbox.x1 + bbox.x0) / 2, bbox.y1)
    elif loc == "s":
        anchor = ((bbox.x1 + bbox.x0) / 2, bbox.y0)
    elif loc == "m":
        anchor = ((bbox.x1 + bbox.x0) / 2, (bbox.y1 + bbox.y0) / 2)
    else:
        raise ValueError()

    return np.array(anchor)


def get_name_from_sys_argv(sys_argv=None, stem=None):
    if sys_argv is None:
        sys_argv = sys.argv

    path = pathlib.Path(sys_argv[0])

    if stem is None:
        stem = path.stem

    return str(path.parent) + pathlib.os.sep + stem


def save_figure(name=None, fig=None, stem=None, clipped=True):
    if name is None:
        name = get_name_from_sys_argv(stem=stem)

    if "." not in name:
        name = name + ".pdf"

    if not clipped:
        return plt.savefig(name)

    if fig is None:
        fig = plt.gcf()

    ax = fig.axes[0]
    ax.set_aspect("equal")
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    fig.canvas.draw()
    ax.axis("off")

    return plt.savefig(name, bbox_inches=ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()))


def write_bpl_tex_file(name=None, pic_name=None, fig=None, clipped=True,
                       fontsize=None, boxed=False, frickel_factor=1):
    if name is None:
        name = get_name_from_sys_argv() + ".bpl_tex"

    if pic_name is None:
        pic_name = get_name_from_sys_argv()

    if "." not in name:
        name = name + ".pdf"

    if fig is None:
        fig = plt.gcf()

    # this string should be used in the latex document, too
    meter = "fdjJklgq12h"
    if clipped:
        ax = fig.axes[0]
        font_height = TextPath((0, 0), meter,
                               **text_path_kws).get_extents().height
        pic_height = ax.viewLim.y1 - ax.viewLim.y0
    else:
        if fontsize is None:
            txt = plt.text(0, 0, meter)
        else:
            txt = plt.text(0, 0, meter, fontsize=fontsize)
        renderer = fig.canvas.get_renderer()
        bbox = txt.get_window_extent(renderer)
        font_height = bbox.transformed(fig.dpi_scale_trans.inverted()).height
        pic_height = fig.get_size_inches()[1]
        txt.set_visible(False)

    with open(name, "w") as file:
        ratio = pic_height / font_height
        line = "\includegraphics[height={}\BplLengthUnit]{{{}}}".format(
            ratio * frickel_factor, pic_name)
        if boxed:
            line = r"{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{1pt}\fbox{" + line + r"}}"
        file.write(line)


def get_extents(patches):
    extents = [p.get_extents() for p in patches]
    x0 = min([e.x0 for e in extents])
    x1 = max([e.x1 for e in extents])
    y0 = min([e.y0 for e in extents])
    y1 = max([e.y1 for e in extents])

    return mpl.transforms.Bbox([[x0, y0], [x1, y1]])


def translate_patch(patch, shift):
    if isinstance(patch, (Circle, Wedge)):
        patch.set_center(patch.center + shift)
    elif isinstance(patch, Rectangle):
        patch.set_xy(patch.get_xy() + shift)
    elif isinstance(patch, PathPatch):
        patch.set_path(patch.get_path().transformed(Affine2D().translate(
            shift[0], shift[1])))
    elif isinstance(patch, PolygonPatch):
        patch.set_xy(patch.get_xy() + shift)


class PatchGroup:
    def __init__(self, geo_patches=None, txt_patches=None):
        if geo_patches:
            self.geo_patches = [p for p in geo_patches]
        else:
            self.geo_patches = list()

        if txt_patches:
            self.txt_patches = [p for p in txt_patches]
        else:
            self.txt_patches = list()

    def __add__(self, other):
        patch = PatchGroup(self.geo_patches + other.geo_patches)
        patch.txt_patches = self.txt_patches + other.txt_patches

        return patch

    def get_patches(self):
        return self.geo_patches + self.txt_patches

    def get_geo_extents(self):
        return get_extents(self.geo_patches)

    def get_txt_extents(self):
        return get_extents(self.txt_patches)

    def get_extents(self):
        return get_extents(self.get_patches())

    def place_text(self, text, loc1="m", pad_xy=(0, 0), params=None, loc=None,
                   loc2=None):
        if loc is not None:
            loc1 = loc
            raise DeprecationWarning(
                "loc kwarg of PatchGroup.place_text() is deprecated.")

        if params is None:
            params = bpl_params

        text_path = TextPath((0, 0), text, **params["mpl_tpath_kws"])
        text_patch = PathPatch(text_path, **params["mpl_tpatch_kws"])
        patch_group = PatchGroup(txt_patches=[text_patch])
        self.place_patch(patch_group, loc1=loc1, loc2=loc2, kind2="txt",
                         pad_xy=pad_xy)

        return self

    def place_patch(self, patch, loc1, loc2=None, kind1="geo", kind2="geo",
                    pad_xy=(0, 0)):
        if loc2 is None:
            loc2 = opposite_loc[loc1]

        if not isinstance(patch, PatchGroup):
            patch = PatchGroup([patch])

        anchor1 = self.get_anchor(loc1, kind1)
        anchor2 = patch.get_anchor(loc2, kind2)
        shift = anchor1 - anchor2
        patch.translate(shift + np.array(pad_xy))
        self.geo_patches += patch.geo_patches
        self.txt_patches += patch.txt_patches

    def get_anchor(self, loc="m", kind="geo"):
        if kind == "geo":
            bbox = self.get_geo_extents()
        elif kind == "txt":
            bbox = self.get_txt_extents()
        elif kind == "all":
            bbox = self.get_extents()
        else:
            raise ValueError

        return get_anchor(bbox, loc)

    def translate(self, shift):
        for patch in self.get_patches():
            translate_patch(patch, shift)

    def set_visible(self, b):
        for patch in self.get_patches():
            patch.set_visible(b)


class RectangleBlock(PatchGroup):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        width = params["rp_block_width"]
        height = params["rp_block_height"]
        stroke_width = params["rp_block_stroke_width"]

        pos = np.array(pos)
        dx = np.array([width / 2, 0])
        dxi = np.array([width / 2 - stroke_width, 0])
        dy = np.array([0, height / 2])
        dyi = np.array([0, height / 2 - stroke_width])
        if width <= 2 * stroke_width or height <= 2 * stroke_width:
            verts = [
                pos - dx - dy, pos - dx + dy, pos + dx + dy, pos + dx - dy,
                pos - dx - dy, pos - dx - dy]
            codes = ([Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY])

        else:
            verts = [
                pos - dx - dy, pos - dx + dy, pos + dx + dy, pos + dx - dy,
                pos - dx - dy, pos - dxi - dyi, pos + dxi - dyi,
                pos + dxi + dyi,
                pos - dxi + dyi, pos - dxi - dyi, pos - dx - dy, pos - dx - dy]
            codes = ([Path.MOVETO] + [Path.LINETO] * 4 + [Path.MOVETO] +
                     [Path.LINETO] * 4 + [Path.MOVETO] + [Path.CLOSEPOLY])

        rect_path = Path(verts, codes, **params["mpl_rpath_kws"])
        self.rect_patch = PathPatch(rect_path, **params["mpl_rpatch_kws"])

        super().__init__((self.rect_patch,))

        if text:
            self.place_text(text, loc, params=params)


class CircleBlock(PatchGroup):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        radius = params["cp_block_radius"]
        width = params["cp_block_stroke_width"]
        if radius <= width:
            self.circle_patch = Circle(pos, radius, **params["mpl_cpatch_kws"])
        else:
            self.circle_patch = Wedge(
                pos, radius, 0, 360, width, **params["mpl_cpatch_kws"])

        super().__init__((self.circle_patch,))

        if text:
            self.place_text(text, loc, params=params)


class RoundCorner(CircleBlock):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        params.update(cp_block_radius=bpl_params["ap_block_line_width"] / 2)
        params.update(cp_block_stroke_width=bpl_params["ap_block_line_width"])

        super().__init__(pos, text, loc, params)


class Corner(RectangleBlock):
    def __init__(self, pos, text=None, loc="m", alpha=0, params=None):
        if params is None:
            params = bpl_params

        params = params.copy()
        params.update(rp_block_width=bpl_params["ap_block_line_width"])
        params.update(rp_block_height=bpl_params["ap_block_line_width"])

        mpl_rpatch_kws = bpl_params["mpl_rpatch_kws"].copy()
        mpl_rpatch_kws.update(alpha=alpha)
        params.update(mpl_rpatch_kws=mpl_rpatch_kws)

        super().__init__(pos, text, loc, params)


class Crossover(CircleBlock):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        params = params.copy()
        params.update(cp_block_radius=params["cop_block_radius"])
        params.update(cp_block_stroke_width=params["cop_block_radius"])

        super().__init__(pos, text, loc, params)


class Node(CircleBlock):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        params = params.copy()
        params.update(cp_block_radius=0)

        super().__init__(pos, text, loc, params)


class Line(PatchGroup):
    def __init__(self, it1, it2, loc1, loc2=None, type="-", params=None):
        if not isinstance(it1, PatchGroup):
            it1 = Node(it1)

        if not isinstance(it2, PatchGroup):
            it2 = Node(it2)

        if params is None:
            params = bpl_params

        if loc2 is None:
            loc2 = opposite_loc[loc1]

        t_angle = params["ap_block_tip_angle"]
        t_length = params["ap_block_tip_length"]
        arrow_lw = params["ap_block_line_width"]

        a1 = it1.get_anchor(loc1)
        a2 = it2.get_anchor(loc2)

        vect = a1 - a2
        angle = np.arctan2(vect[1], vect[0])
        normal = np.array([-np.sin(angle), np.cos(angle)])
        tang = np.array([np.cos(angle), np.sin(angle)])
        dist = np.linalg.norm(vect)

        if type == "->":
            arrow_p1 = a2 + np.array([np.cos(angle + t_angle),
                                      np.sin(angle + t_angle)]) * t_length
            arrow_p2 = a2 + np.array([np.cos(angle - t_angle),
                                           np.sin(angle - t_angle)]) * t_length
            arrow_edge = normal * (2 * t_length * np.sin(t_angle) - arrow_lw) / 2
            arrow_p11 = arrow_p1 - arrow_edge
            arrow_p22 = arrow_p2 + arrow_edge
            arrow_line = tang * (np.linalg.norm(vect) - t_length * np.cos(t_angle))
            arrow_p111 = arrow_p11 + arrow_line
            arrow_p222 = arrow_p22 + arrow_line

            verts = [a2, arrow_p1, arrow_p11, arrow_p111, arrow_p222, arrow_p22,
                     arrow_p2]

        elif type == "-" or type == "--" or "---":
            p1 = a1 + normal * arrow_lw / 2
            p2 = a2 + normal * arrow_lw / 2
            p3 = a2 - normal * arrow_lw / 2
            p4 = a1 - normal * arrow_lw / 2
            verts = [p1, p2, p3, p4]

        else:
            raise NotImplementedError()

        arrow_patch = PolygonPatch(verts, **params["mpl_apatch_kws"])

        if type == "--" or type == "---":
            seg_len = {"--": params["ap_block_seg_len"],
                       "---": dist / (params["ap_block_seg_num"] * 2 + 1)}[type]
            polys = list()
            i = 0
            while i * seg_len < dist * 2:
                poly = [
                    a1 + normal * arrow_lw / 2 - tang * seg_len * (2 * i + 1),
                    a1 + normal * arrow_lw / 2 - tang * seg_len * (2 * i + 2),
                    a1 - normal * arrow_lw / 2 - tang * seg_len * (2 * i + 2),
                    a1 - normal * arrow_lw / 2 - tang * seg_len * (2 * i + 1)]
                polys.append(PolygonPatch(poly).get_path())
                i += 1
            path = Path.make_compound_path(*polys)
            path = path.clip_to_bbox(arrow_patch.get_extents())
            arrow_patch = PathPatch(path, **params["mpl_apatch_kws"])

        super().__init__((arrow_patch,))


class Arrow(Line):
    def __init__(self, it1, it2, loc1, loc2=None, type="->", params=None):
        super().__init__(it1, it2, loc1, loc2, type, params)


class CompoundPatch(PatchGroup):
    def __init__(self, patches, params=None):
        if params is None:
            params = bpl_params

        polygons = list()
        for patch_group in patches:
            for gp in patch_group.geo_patches:
                polygons.append(Polygon(gp.get_path().vertices))

        polygon = unary_union(polygons)
        xy = [(x, y) for x, y in zip(polygon.exterior.coords.xy[0],
                                     polygon.exterior.coords.xy[1])]
        compound_patch = PolygonPatch(xy, **params["mpl_rpatch_kws"])

        super().__init__([compound_patch])

        txt_patches = list()
        for patch_group in patches:
            for tp in patch_group.txt_patches:
                txt_patches.append(tp)

        self.txt_patches = txt_patches


allowed_patches = (Patch, PatchGroup)
