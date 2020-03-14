import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge, Circle, Rectangle, PathPatch, Patch, FancyArrowPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
import numpy as np


__all__ = ["bpl_params", "opposite_loc", "get_anchor", "PatchGroup",
           "RectangleBlock", "get_shift_to_align_b1_to_b2", "change_params",
           "place_patches", "CircleBlock", "cp", "Corner", "Node", "Line",
           "Arrow", "Crossover"]


bpl_params = dict(
    rp_block_width=8,
    rp_block_height=2,
    rp_block_stroke_width=0.125,
    cp_block_radius=0.5,
    cp_block_stroke_width=0.125,
    cop_block_radius=0.18,
    ap_block_tip_angle=0.5,
    ap_block_tip_length=0.45,
    ap_block_line_width=0.08,
    mpl_rpath_kws=dict(),
    mpl_rpatch_kws=dict(facecolor="black", edgecolor=None, linewidth=None),
    mpl_cpatch_kws=dict(facecolor="black", edgecolor=None, linewidth=None),
    mpl_tpath_kws=dict(usetex=mpl.rcParams['text.usetex'], size=1),
    mpl_tpatch_kws=dict(facecolor="black", edgecolor=None, linewidth=None),
)
opposite_loc = {"w": "e", "e": "w", "n": "s", "s": "n", "m":"m"}


def change_params(**kwargs):
    params = bpl_params.copy()
    params.update(kwargs)

    return params


def cp(**kwargs):
    return change_params(**kwargs)


def place_patches(patches=None, workspace=None, axis=None):
    if axis is None:
        axis = plt.gca()

    if not (workspace or patches):
        raise ValueError

    def add_patch(patch):
        if patch not in axis.patches:
            if isinstance(patch, PatchGroup):
                for p in patch.get_patches():
                    axis.add_patch(p)
            else:
                axis.add_patch(patch)

    allowed_patches = (Patch, PatchGroup)

    if patches is None:
        for v in workspace.values():
            if isinstance(v, allowed_patches):
                add_patch(v)

    else:
        for patch in patches:
            add_patch(patch)


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


def get_shift_to_align_b1_to_b2(b1, loc1, b2, loc2=None, pad_xy=(0, 0)):
    if loc2 is None:
        loc2 = opposite_loc[loc1]

    anchor1 = get_anchor(b1, loc1)
    anchor2 = get_anchor(b2, loc2)

    return anchor2 - anchor1 + np.array(pad_xy).flatten()


class PatchGroup:
    def __init__(self, geo_patches, text=None, loc="m"):
        self.geo_patches = [p for p in geo_patches]
        self.txt_patches = list()

        if text is not None:
            self.place_text(text, loc)

    def get_patches(self):
        return self.geo_patches + self.txt_patches

    def get_geo_extents(self):
        return PatchGroup.get_extents(self.geo_patches)

    def get_txt_extents(self):
        return PatchGroup.get_extents(self.txt_patches)

    @staticmethod
    def get_extents(patches):
        extents = [p.get_extents() for p in patches]
        x0 = min([e.x0 for e in extents])
        x1 = max([e.x1 for e in extents])
        y0 = min([e.y0 for e in extents])
        y1 = max([e.y1 for e in extents])

        return mpl.transforms.Bbox([[x0, y0], [x1, y1]])

    def place_text(self, text, loc="m", params=None):
        if params is None:
            params = bpl_params

        if len(self.geo_patches) == 0:
            raise NotImplementedError

        text_path = TextPath((0, 0), text, **params["mpl_tpath_kws"])
        shift = get_shift_to_align_b1_to_b2(text_path.get_extents(), loc,
                                            self.get_geo_extents())
        text_path = text_path.transformed(Affine2D().translate(shift[0],
                                                               shift[1]))
        self.txt_patches.append(
            PathPatch(text_path, **params["mpl_tpatch_kws"]))

    def get_geo_anchor(self, loc="m"):
        bbox = self.get_geo_extents()

        return get_anchor(bbox, loc)


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
                pos - dx - dy, pos - dxi - dyi, pos + dxi - dyi, pos + dxi + dyi,
                pos - dxi + dyi, pos - dxi - dyi, pos - dx - dy, pos - dx - dy]
            codes = ([Path.MOVETO] + [Path.LINETO] * 4 + [Path.MOVETO] +
                     [Path.LINETO] * 4 + [Path.MOVETO] + [Path.CLOSEPOLY])

        rect_path = Path(verts, codes, **params["mpl_rpath_kws"])
        self.rect_patch = PathPatch(rect_path, **params["mpl_rpatch_kws"])

        super().__init__((self.rect_patch,), text, loc)


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

        super().__init__((self.circle_patch,), text, loc)


class RoundCorner(CircleBlock):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        params.update(cp_block_radius=bpl_params["ap_block_line_width"] / 2)
        params.update(cp_block_stroke_width=bpl_params["ap_block_line_width"])

        super().__init__(pos, text, loc, params)


class Corner(RectangleBlock):
    def __init__(self, pos, text=None, loc="m", alpha=None, params=None):
        if params is None:
            params = bpl_params

        params = params.copy()
        params.update(rp_block_width=bpl_params["ap_block_line_width"])
        params.update(rp_block_height=bpl_params["ap_block_line_width"])

        if alpha is not None:
            mpl_rpatch_kws = bpl_params["mpl_rpatch_kws"].copy()
            mpl_rpatch_kws.update(alpha=0)
            params.update(mpl_rpatch_kws=mpl_rpatch_kws)

        super().__init__(pos, text, loc, params)


class Crossover(CircleBlock):
    def __init__(self, pos, text=None, loc="m", params=None):
        if params is None:
            params = bpl_params

        params = params.copy()
        params.update(cp_block_radius=bpl_params["cop_block_radius"])
        params.update(cp_block_stroke_width=bpl_params["cop_block_radius"])

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
        if params is None:
            params = bpl_params

        if loc2 is None:
            loc2 = opposite_loc[loc1]

        t_angle = params["ap_block_tip_angle"]
        t_length = params["ap_block_tip_length"]
        arrow_lw = params["ap_block_line_width"]

        a1 = it1.get_geo_anchor(loc1)
        a2 = it2.get_geo_anchor(loc2)

        vect = a1 - a2
        angle = np.arctan2(vect[1], vect[0])
        normal = np.array([-np.sin(angle), np.cos(angle)])
        tang = np.array([np.cos(angle), np.sin(angle)])

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

            verts = [a2, arrow_p1, arrow_p11, arrow_p111, arrow_p222, arrow_p22, arrow_p2, a2, a2]
            codes = [Path.MOVETO] + [Path.LINETO] * 7 + [Path.CLOSEPOLY]

        elif type == "-":
            p1 = a1 + normal * arrow_lw / 2
            p2 = a2 + normal * arrow_lw / 2
            p3 = a2 - normal * arrow_lw / 2
            p4 = a1 - normal * arrow_lw / 2
            verts = [p1, p2, p3, p4, p1, p1]
            codes = [Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY]

        else:
            raise NotImplementedError()

        arrow_path = Path(verts, codes)
        arrow_patch = PathPatch(arrow_path, **params["mpl_rpatch_kws"])

        super().__init__((arrow_patch,))


class Arrow(Line):
    def __init__(self, it1, it2, loc1, loc2=None, type="->", params=None):
        super().__init__(it1, it2, loc1, loc2, type, params)


if __name__ == "__main__":
    from shapely.geometry import Polygon
    from shapely.ops import cascaded_union
