from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from blockplotlib import *


fig = plt.figure()

text = TextPath((0,0), r"A formula in displaystyle as pdf picture: \\"
                       r"\begin{equation}"
                       r"f(x) = x^2"
                       r"\end{equation}",
                **bpl_params["mpl_tpath_kws"])
text_patch = PathPatch(text, **bpl_params["mpl_tpatch_kws"])

place_patches(workspace=locals())

save_figure()
write_bpl_tex_file()

plt.show()

