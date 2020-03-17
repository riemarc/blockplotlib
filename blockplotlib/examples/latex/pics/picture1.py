from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from blockplotlib import *


fig = plt.figure()

text = TextPath((0,0), "Some text as pdf picture.",
                **bpl_params["mpl_tpath_kws"])
text_patch = PathPatch(text, **bpl_params["mpl_tpatch_kws"])

place_patches(workspace=locals())

save_figure()
write_bpl_tex_file()

plt.show()

