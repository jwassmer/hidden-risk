# %%
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_size(width="default", fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "default":
        width_pt = 510 * 2
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def mpl_params(fontsize=20):
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "sans-serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.titlesize": fontsize,
        "figure.titlesize": fontsize,
        "axes.labelsize": fontsize,  # LaTeX default is 10pt font.
        "font.size": fontsize,
        "legend.fontsize": fontsize,  # Make the legend/label fonts
        "xtick.labelsize": fontsize,  # a little smaller
        "ytick.labelsize": fontsize,
        "figure.figsize": set_size(),  # default fig size of 0.9 textwidth
        "pgf.preamble": "\n".join(
            [  # plots will use this preamble
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
            ]
        ),
        "figure.constrained_layout.use": True,  # set constrained_layout to True
    }

    mpl.rcParams.update(pgf_with_latex)


import platform

if platform.system() == "Darwin":

    mpl_params()

# %%
