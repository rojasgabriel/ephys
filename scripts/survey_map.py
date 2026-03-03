#!/usr/bin/env -S uv run --script
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.widgets as mwidgets
import seaborn as sns


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot interactive survey map")
parser.add_argument("txt_file", help="path to the survey map txt file")
args = parser.parse_args()

plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

survey_map = pd.read_csv(args.txt_file, sep="\t")

pal = "rocket_r"
vmin = 1
vmax_init = 30
zum_min = int(survey_map["Zum"].min())
zum_max = int(survey_map["Zum"].max())
zum_init = (max(zum_min, 0), min(zum_max, 5000))

# Build figure: main axes + two slider axes at the bottom
fig = plt.figure(figsize=(7, 9))
ax = fig.add_axes((0.15, 0.28, 0.70, 0.65))
ax_vmax = fig.add_axes((0.15, 0.13, 0.70, 0.03))
ax_zum = fig.add_axes((0.15, 0.07, 0.70, 0.03))

sm = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax_init), cmap=pal)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="norm channel voltage")

vmax_slider = mwidgets.Slider(
    ax=ax_vmax,
    label="vmax",
    valmin=1,
    valmax=60,
    valinit=vmax_init,
    valstep=1,
)

zum_slider = mwidgets.RangeSlider(
    ax=ax_zum,
    label="Zum",
    valmin=zum_min,
    valmax=zum_max,
    valinit=zum_init,
)


def draw(_=None):
    vmax = float(vmax_slider.val)
    zmin, zmax = zum_slider.val

    df = survey_map[(survey_map["Zum"] >= zmin) & (survey_map["Zum"] <= zmax)]

    ax.clear()
    sns.stripplot(
        data=df,
        x="Shank",
        y="Zum",
        hue="Val",
        hue_norm=(vmin, vmax),
        palette=pal,
        size=3.5,
        alpha=0.5,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("shanks (M->L)")
    ax.set_ylabel("depth from probe tip (µm)")

    sm.set_clim(vmin, vmax)
    fig.canvas.draw_idle()


vmax_slider.on_changed(draw)
zum_slider.on_changed(draw)

draw()
plt.show()
