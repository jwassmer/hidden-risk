# %%
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import numpy as np
import contextily as ctx
import matplotlib.ticker as ticker

from src import RoadNetwork2 as rn
from src import FloodRaster as fr
from src import GeoModule as gm
from src import WetRoads as wr
from src import Plotting as pl


import matplotlib.pyplot as plt

path = "germanyFloods/data/wse2_clip_fdsc-r05_wsh.tif"
# path2 = "data_LFS/haz/rim2019/nuts3/tifs/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif"


raster = fr.read_raster(path)
xmin, ymin, xmax, ymax = 7.285, 52.7165, 7.287, 52.7175
small_raster = raster.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)


# %%

bbox = gm.bbox(north=52.75, south=52.69, west=7.27, east=7.32)

driving_tags = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    # "residential",
]

region = rn.RoadNetwork(
    osmpbf="ger-buffered-200km.osm.pbf",
    highway_filter=f"w/highway={','.join(driving_tags)}",
    gdf=bbox.gdf,
)
wr.water_depth(region, raster, polygon=bbox.shape)
edges = region.edges

bridge_gdf = wr.get_bridge_polys(region)
clipped = wr.mask_poly_raster(small_raster, bridge_gdf)

# clipped.plot()
# %%
fig, axs = plt.subplots(
    1,
    2,
    sharex=True,
    sharey=True,
    figsize=(10, 5),
    # constrained_layout=True,
)
cmap_b = plt.get_cmap("cividis_r")
cmap_b.set_under("none")
norm = plt.Normalize(vmin=1e-3, vmax=2.5)
sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm)

xmin, ymin, xmax, ymax = 7.285, 52.7165, 7.287, 52.7175

small_raster.plot(
    ax=axs[0],
    cmap=cmap_b,
    norm=norm,
    zorder=3,
    add_colorbar=False,
    rasterized=True,
)
clipped.plot(
    ax=axs[1],
    cmap=cmap_b,
    norm=norm,
    zorder=3,
    add_colorbar=False,
    rasterized=True,
)

ax_labels = [r"\textbf{a}", r"\textbf{b}"]
for i, ax in enumerate(axs):
    ax.text(
        0.0,
        1.09,
        ax_labels[i],
        transform=ax.transAxes,
        fontsize=24,
        verticalalignment="top",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid()
    bridge_gdf.plot(
        color="black",
        ax=ax,
        zorder=2,
        linewidth=15,
        edgecolor="black",
        label="bridge",
    )
    edges.plot(
        norm=norm, color="lightgrey", ax=ax, linewidth=15, zorder=1, label="road"
    )

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.4f}°E"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.4f}°N"))
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=45)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

axs[1].legend(loc="upper right")
cbar = fig.colorbar(
    sm,
    ax=axs,
    extend="both",
    orientation="horizontal",
    shrink=0.5,
)
cbar.set_label("Water depth [m]")
# fig.suptitle("Remove water on bridges")
# %%


fig.savefig(f"germanyFloods/figs/METHODS-wet-bridge.png", dpi=300, bbox_inches="tight")
# %%
fig.savefig(f"germanyFloods/figs/METHODS-wet-bridge.pdf", dpi=300, bbox_inches="tight")

# %%
