# %%
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import numpy as np
import contextily as ctx
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from src import RoadNetwork2 as rn
from src import FloodRaster as fr
from src import GeoModule as gm
from src import WetRoads as wr
from src import Plotting as pl


import matplotlib.pyplot as plt

path = "germanyFloods/data/wse2_clip_fdsc-r05_wsh.tif"
# path = "data_LFS/haz/rim2019/nuts3/tifs/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif"

raster = fr.read_raster(path)
xmin, ymin, xmax, ymax = 7.24, 52.69, 7.34, 52.72
small_raster = raster.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)


# %%

bbox = gm.bbox(north=ymax, south=ymin, west=xmin, east=xmax)

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
    "residential",
]

region = rn.RoadNetwork(
    osmpbf="ger-buffered-200km.osm.pbf",
    highway_filter=f"w/highway={','.join(driving_tags)}",
    gdf=bbox.gdf,
)
region.water_depth(raster, polygon=bbox.shape)
edges = region.edges


bridge_gdf = wr.get_bridge_polys(region)
clipped = wr.mask_poly_raster(small_raster, bridge_gdf)

wet_roads = edges[edges["water_depth"] > 0.3]
region.remove_edges(wet_roads.index)
# %%
cmap = plt.get_cmap("cividis_r")
cmap.set_under("none")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)


fig = plt.figure(layout="constrained", figsize=(12, 12))
gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[1, 1])

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
plt.setp(ax1.get_yticklabels(), visible=False)

ax = fig.add_subplot(gs[1, 0:2])


ax_labels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}"]
for i, a in enumerate([ax0, ax1, ax]):
    a.text(
        0.0,
        1.08,
        ax_labels[i],
        transform=a.transAxes,
        fontsize=26,
        verticalalignment="top",
    )


small_raster.plot(
    ax=ax0,
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    add_labels=False,
    zorder=1,
    rasterized=True,
)
clipped.plot(
    ax=ax1,
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    add_labels=False,
    zorder=1,
    rasterized=True,
)
clipped.plot(
    ax=ax,
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    add_labels=False,
    zorder=2,
    rasterized=True,
)


xmin_b, ymin_b, xmax_b, ymax_b = 7.285, 52.7164, 7.287, 52.7175
for a in [ax0, ax1]:
    a.set_xlim([xmin_b, xmax_b])
    a.set_ylim([ymin_b, ymax_b])
    region.edges.plot(ax=a, color="grey", linewidth=15, zorder=0, label="Road")
    bridge_gdf.plot(
        color="black",
        ax=a,
        zorder=0,
        linewidth=15,
        edgecolor="black",
        label="Bridge",
    )
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.4f}°E"))
    a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.4f}°N"))
    a.tick_params(axis="x", rotation=30)
    a.xaxis.set_tick_params(labelsize=18)
    a.yaxis.set_tick_params(labelsize=18)

ax1.legend()


region.edges.plot(ax=ax, color="grey", zorder=1, label="Road")
region.edges[region.edges["removed"] == True].plot(
    ax=ax, color="red", label="Disrupted road", zorder=2
)


ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.2f}°E"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.3f}°N"))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

ax.legend()

for a in [ax0, ax1, ax]:
    a.grid(zorder=3)


cax = fig.add_axes([1, 0.25, 0.02, 0.5])
cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="min")

cbar.ax.set_ylabel(r"Water surface height (WSH) $w_{kl}$ [m]")
# %%

# fig.savefig("germanyFloods/figs/methods-wet-roads.pdf", bbox_inches="tight")

fig.savefig(f"germanyFloods/figs/METHODS-wet-roads.png", bbox_inches="tight", dpi=300)

# %%
fig.savefig(
    f"germanyFloods/figs/METHODS-wet-roads.pdf",
    bbox_inches="tight",
    dpi=300,
)

# %%
