# %%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines  # i
import cartopy.crs as ccrs

from src import RoadNetwork2 as rn
from src import FloodRaster as fr
from src import GeoModule as gm
from src import WetRoads as wr
from src import Plotting as pl
from src import EmergencyModule as em

# pl.mpl_params(fontsize=22)

path = "germanyFloods/data/wse2_clip_fdsc-r05_wsh.tif"
# path2 = "data_LFS/haz/rim2019/nuts3/tifs/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif"

raster = fr.read_raster(path)
# %%

bbox = gm.bbox(north=52.9, south=52.6, west=7.1, east=7.6)

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
region.add_pois("amenity", "hospital")
wr.water_depth(region, raster, polygon=bbox.shape)
region.loads(threads=5, weight="travel_time")
region.effective_spillover_velocities(0.1)

region_r = region.copy()
region_r.remove_edges(region_r.edges[region_r.edges["water_depth"] > 0.3].index)

region_r.loads(threads=5, weight="travel_time")
region_r.effective_spillover_velocities(0.1)
# %%

# catchement_gdf, catchement_gdf_r = em.access_gdfs(region, region_r)

catchement_gdf = em.get_access_gdf2(region)
catchement_gdf_r = em.get_access_gdf2(region_r)
catchement_gdf["min_dist"], catchement_gdf_r["min_dist"] = em.dists_to_raster(
    region, raster
), em.dists_to_raster(region_r, raster)

# %%

fig, axs = plt.subplots(
    1,
    2,
    layout="constrained",
    figsize=(12, 12),
    sharex=True,
    sharey=True,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

vmin, vmax = 40_000, 50_000
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# cmap.set_under("white", 1)
# cmap.set_over("red", 1)

# titles = ["Pre flood", "Post flood"]
data = (
    region_r.edges["spillover_load"] * region_r.edges["spillover_travel_time"] / 60 / 60
    - region.edges["spillover_load"] * region.edges["spillover_travel_time"] / 60 / 60
)
cmap_edg = mpl.cm.coolwarm
norm_edg = mpl.colors.SymLogNorm(linthresh=1e-2, vmin=-1e1, vmax=1e1)

labels = [r"$\textbf{a}$", r"$\textbf{b}$"]
for i, r in enumerate([region, region_r]):
    ax = axs[i]
    ax.text(
        0.055,
        0.99,
        labels[i],
        transform=ax.transAxes,
        fontsize=26,
        va="top",
        ha="right",
    )

    hospitals = r.nodes[region.nodes["amenity"] == "hospital"]
    edges = r.edges

    edges.plot(
        ax=ax, color="black", cmap=cmap_edg, norm=norm_edg, linewidth=0.5, zorder=2
    )
    hospitals.plot(
        ax=ax,
        color="lightgrey",
        markersize=250,
        marker="P",
        zorder=3,
        edgecolor="black",
    )

    catchement_gdf = em.get_access_gdf2(r)
    xmin, ymin, xmax, ymax = catchement_gdf.total_bounds
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle="-", alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}
    if i == 1:
        gl.left_labels = False

    catchement_gdf.sort_values("prob_service_population", inplace=True)

    catchement_gdf.plot(
        ax=ax,
        column="prob_service_population",
        cmap=cmap,
        norm=norm,
        edgecolor="grey",
        linewidth=2,
        zorder=1,
        alpha=0.9,
    )
    # if i == 1:
    #    ax.set_title(titles[i], y=1.00, x=0.4)
    # else:
    #    ax.set_title(titles[i], y=1.00)
removed_edges = region.edges[region.edges["water_depth"] > 0.3]
removed_edges.plot(ax=axs[1], color="red", linewidth=4, zorder=1)

number_of_ticks = 5
tick_interval = (vmax - vmin) / (number_of_ticks - 1)

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=axs,
    orientation="vertical",
    pad=0.01,
    aspect=30,
    shrink=3 / 8,
    ticks=[vmin + i * tick_interval for i in range(number_of_ticks)],
)
cbar.ax.set_ylabel("Service population of hospital $N(k_H)$")

roads = mlines.Line2D([], [], color="black", label="Road", linewidth=1)
froads = mlines.Line2D([], [], color="red", label="Disrupted road", linewidth=2)

hosp_patch = axs[1].scatter(
    [],
    [],
    marker="P",
    color="lightgrey",
    s=250,
    label=r"Hospital $k_H$",
    edgecolor="black",
)

ax.legend(
    handles=[hosp_patch, roads, froads],
    loc="upper right",
    bbox_to_anchor=(1.1, 1.15),
    framealpha=1,
)
# %%

fig.savefig(
    f"germanyFloods/figs/METHODS-service-area.png",
    bbox_inches="tight",
    dpi=300,
)


# %%
fig.savefig(
    f"germanyFloods/figs/METHODS-service-area.pdf",
    bbox_inches="tight",
    dpi=300,
)

# %%
