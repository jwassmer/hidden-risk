# %%
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import pickle
from shapely import wkt
from matplotlib.colors import ListedColormap

from matplotlib import patches as mpatches

from tqdm import tqdm
from src import FloodRaster as fr
from src import Plotting as pl
from germanyFloods import readFiles as rf

import cartopy.feature as cfeature
from matplotlib.legend_handler import HandlerTuple

catchment_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
)
path = "germanyFloods/data"

tail = pd.read_csv(f"{path}/worst_9_events.csv")
tail["geometry"] = tail["geometry"].apply(wkt.loads)
tail = gpd.GeoDataFrame(tail, geometry="geometry", crs="EPSG:4326")
tail.sort_values("population_percentual_diff", ascending=False, inplace=True)


markers = ["o", "s", "D", "v", "^", "p", "*", ">", "<"]

tab_cmap = mpl.cm.get_cmap("tab10")
colors = tab_cmap.colors
new_order = [
    colors[7],
    colors[1],
    colors[2],
    colors[3],
    colors[8],
    colors[0],
    colors[-1],
]
tab_cmap = ListedColormap(new_order)

tab_norm = mpl.colors.Normalize(vmin=0, vmax=7)

# %%
path = "germanyFloods/data/max_raster_dict.pkl"
with open(path, "rb") as f:
    max_raster_dict = pickle.load(f)

# %%
gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=0.08
)
gdf["catchment_code"] = gdf["catchment"].astype("category").cat.codes

# gdf = gdf[gdf["population_pre_flood"] > 1_000]
gdf.sort_values(by="population_percentual_diff").tail(20)


# %%

gdf[gdf["population_percentual_diff"].isna()]

# %%
agg_gdf = rf.aggregate_hospitals(
    gdf, col="population_percentual_diff", new_index="idxmax"
)


# %%

plt_key = "population_percentual_diff"

agg_gdf.sort_values(plt_key, ascending=True, inplace=True)
agg_gdf = agg_gdf.reset_index().drop_duplicates(subset="idxmax", keep="last")
print("There are", len(agg_gdf), " unique hospitals in total")
vmax = 5  # gdf[plt_key].max()
vmin = agg_gdf[plt_key].min()

fig, ax = plt.subplots(
    figsize=(12, 12),
    constrained_layout=True,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

gl = ax.gridlines(draw_labels=True, linestyle="-", alpha=0.5)
gl.right_labels = False
gl.top_labels = False
ax.set_extent([5, 15.5, 47, 54.5], crs=ccrs.PlateCarree())

# ax.axis("off")
fcmap = mpl.colormaps.get_cmap("cividis_r")

cmap = mpl.colormaps.get_cmap("Reds")
cmap.set_under("None")
cmap.set_bad("None")
norm = mpl.colors.LogNorm(vmin=1e-2, vmax=vmax)

agg_gdf.set_crs("EPSG:4326").plot(
    column=plt_key, linewidth=0.3, cmap=cmap, norm=norm, ax=ax
)


catchment_gdf = catchment_gdf.to_crs("EPSG:4326")
catchment_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=1)

for catchment, raster in max_raster_dict.items():
    print(catchment)
    listed_cmap = mpl.colors.ListedColormap(["None", cfeature.COLORS["water"]])
    if "upper" in catchment:
        major_catchment = catchment.replace("_upper", "")
    elif "lower" in catchment:
        major_catchment = catchment.replace("_lower", "")
    else:
        major_catchment = catchment
    river_raster = fr.read_raster(
        f"data_LFS/haz/rim2019/burned_domains/{major_catchment}.tif"
    )
    river_raster.values = np.where(
        river_raster.values < 9999, np.nan, river_raster.values
    )
    river_raster.plot(
        ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap=listed_cmap,
        norm=norm,
        rasterized=True,
    )

    raster.values = np.where(raster.values > 1e-3, 1, raster.values)
    raster.plot(
        ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap=fcmap,
        norm=norm,
        rasterized=True,
    )


for idx, row in catchment_gdf.iterrows():
    centroid = row.geometry.centroid
    name = row["name"]
    if "_" in name:
        name = " ".join(name.split("_")[::-1])
    name = " ".join(word.capitalize() for word in name.split())
    print(name)
    c = "darkgrey"
    if name == "Upper Rhine":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 1.9, centroid.y + 0.5),
            ha="right",
            zorder=5,
            color=c,
        )
    elif name == "Lower Rhine":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 0.1, centroid.y - 0.05),
            ha="center",
            va="top",
            zorder=5,
            color=c,
        )
    elif name == "Lower Elbe":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 0.85, centroid.y - 0.2),
            ha="center",
            va="top",
            zorder=5,
            color=c,
        )
    elif name == "Upper Elbe":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x, centroid.y - 0.2),
            ha="center",
            zorder=5,
            color=c,
        )
    elif name == "Donau":
        ax.annotate(
            text=rf"\textbf{{Danube}}",
            xy=(centroid.x, centroid.y),
            ha="center",
            zorder=5,
            color=c,
        )
    elif name == "Weser":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x, centroid.y - 0.2),
            ha="center",
            zorder=5,
            color=c,
        )

    else:
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x, centroid.y),
            ha="center",
            zorder=5,
            color=c,
        )

    ##


for i, geom in enumerate(tail.geometry.centroid):
    name = tail.iloc[i]["amenity_name"]
    if i == 0:
        offset = 0.075
    else:
        offset = 0

    catchment = tail.iloc[i]["catchment"]
    catchment_code = gdf[gdf["catchment"] == catchment]["catchment_code"].values[0]

    color = tab_cmap(catchment_code)

    ax.scatter(
        geom.x,
        geom.y + offset,
        color=color,
        zorder=3,
        s=300,
        marker=markers[i],
        edgecolor="black",
        linewidth=1,
    )

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=1 / 2,
    extend="both",
    pad=0.01,
    aspect=30,
    # orientation="horizontal",
    ticks=[1e-2, 1e-1, 1e0, 1e1, 1e2],
)
cbar.ax.set_ylabel(r"Maximum change in service population max($\Delta N(k_H)$)")

# cbar.ax.set_yticklabels([r"$1\%$", r"$10\%$", r"$100\%$", r"$1000\%$", r"$10000\%$"])


water_patch = mpatches.Patch(
    facecolor=cfeature.COLORS["water"],
    # edgecolor="lightblue",
    label="Permanent water",
    linewidth=0,
)

flood_patch = mpatches.Patch(
    facecolor=fcmap(255),
    # edgecolor="lightblue",
    label="Overbank flooding",
    linewidth=0,
)
basin_patch = mpatches.Patch(
    facecolor="none",
    edgecolor="black",
    label="Analysis basins",
    linewidth=1,
    # linestyle="--"
)


hospital_patchs = ()
for i in markers:
    hospital = ax.scatter(
        [],
        [],
        color="grey",
        marker=i,
        edgecolor="black",
        linewidth=0.5,
        s=200,
        label="Hospital",
    )
    hospital_patchs += (hospital,)

ax.legend(
    handles=[
        (water_patch),
        (flood_patch),
        (basin_patch),
        # hospital_patchs,
    ],
    labels=["Permanent water", "Overbank flooding", "Analysis basin"],
    # ["potatoes", "tomatoes"],
    # scatterpoints=1,
    # numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc="upper right",
    framealpha=1,
)

# %%


fig.savefig(
    f"germanyFloods/figs/RESULTS-ger-service_area.png",
    bbox_inches="tight",
    dpi=300,
)
# %%
fig.savefig(
    f"germanyFloods/figs/RESULTS-ger-service_area.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
