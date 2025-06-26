# %%
from src import RoadNetwork2 as rn
from src import Plotting as pl
from src import FloodRaster as fr

import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np

import pyproj
from shapely.ops import transform as shapely_transform

pl.mpl_params(fontsize=28)


# %%

vrt_file_path = "data_LFS/haz/rim2019/0303_downscale_20240629/rhine_lower/vrts/downscale_wsh_000081.vrt"

bbox_wgs84 = (6.82, 50.9, 7.1, 51.0)

# Open the VRT file
with rasterio.open(vrt_file_path) as src:
    # 1. Transform the bounding box to the CRS of the raster
    src_crs = src.crs  # Get the raster's CRS

    # Convert the bounding box from WGS84 to the source CRS
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", src_crs, always_xy=True
    ).transform
    bbox_geom_wgs84 = box(*bbox_wgs84)  # Create bounding box geometry in WGS84
    bbox_geom_src_crs = shapely_transform(
        project, bbox_geom_wgs84
    )  # Transform bbox to raster's CRS

    # 2. Clip the raster using the bounding box in the source CRS
    out_image, out_transform = mask(src, [bbox_geom_src_crs], crop=True)

    # Update the metadata to reflect the clipping
    out_meta = src.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    # Optionally save the clipped raster
    with rasterio.open(
        "germanyFloods/data/cologne-flood.tif", "w", **out_meta
    ) as out_dst:
        out_dst.write(out_image)


# %%

raster = fr.read_raster("germanyFloods/data/cologne-flood.tif")


# %%
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
    place="Cologne,Germany",
)

region.loads(weight="travel_time")

nodes, edges = region.nodes, region.edges
# %%

region_r = region.copy()

region_r.water_depth(raster)
wet_roads = region_r.edges[region_r.edges["water_depth"] > 0.3].index
print(f"There are {len(wet_roads)} wet roads")


# %%

region_r.remove_edges(wet_roads)

region_r.loads(weight="travel_time", threads=4)
nodes_r, edges_r = region_r.nodes, region_r.edges
# %%

fig, ax = plt.subplots(figsize=(8, 6))

cmap = mpl.cm.get_cmap("viridis")
cmap.set_under("lightgrey")
norm = mpl.colors.LogNorm(vmin=1e3, vmax=edges["load"].max())

ax.set_xlim(6.82, 7.1)
ax.set_ylim(50.9, 51.0)
ax.grid()

edges.sort_values("load", ascending=True, inplace=True)

edges.plot(ax=ax, column="load", cmap=cmap, norm=norm, legend=False, linewidth=2.5)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=0.5,
    pad=0.01,
    aspect=20,
    extend="both",
    orientation="horizontal",
)

cbar.ax.set_xlabel(r"$L_{ij}$")
# %%


delta = edges_r["load"] - edges["load"]

edges_r["delta"] = delta
edges_r.sort_values(by="delta", key=np.abs, ascending=True, inplace=True)


vmin, vmax = -np.abs(delta).max(), np.abs(delta).max()

fig, ax = plt.subplots(figsize=(8, 6))

cmap = mpl.cm.get_cmap("Blues")
norm = mpl.colors.Normalize(vmin=0.3, vmax=3)
raster.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)


cmap = mpl.cm.get_cmap("coolwarm")

norm = mpl.colors.SymLogNorm(linthresh=11000, linscale=1, vmin=vmin, vmax=vmax)

ax.set_xlim(6.82, 7.1)
ax.set_ylim(50.9, 51.0)
ax.grid()

removed_edges = edges.loc[edges.index.isin(wet_roads)]
removed_edges.plot(ax=ax, color="black", linewidth=2.5, zorder=3)

edges_r.plot(ax=ax, column=delta, cmap=cmap, norm=norm, legend=False, linewidth=2.5)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=0.5,
    pad=0.01,
    aspect=20,
    extend="both",
    orientation="horizontal",
)

cbar.ax.set_xlabel(r"$\Delta L_{ij}$")

# %%
