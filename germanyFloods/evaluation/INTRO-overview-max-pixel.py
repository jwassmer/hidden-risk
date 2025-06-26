# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import rasterio
import rioxarray
import xarray as xr

from rasterio.enums import Resampling
from scipy.ndimage import generic_filter
import numpy as np

osm_tiles = OSM()

from src import Plotting as pl
from src import FloodRaster as fr
from src import OSMparser as op
from src import GeoModule as gm
import os
import pickle

pl.mpl_params(fontsize=22)


# %%


# Define a function to apply over the raster using a moving window
def calculate_volume_within_radius(window):
    # Calculate and return the sum of volumes within the window
    return np.sum(window)


def downsampling(raster, downscale_factor):
    # Caluculate new height and width using downscale_factor
    new_width = raster.rio.width * downscale_factor
    new_height = raster.rio.height * downscale_factor

    # downsample raster
    down_sampled = raster.rio.reproject(
        raster.rio.crs,
        shape=(int(new_height), int(new_width)),
        resampling=Resampling.bilinear,
    )
    return down_sampled


def max_volume_location(raster, kernel_size=100):
    # Apply the function to the raster using a moving window
    downscale_factor = 0.1
    downscaled = downsampling(raster, downscale_factor)
    volume_raster = generic_filter(
        downscaled.data[0],
        calculate_volume_within_radius,
        size=int(kernel_size * downscale_factor),
        mode="constant",
        cval=np.nan,
    )

    # Find the location of the maximum volume
    y, x = np.unravel_index(np.nanargmax(volume_raster), volume_raster.shape)

    origin_x, _, _, origin_y = downscaled.rio.bounds()
    cell_width, cell_height = downscaled.rio.resolution()

    max_lat = origin_y + y * cell_height
    max_lon = origin_x + x * cell_width

    return max_lat, max_lon


def total_volumne_cubic_metres(raster):

    reproj = raster.rio.reproject("EPSG:3035")
    raster_array = reproj.data[0]

    cell_width, cell_height = reproj.rio.resolution()
    # Calculate volume per cell (assuming equal width and height for simplicity)
    cell_area = cell_width * cell_height * (-1)
    volume_array = raster_array * cell_area
    return np.nansum(volume_array)


def compute_max_raster(raster_paths):
    rasters = [fr.read_raster(path) for path in raster_paths]
    # Align rasters to the same resolution and extent
    # min_shape = (min(r.shape[1] for r in rasters), min(r.shape[2] for r in rasters))

    # Compute the maximum raster
    max_raster = np.maximum.reduce([r.data for r in rasters])

    # Save the max raster
    max_raster_da = xr.DataArray(
        max_raster, dims=rasters[0].dims, coords=rasters[0].coords
    )
    max_raster_da.rio.write_crs(rasters[0].rio.crs, inplace=True)

    return max_raster_da


# %%
catchment_dict = {
    "rhine_lower": "Lower Rhine",
    "donau": "Danube",
    "ems": "Ems",
    "elbe_lower": "Lower Elbe",
    "elbe_upper": "Upper Elbe",
    "rhine_upper": "Upper Rhine",
    "weser": "Weser",
}

path = f"data_LFS/haz/rim2019/03_rasters"
# concat_df = pd.read_csv(f"{path}/concat_downscale_wsh_index_df.csv", index_col=[0, 5])

catchments = list(catchment_dict.keys())

# %%
raster_dict = {}
for catchment in catchments:
    print(catchment)
    p = os.path.join(path, catchment)
    tif_files = []
    for root, dirs, files in os.walk(p):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))
    # tif_files = [f for f in os.listdir(p) if f.endswith(".tif")]
    # tif_paths = [os.path.join(p, f) for f in tif_files]

    max_raster = compute_max_raster(tif_files)
    raster_dict[catchment] = max_raster

max_volume_location_dict = {
    catchment: max_volume_location(raster) for catchment, raster in raster_dict.items()
}

# %%
# Save max_raster as a pickle file
with open("germanyFloods/data/max_raster_dict.pkl", "wb") as f:
    pickle.dump(raster_dict, f)

# %%
catchment_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
)
# catchment_gdf.to_crs("EPSG:4326", inplace=True)
catchment_gdf = catchment_gdf.to_crs(epsg=4326)

basins_gdf = gpd.read_file(
    "data_LFS/haz/rim2019/0303_downscale_20240629/zones/basin_polygons/rim2019_basins_merged_20230804.gpkg"
)
basins_gdf = basins_gdf.to_crs(epsg=4326)

nuts_gdf = gpd.read_file(
    "data_LFS/haz/rim2019/0303_downscale_20240629/zones/nuts/NUTS_3_DE_20231009.gpkg"
)
nuts_gdf = nuts_gdf.to_crs(epsg=4326)


# %%
# Create a figure and add a map
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()})
# set extend to focus on germany
ax.set_extent([5, 16, 47, 55], crs=ccrs.PlateCarree())

cmap = mpl.colormaps.get_cmap("Blues")
cmap.set_bad("None")
cmap.set_under("None")
norm = plt.Normalize(1e-3, 1)

# Add gridlines
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color="gray",
    alpha=0.5,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False


basins_gdf.boundary.plot(
    ax=ax,
    color="#F911E0",
    transform=ccrs.PlateCarree(),
    linestyle="-",
    zorder=3,
    linewidth=1,
)

catchment_gdf.plot(
    ax=ax,
    color="lightgrey",
    edgecolor="#00A05A",
    linewidth=5,
    transform=ccrs.PlateCarree(),
    zorder=2,
)

cologne_pos = (6.956944, 50.938056)
frankfurt_pos = (8.682127, 50.110924)
karslruhe_pos = (8.403653, 49.00689)
basel_pos = (7.588576, 47.558399)

ax.scatter(
    *cologne_pos,
    color="red",
    edgecolors="black",
    marker="*",
    s=350,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

ax.scatter(
    *frankfurt_pos,
    color="red",
    edgecolors="black",
    marker="*",
    s=350,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

ax.scatter(
    *karslruhe_pos,
    color="red",
    edgecolors="black",
    marker="*",
    s=350,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

ax.scatter(
    *basel_pos,
    color="red",
    edgecolors="black",
    marker="*",
    s=350,
    transform=ccrs.PlateCarree(),
    zorder=5,
)


nuts_gdf.boundary.plot(
    ax=ax, color="grey", transform=ccrs.PlateCarree(), linewidth=0.5, zorder=4
)

for idx, row in catchment_gdf.iterrows():
    centroid = row.geometry.centroid
    name = row["name"]
    if "_" in name:
        name = " ".join(name.split("_")[::-1])
    name = " ".join(word.capitalize() for word in name.split())
    print(name)

    if name == "Upper Rhine":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 1.9, centroid.y + 0.5),
            ha="right",
            zorder=5,
        )
    elif name == "Lower Rhine":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 0.1, centroid.y - 0.1),
            ha="center",
            va="top",
            zorder=5,
        )
    elif name == "Lower Elbe":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x + 0.85, centroid.y - 0.2),
            ha="center",
            va="top",
            zorder=5,
        )
    elif name == "Upper Elbe":
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x, centroid.y - 0.2),
            ha="center",
            zorder=5,
        )
    elif name == "Donau":
        ax.annotate(
            text=rf"\textbf{{Danube}}",
            xy=(centroid.x, centroid.y),
            ha="center",
            zorder=5,
        )
    else:
        ax.annotate(
            text=rf"\textbf{{{name}}}",
            xy=(centroid.x, centroid.y),
            ha="center",
            zorder=5,
        )


austria_loc = (13.4, 47.25)
swiss_loc = (8.3, 47.25)
france_loc = (7, 48.5)
netherlands_loc = (5.5, 51.5)
belgium_loc = (5.5, 50.5)
denmark_loc = (9.5, 54.7)
luxembourg_loc = (5.5, 49.5)
czech_loc = (14.75, 50.5)
poland_loc = (16, 52.5)

neighboring_countries = {
    "Austria": austria_loc,
    "Switzerland": swiss_loc,
    "France": france_loc,
    "Netherlands": netherlands_loc,
    "Belgium": belgium_loc,
    "Denmark": denmark_loc,
    "Luxembourg": luxembourg_loc,
    "Czech Republic": czech_loc,
    "Poland": poland_loc,
}

for country, loc in neighboring_countries.items():
    ax.annotate(
        text=rf"{country}",
        xy=loc,
        ha="center",
        zorder=5,
        color="darkgrey",
    )


# Create a custom legend
# patch = mpatches.Patch(color='lightgrey', edgecolor='#d95f02', label='Catchments')
# ax.legend(handles=[patch])

catch_patch = mpatches.Patch(
    facecolor="lightgrey",
    edgecolor="#00A05A",
    label="Analysis basins",
    linewidth=2.5,
    # linestyle="--"
)

rtm_patch = mpatches.Patch(
    facecolor="None",
    edgecolor="#F911E0",
    label="RFM basins",
    linestyle="-",
    linewidth=1.5,
)

cntry_patch = mlines.Line2D(
    [],
    [],
    color="black",
    label="National borders",
    linestyle="-",
    linewidth=1.5,
)

nuts_patch = mlines.Line2D(
    [],
    [],
    color="grey",
    label="NUTS 3",
    linestyle="-",
    linewidth=1,
)

water_patch = mpatches.Patch(
    facecolor=cfeature.COLORS["water"],
    # edgecolor="lightblue",
    label="Permanent water",
    linewidth=0,
)
flood_patch = mpatches.Patch(
    facecolor=cmap(255),
    # edgecolor="lightblue",
    label="Overbank flooding",
    linewidth=0,
)


# Optional: Add more features like rivers, lakes, etc.
ax.add_feature(cfeature.COASTLINE, zorder=3)  # Add coastlines
ax.add_feature(cfeature.LAKES, zorder=1)
ax.add_feature(cfeature.RIVERS, zorder=3)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS, zorder=3)  # Add country borders


ax.legend(
    handles=[
        catch_patch,
        rtm_patch,
        nuts_patch,
        cntry_patch,
        water_patch,
        flood_patch,
    ],
    loc="upper right",
    framealpha=1,
)

max_width, max_height = 0.3, 0.4
axins_dict = {
    "ems": [-0.25, 4 * 1 / 7, 0.2, 1 / 7],
    "rhine_lower": [-0.25, 2 * 1 / 7, 0.2, 1 / 7],
    "rhine_upper": [-0.25, 0, 0.2, 1 / 7],
    "weser": [-0.25, 1 - 1 / 7, 0.2, 1 / 7],
    "elbe_lower": [1.0, 1 - 1 / 7, 0.2, 1 / 7],
    "elbe_upper": [1.0, 0.5 - 1 / 14, 0.2, 1 / 7],
    "donau": [1.0, 0, 0.2, 1 / 7],
}
for j, (catchment, raster) in enumerate(raster_dict.items()):
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
    # river_raster.values = np.where(river_raster.values == 9999, 1, river_raster.values)
    river_raster.plot(
        ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap=listed_cmap,
        norm=norm,
        rasterized=True,
    )

    max_lat, max_lon = max_volume_location_dict[catchment]

    xmin, xmax = max_lon - max_width / 2, max_lon + max_width / 2
    ymin, ymax = max_lat - max_height / 2, max_lat + max_height / 2

    axins = ax.inset_axes(
        axins_dict[catchment],
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        transform=ax.transAxes,
        projection=ccrs.PlateCarree(),
    )
    print(catchment, xmin, xmax, ymin, ymax)

    axins.set_extent(
        [xmin, xmax, ymin, ymax],
        crs=ccrs.PlateCarree(),
    )
    ax.indicate_inset_zoom(axins, edgecolor="black")

    raster.values = np.where(raster.values > 1e-3, 1, raster.values)
    raster.plot(
        ax=axins,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        add_labels=False,
        zorder=2,
        rasterized=True,
    )
    river_raster.plot(
        ax=axins,
        add_colorbar=False,
        add_labels=False,
        cmap=listed_cmap,
        norm=norm,
        rasterized=True,
    )

    # axins.axis("off")

    nuts_gdf.boundary.plot(
        ax=axins, color="grey", transform=ccrs.PlateCarree(), linewidth=0.5, zorder=4
    )

    axins.add_feature(cfeature.COASTLINE, zorder=3)  # Add coastlines
    axins.add_feature(cfeature.LAKES, zorder=1)
    axins.add_feature(cfeature.RIVERS, zorder=3)
    axins.add_feature(cfeature.LAND)
    axins.add_feature(cfeature.OCEAN)
    axins.add_feature(cfeature.BORDERS, zorder=3)

    raster.plot(
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        add_labels=False,
        zorder=2,
        rasterized=True,
    )


# %%
fig.savefig("germanyFloods/figs/INTRO-germany-basins.png", bbox_inches="tight", dpi=300)

# %%
fig.savefig("germanyFloods/figs/INTRO-germany-basins.pdf", bbox_inches="tight", dpi=300)

# %%
