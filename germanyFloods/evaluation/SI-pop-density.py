# %%
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import rioxarray as riox
from rasterstats import zonal_stats


from matplotlib import patches as mpatches
from rasterio.enums import Resampling


from tqdm import tqdm
from src import FloodRaster as fr
from src import Plotting as pl
from germanyFloods import readFiles as rf

import cartopy.feature as cfeature


def read_raster(
    rescale=1 / 100,
    clip_gdf=None,
    raster_path="data/GHS/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif",
):
    if clip_gdf is None:
        raster = riox.open_rasterio(raster_path, masked=True).rio.reproject("epsg:4326")
    else:
        raster = (
            riox.open_rasterio(raster_path, masked=True)
            # .rio.reproject("epsg:4326")
            # .rio.clip(clip_geom, from_disk=True)
        )
        crs = raster.spatial_ref.crs_wkt
        clip_gdf = clip_gdf.to_crs(crs)
        clip_geom = clip_gdf.geometry.unary_union.convex_hull

        clipped = raster.rio.clip_box(*clip_geom.bounds)
        clipped = clipped.rio.clip([clip_geom])

    # Caluculate new height and width using downscale_factor
    new_width = raster.rio.width * rescale
    new_height = raster.rio.height * rescale

    # downsample raster
    rescaled = clipped.rio.reproject(
        clipped.rio.crs,
        shape=(int(new_height), int(new_width)),
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )

    return rescaled.rio.reproject("epsg:4326")


raster_path = "data/GHS/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif"
basins = gpd.read_file("data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson")

# %%

k = 1
catchment = basins.iloc[k : k + 1]["name"].values[0]

hospitals = rf.read_hospital_catchement2(
    catchment=catchment, population_kwd="prob_service_population"
)
hospitals = hospitals.sort_values(by="population_percentual_diff", ascending=False)
hospitals = hospitals.drop_duplicates(subset="node", keep="first")


population_raster = read_raster(raster_path=raster_path, clip_gdf=hospitals.geometry)

# %%

st_kwd = "mean"

hospitals = hospitals.dropna(subset=["geometry"])

stats = zonal_stats(
    hospitals.geometry,
    population_raster.data[0],
    affine=population_raster.rio.transform(),
    stats=st_kwd,
    nodata=np.nan,
    all_touched=False,
)


hospitals["pop"] = [s[st_kwd] for s in stats]
# %%

fig, ax = plt.subplots()

hospitals.plot("pop", legend=True, ax=ax)
# hospitals.boundary.plot(ax=ax, color="grey", linewidth=0.1, zorder=0)
# %%
fig, ax = plt.subplots(figsize=(12, 8))


ax.scatter(hospitals["pop"], hospitals["population_percentual_diff"])
ax.grid()
ax.set_xlabel("Population density")
ax.set_ylabel("$\Delta N(k_h)$")
ax.set_yscale("log")
ax.set_xscale("log")

# %%
