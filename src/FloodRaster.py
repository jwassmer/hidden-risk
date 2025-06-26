# %%
import os
import numpy as np
import rioxarray as riox
from rioxarray import merge
from rasterio.enums import Resampling
import warnings
import pandas as pd


def read_raster(path, bounds=False):
    if bounds != False:
        xmin, ymin, xmax, ymax = bounds
        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                    ]
                ],
            }
        ]
        raster = (
            riox.open_rasterio(path, masked=True)
            .rio.reproject("epsg:4326")
            .rio.clip(geometries, from_disk=True)
        )
        return raster
    else:
        raster = riox.open_rasterio(path, masked=True).rio.reproject("epsg:4326")
    return raster


def merge_function(arrays):
    return np.nanmax(arrays, axis=0)


def get_tifs_of_event(
    basin, event, meta_df, path="data_LFS/haz/rim2019/0303_downscale_20240629"
):
    selected_event_df = meta_df.loc[basin, event]
    len_tifs = len(selected_event_df)
    print(f"Selected event has {len_tifs} tifs files.")
    tif_files = []
    for index, row in selected_event_df.iterrows():
        rel_dir = row["rel_dir"]
        file_name = row["wsh_fdsc_fn"]
        tif_files.append(f"{path}/{basin}/{rel_dir}/{file_name}")

    return tif_files


def read_total_event(
    catchment,
    event,
    rescale=1 / 2,
    bounds=False,
    path="data_LFS/haz/rim2019/0303_downscale_20240629",
):

    meta_df = pd.read_csv(f"{path}/concat_downscale_wsh_index_df.csv", index_col=[0, 5])

    # event_df = meta_df.loc[catchment, event]

    tifs = get_tifs_of_event(catchment, event, meta_df, path=path)

    if isinstance(bounds, np.ndarray):
        bounds = bounds.tolist()
    if bounds != False:
        xmin, ymin, xmax, ymax = bounds
        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                    ]
                ],
            }
        ]

    raster_list = []
    for tif in tifs:
        if bounds != False:
            try:
                raster = (
                    riox.open_rasterio(tif, masked=True)
                    .rio.reproject("epsg:4326")
                    .rio.clip(geometries, from_disk=True)
                )
            except:
                continue
        else:
            raster = riox.open_rasterio(tif, masked=True).rio.reproject("epsg:4326")

        # reproj = raster.rio.reproject("epsg:4326")
        raster_list.append(raster)

    if len(raster_list) == 0:
        warnings.warn("No rasters found in specified bounds. Returning None.")
        return None
    merged = merge.merge_arrays(raster_list, method="max")

    if rescale == 1 or rescale == False:
        return merged

    # Caluculate new height and width using downscale_factor
    new_width = raster.rio.width * rescale
    new_height = raster.rio.height * rescale

    # downsample raster
    rescaled = merged.rio.reproject(
        merged.rio.crs,
        shape=(int(new_height), int(new_width)),
        resampling=Resampling.bilinear,
    )

    return rescaled


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    catchment = "donau"
    event = "00547"
    raster = read_total_event(
        catchment, event, rescale_factor=1 / 2, bounds=[11.2, 47.4, 11.6, 47.7]
    )

    raster.plot()
# %%
