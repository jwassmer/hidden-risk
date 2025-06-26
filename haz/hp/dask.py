'''
Created on Jan. 10, 2024

@author: cef
'''

from haz.hp.basic import *

from dask.distributed import LocalCluster as Cluster
from dask.distributed import Client
from dask.diagnostics import ResourceProfiler, visualize
import dask
from dask.diagnostics import ProgressBar

import xarray as xr


import geopandas as gpd
from shapely.geometry import Polygon

# supress some warnings
from bokeh.util.warnings import BokehUserWarning
import warnings

warnings.simplefilter(action="ignore", category=BokehUserWarning)

# ===============================================================================
# dask runners--------
# ===============================================================================
def dask_profile_func(func, *args, threads_per_worker=16, n_workers=1, **kwargs):
    with Client(
        threads_per_worker=threads_per_worker,
        n_workers=n_workers,
        memory_limit="auto",
        processes=False,
    ) as client:
        print(f" opening dask client {client.dashboard_link}")
        webbrowser.open(client.dashboard_link)

        with ResourceProfiler(dt=0.25) as rprof:
            func(*args, **kwargs)

        # profile results
        _wrap_rprof(rprof)

        """seems to be ignoring the filename kwarg"""
        """this also doesn't fix it
        os.chdir(os.path.expanduser('~'))"""

        rprof.visualize(
            # filename=os.path.join(os.path.expanduser('~'), f'dask_ReserouceProfile_{today_str}.html'),
            # filename=os.path.join(wrk_dir, f'dask_ReserouceProfile_{today_str}.html')
        )

    return rprof


def dask_threads_func(func, n_workers=None, **kwargs):
    pbar = ProgressBar(
        5.0, dt=1.0
    )  # show progress bar for computatinos greater than 5 secs
    pbar.register()

    # zarr_rechunker(**kwargs)
    with dask.config.set(scheduler="threads", n_workers=n_workers):
        func(**kwargs)


def _wrap_rprof(rprof):
    # initialize variables to store the maximum values
    max_mem = 0
    max_cpu = 0
    # iterate over the results to find the maximum values
    for result in rprof.results:
        max_mem = max(max_mem, result.mem)
        max_cpu = max(max_cpu, result.cpu)

    total_time = rprof.results[-1].time - rprof.results[0].time
    # print the maximum values
    print(
        f"total_time={total_time:.2f} secs, max_mem={max_mem:.2f} MB, max_cpu={max_cpu:.1f} %"
    )


# ===============================================================================
# SPARSE------
# ===============================================================================
def dataArray_todense(da):
    """convert a sparse dataArray to a dense one"""

    return xr.DataArray(da.data.todense(), dims=da.dims, coords=da.coords)

#===============================================================================
# RIOXARRAY--------
#===============================================================================
def save_bounding_box_as_gpkg(raster, filename):
    """
    Save the bounding box of a rioxarray as a GPKG polygon.

    Parameters:
    raster (rioxarray.raster_array.RasterArray): The rioxarray object.
    filename (str): The filename for the output GPKG file.
    """
    # Get the bounding box of the raster
    bbox = raster.rio.bounds()

    # Create a Polygon from the bounding box
    bbox_polygon = Polygon([
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
        (bbox[2], bbox[1])
    ])

    # Create a GeoDataFrame from the Polygon
    gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=raster.rio.crs)

    # Save the GeoDataFrame as a GPKG file
    gdf.to_file(filename, driver='GPKG')
    
    print(f'wrote to \n    {filename}')
    
    

def dataArray_toraster(da, ofp, compute=True):
    
    """delay not working... seems to always compute"""
    
    return da.rio.write_crs(f'EPSG:{epsg_id}'
                    ).rio.write_nodata(-9999
                    ).rio.to_raster(ofp, dtype='float32', compute=compute, compress='LZW')
    
@dask.delayed
def delay_dataArray_toraster(da, ofp):
    
    
    return da.rio.write_crs(f'EPSG:{epsg_id}'
                    # ).rio.write_nodata(-9999
                    ).rio.to_raster(ofp, dtype='float32', compute=True, 
                                    compress='LZW')