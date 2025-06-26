'''
Created on Aug. 5, 2023

@author: cefect

merge basins together
'''


import os, warnings, psutil, math
from datetime import datetime

import numpy as np
import pandas as pd

import dask
 

from osgeo import gdal # Import gdal before rasterio
 
 
import xarray as xr
import rioxarray
import rioxarray.merge

from definitions import src_dir, asc_lib_d
from haz.hp import (
    view, get_temp_dir, today_str, init_log, dask_profile_func, dstr, dask_threads_func,
    get_directory_size
    )

from haz.rim2019.coms import  (
    out_base_dir, lib_dir, epsg_id, basin_chunk_d, dataArray_toraster, _filesearch,
    delay_dataArray_toraster
    )


def get_search_dirs_from_outs(
        search_dir=None,
        ):
    """build the search directories from outputs of quantile2_04.quantile_annMax_sparse()
    
    """
    
    if search_dir is None:
        search_dir=out_base_dir
        
    print(f'building from \n    {search_dir}')
    
    
    dir_d=dict()
    for root, dirs, files in os.walk(search_dir):
        if root.endswith('quant'):
            print(files)
        print(root)
        print(files)
    

def get_search_dirs_from_data(
        search_dir=None,
        basin_l=None,
        ):
    """build the search directories from data_LFS
    
    """
    
    if search_dir is None:
        search_dir=os.path.join(src_dir, 'data_LFS', 'haz', 'quant')
        
    if basin_l is None:
        basin_l=list(asc_lib_d.keys())
        
    print(f'building \'dir_d\' from \n    {search_dir}')
    
    
    dir_d=dict()
    for root, dirs, files in os.walk(search_dir):
        #print(f'{root}\n    {dirs}\n    {files}')
        root_fn = os.path.basename(root)
        
        if root_fn in basin_l:
            dir_d[root_fn] = root
            
            
    assert len(dir_d)==len(basin_l)
    print(f'got {len(dir_d)}\n    {dstr(dir_d)}')
    
    return dir_d
        
 

def merge_basins_quantiles(
        dir_d=None,
        out_dir=None,
        data_var='inundation_depth_annMax',
        encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},
        ):
    """collect quantiles from each basin and mosaic together"""
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if dir_d is None:
        dir_d = get_search_dirs_from_data()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir,'mosaic', today_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
    log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    log.info(f'merging from {len(dir_d)} basins')
    
    
    #===========================================================================
    # collect filepaths
    #===========================================================================
    fp_d = dict()
    for basinName, search_dir in dir_d.items():
        #log.info(f'{basinName}')
        
        #get teh filepath netCDF
        fp_d[basinName] = _filesearch(search_dir, ext='nc')
    assert len(fp_d)==len(dir_d)
    
    
    #===========================================================================
    # load into dataset
    #===========================================================================
    ds_d=dict()
    for basinName, fp in fp_d.items():
        #log.info(f'loading {basinName} from \n    {fp}')
        ds_i = xr.open_dataset(fp, engine="netcdf4", 
                               chunks={'x':1000, 'y':1000, 'quantile':1},
                               )#.transpose('x', 'y', ...)
        
        #check
        assert data_var in ds_i.data_vars
        #drop to dataarray
        #da_i = ds_i['inundation_depth_annMax'].rio.write_crs(f'EPSG:{epsg_id}').rio.write_nodata(-9999)
        
        log.info((f'loaded {basinName} from\n    {fp}\n    {ds_i.dims}' + 
                     f'\n    coors: {list(ds_i.coords)}\n    attrs:{ds_i.attrs}' 
                     ))
        ds_d[basinName] = ds_i
        
    
    #===========================================================================
    # write netcdf
    #===========================================================================
        
    #===========================================================================
    # merge per quantile------
    #===========================================================================
    #===========================================================================
    # rioxarray.merge.merge_datasets
    #=========================================================================== 
    da_d=dict()
    for quant in ds_i.coords['quantile'].values:
        log.info(f'mosaicing for q={quant}')
        #=======================================================================
        # da_l=list()
        # for basinName, ds in ds_d.items():
        #     da_l.append(
        #         ds.loc[{'quantile':quant}][data_var].drop('quantile').rio.write_crs(f'EPSG:{epsg_id}')
        #         )
        #=======================================================================
        da_l =  [ds.loc[{'quantile':quant}][data_var].drop('quantile').rio.write_crs(f'EPSG:{epsg_id}') for ds in ds_d.values()]
        
        da_d[quant] = rioxarray.merge.merge_arrays(da_l, method='max').rio.write_nodata(-9999)
        
        

    
    
    #===========================================================================
    # write raseter
    #===========================================================================
    for quant, da_m in da_d.items():
        
        shape_str = '-'.join([str(e) for e in da_m.shape])
        ofp = os.path.join(out_dir, f'quant_aMax_mosaic_{shape_str}_q{quant:.4f}_{today_str}'.replace('.', '') + '.tif') 
        log.info(f'queing write of {da_m.shape} q={quant} to \n    {ofp}')
        
        da_m.rio.to_raster(ofp, dtype='float32', compute=True, compress='LZW')
 
         
        #rioxarray.exceptions.MissingSpatialDimensionError: x dimension not found
        #ds_mi = rioxarray.merge.merge_datasets(ds_l, method='max', crs=f'EPSG:{epsg_id}')
        

    #===========================================================================
    # write complete netcdf
    #===========================================================================
    #concat mosaiced data array
    da_l = [da.assign_coords({'quantile':quant}).drop('spatial_ref') for quant, da in da_d.items()]
    
    ds_m = xr.concat(da_l, dim='quantile').to_dataset(name=data_var)
    
    ds_m.attrs['basin_l'] = list(fp_d.keys())
    ds_m.attrs['date'] = today_str
    
    #get filepath
    sh = ds_i[data_var].shape
    shape_str = '-'.join([str(e) for e in sh])
    ofp_nc = os.path.join(out_dir, f'quant_mosaic_q{len(da_d)}_{shape_str}_{today_str}.nc')
    log.info(f'to_netcdf {sh} to \n    {ofp_nc}') 
    
    #write
    ds_m.to_netcdf(ofp_nc, mode ='w', format ='netcdf4', engine='netcdf4', compute=True,
                     encoding={data_var:encoding})
 
    #===========================================================================
    # xr.combine_by_coords
    #===========================================================================
    #===========================================================================
    # """need to iterate on quantile as this is not merged"""
    # for quant in ds_i.coords['quantile'].values:
    #     ds_l =  [ds.loc[{'quantile':quant}] for ds in ds_d.values()]
    #     
    #     xr.combine_by_coords(ds_l, combine_attrs='drop')
    #     #ValueError: Resulting object does not have monotonic global indexes along dimension x
    #===========================================================================
    
    #===========================================================================
    # xr.combine_nested
    #===========================================================================
    #mosaic together (x,y) for single data_var
    #===========================================================================
    # """code runs... but result is wonky... xy are mixed?"""
    # ds_m = xr.combine_nested([list(ds_d.values())], concat_dim=['x','y'], 
    #                          join='outer',
    #                          fill_value=np.nan,
    #                          combine_attrs='drop',
    #                          )
    #===========================================================================
    

    
 
    
 
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now()-start).total_seconds(),
        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
        'output_MB':get_directory_size(out_dir),
        #'output_MB':os.path.getsize(ofp_nc)/(1024**2)
        }
    
    log.info(f'finished w/ \n {dstr(meta_d)} \n    {out_dir}')
    
    return out_dir
        
        
        
 
 
    
        
    
    
    

if __name__=="__main__":
    
    dir_d = get_search_dirs_from_data()
    
    merge_basins_quantiles()