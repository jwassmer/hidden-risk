'''
Created on Oct. 11, 2023

@author: cefect

write the nuts3 events as rasters
'''
import os, warnings, psutil, hashlib, pickle, gc
 
from datetime import datetime

from tqdm import tqdm

import numpy as np
import pandas as pd
idx = pd.IndexSlice
import sparse
import dask

import shapely.geometry 

import geopandas as gpd
import xarray as xr
import rioxarray

from haz.rim2019.parameters import epsg_id
from definitions import asc_lib_d

from haz.hp.basic import (
    view, get_temp_dir, today_str,   dstr,   get_directory_size,   get_log_stream
    )

#from haz.hp.dask import dataArray_todense
 
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, exclude_lib, cache_base_dir
    )

#from haz.rim2019.nuts3._03_nuts3_event_selec import get_select_meta_fp, get_select_haz_stack_fp

#from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray

def _apply_toRaster(da, out_dir=None, use_cache=True, basinName2=None):
 
    
    k = f'{basinName2}_{da.raster_index.values[0]}'
    assert da.rio.crs.to_epsg()==epsg_id
    #===========================================================================
    # extract meta
    #===========================================================================
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    #get some meta from the dataset
    d=dict()
    #d['basinName2'] = os.path.basename(out_dir)
    
    for k in ['raster_index']:
        d[k] = da.coords[k].values[0]
        
    #===========================================================================
    # set filepath
    #===========================================================================
 
    
    #fnstr = '_'.join([f'{k}-{v}' for k,v in d.items()])
    ri = d['raster_index']
    
    ofp = os.path.join(out_dir, f'rim2019_wd_{basinName2}_{ri:06d}.tif')
    
    #===========================================================================
    # precheck
    #===========================================================================
    assert da.max()>0, k
    assert da.notnull().sum()>0, k
    #===========================================================================
    # write
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
    
        da.encoding=dict()
        
        da.squeeze().reset_coords('raster_index', drop=True).rio.write_nodata(-9999
                  ).rio.to_raster(ofp, dtype='float32', compute=False, compress='LZW')
              
        print(d)
    else:
        print('file exists... skipping')
        
        
    #===========================================================================
    # post-check
    #===========================================================================
    assert os.path.getsize(ofp)>1e3, k
    
    return xr.DataArray(os.path.basename(ofp), coords=d, name='raster_fn')
              

def run_write_rasters( 
                  index_fp=None,
                  haz_index_fp=None,
                  out_dir=None, 
                  data_dir=None,
                  dev=False,use_cache=True,log=None,
                  ):
    """convert the datasets to rasters
    
    
    Params
    --------
    index_fp: str
        filepath to the analysis basin index
        links each basin to its event stack DataSet
        
    haz_index_fp: str
        filepath to the event index
        metadata on each event
        
    Writes
    ---------
    event rasters: geoTiff
        for each event in the stack, writes a raster
    
    updated hazard index: pd.DataFrame
        adds teh event raster filepath 
        
        
        
    """
    
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'nuts3', '04_rasters')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(index_fp), 'event_ds')
        
    if haz_index_fp is None:
        srch_dir =os.path.dirname(index_fp)
        fns = [e for e in os.listdir(srch_dir) if e.startswith(f'nuts3_collect_haz_dx') and e.endswith('.pkl')]
        assert len(fns)==1
        haz_index_fp = os.path.join(srch_dir, fns[0])
        
 
    if log is None:
        log = get_log_stream()
    
    #===========================================================================
    # load index file
    #===========================================================================
    index_gdf = gpd.read_file(index_fp, ignore_geometry=True)
    log.info(f'loaded {index_gdf.shape} basins fron \n    {index_fp}')
    
    """
    view(index_gdf.head(100))
    """
    #===========================================================================
    # loop and load on basins
    #===========================================================================
    res_d=dict()
    for i, row in tqdm(index_gdf.iterrows(), desc='basinName2'):
        #=======================================================================
        # defaults
        #=======================================================================
        basinName2 =row['basinName2']
        log.debug(f'on {basinName2}')
        #if basinName2=='elbe_lower': continue
        
        #=======================================================================
        # odi = os.path.join(out_dir, basinName2)
        # if not os.path.exists(odi):os.makedirs(odi)
        #=======================================================================
        
        #=======================================================================
        # load
        #=======================================================================
        ds_fp = os.path.join(data_dir, row['haz_ds_fp'])
        assert os.path.exists(ds_fp), basinName2
 
        with xr.open_mfdataset(ds_fp, parallel=True, engine='netcdf4') as ds:
            log.debug(f'loaded {ds.dims}' + 
                     f'\n    coors: {list(ds.coords)}' + 
                     f'\n    data_vars: {list(ds.data_vars)}' + 
                     f'\n    crs:{ds.rio.crs}'
                     f'\n    chunks:{ds.chunks}'
                     )
            
            #get the data array
            da =ds['inundation_depth']
            
            if dev:
                da = da.isel(raster_index=slice(0,2))
            
            #loop and write each raster to file
            res_d[basinName2] = da.groupby('raster_index', squeeze=False).apply(
                _apply_toRaster, out_dir=os.path.join(out_dir, 'tifs'), use_cache=use_cache, basinName2=basinName2
                ).compute().to_dataframe()
            
            log.debug(f'finished on {res_d[basinName2].shape}')
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_dx1 = pd.concat(res_d, names=['basinName2'])
    
    #join basins from index
    res_dx2 = res_dx1.join(index_gdf.loc[:, ['basinName', 'name']].rename(columns={'name':'basinName2'}).set_index('basinName2'), 
                           on='basinName2').set_index('basinName', append=True)
                           
    #join event meta
    haz_dx = pd.read_pickle(haz_index_fp).droplevel([1,2,4]) 
    res_dx3 = res_dx2.join(haz_dx,how='inner')
    
    #write
    ofp = os.path.join(out_dir, f'nuts3_haz_event_dx_{len(res_dx3)}_{today_str}.pkl')
    res_dx3.to_pickle(ofp)
    
    res_dx3.to_csv(os.path.join(out_dir, f'nuts3_haz_event_dx_{len(res_dx3)}_{today_str}.csv'))
    
    log.info(f'wrote {res_dx3.shape} to \n    {ofp}')
    
    #===========================================================================
    # wrap
    #===========================================================================
 
    
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'outdir_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 
                           
 
def run_write_rasters_perBasin(
                    basinName2, 
                  index_fp=None,
                  haz_index_fp=None,
                  out_dir=None, 
                  data_dir=None,
                  dev=False,use_cache=True,log=None,
                  ):
    """convert the datasets to rasters
    
    
    Params
    --------
    index_fp: str
        filepath to the analysis basin index
        links each basin to its event stack DataSet
        
    haz_index_fp: str
        filepath to the event index
        metadata on each event
        
    Writes
    ---------
    event rasters: geoTiff
        for each event in the stack, writes a raster
    
    updated hazard index: pd.DataFrame
        adds teh event raster filepath 
        
        
        
    """
    
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'nuts3', '04_rasters', basinName2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if index_fp is None:
        srch_dir = os.path.join(out_base_dir, 'nuts3', '04_collect')
        fns = [e for e in os.listdir(srch_dir) if e.startswith(f'nuts3_collect_meta') and e.endswith('.gpkg')]
        assert len(fns)==1
        index_fp = os.path.join(srch_dir, fns[0])
        
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(index_fp), 'event_ds')
        
    if haz_index_fp is None:
        srch_dir =os.path.dirname(index_fp)
        fns = [e for e in os.listdir(srch_dir) if e.startswith(f'nuts3_collect_haz_dx') and e.endswith('.pkl')]
        assert len(fns)==1
        haz_index_fp = os.path.join(srch_dir, fns[0])
        
 
    if log is None:
        log = get_log_stream()
        
    log = log.getChild(basinName2)
    #===========================================================================
    # load index file
    #===========================================================================
    index_gdf = gpd.read_file(index_fp, ignore_geometry=True)
    log.info(f'loaded {index_gdf.shape} basins fron \n    {index_fp}')
    
    """
    view(index_gdf.head(100))
    """
 
 
    log.info(f'on {basinName2}')
    row = index_gdf.loc[index_gdf['basinName2']==basinName2, :].iloc[0,:]
    #=======================================================================
    # load
    #=======================================================================
    ds_fp = os.path.join(data_dir, row['haz_ds_fp'])
    assert os.path.exists(ds_fp), basinName2
    
    with xr.open_mfdataset(ds_fp, parallel=True, engine='netcdf4') as ds:
        log.debug(f'loaded {ds.dims}' + 
                 f'\n    coors: {list(ds.coords)}' + 
                 f'\n    data_vars: {list(ds.data_vars)}' + 
                 f'\n    crs:{ds.rio.crs}'
                 f'\n    chunks:{ds.chunks}'
                 )
        
        #get the data array
        da =ds['inundation_depth']
        
        if dev:
            da = da.isel(raster_index=slice(0,2))
        
        #loop and write each raster to file
        res_dx = da.groupby('raster_index', squeeze=False).apply(
            _apply_toRaster, out_dir=os.path.join(out_dir, 'tifs'), use_cache=use_cache, basinName2=basinName2
            ).compute().to_dataframe()
        
        log.info(f'finished on {res_dx.shape}')
            
    #===========================================================================
    # wrap
    #===========================================================================
    #add some meta
    for k in ['basinName2', 'basinName', 'haz_ds_fp']:
        res_dx[k] = row[k]
    
 
                           
    #join event meta
    haz_dx = pd.read_pickle(haz_index_fp).droplevel([1,2,4]) 
    
    
    res_dx3 = res_dx.reset_index().set_index(haz_dx.index.names).join(haz_dx,how='inner')
    
    #write
    ofp = os.path.join(out_dir, f'nuts3_haz_event_dx_{len(res_dx3)}_{today_str}.pkl')
    res_dx3.to_pickle(ofp)
    
    res_dx3.to_csv(os.path.join(out_dir, f'nuts3_haz_event_dx_{len(res_dx3)}_{today_str}.csv'))
    
    log.info(f'wrote {res_dx3.shape} to \n    {ofp}')
    
    #===========================================================================
    # wrap
    #===========================================================================
 
    
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'outdir_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 



if __name__=="__main__":
    run_write_rasters_perBasin(
        'ems',
        index_fp=r'l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_collect\nuts3_collect_meta_7_7_20240125.gpkg',
        dev=False,
        use_cache=True,
        log = get_log_stream(),
        )
 