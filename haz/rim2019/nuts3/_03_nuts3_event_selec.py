'''
Created on Oct. 9, 2023

@author: cefect

identify the worst events per nuts 3 region
'''

 
import os, warnings, psutil, hashlib
 
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

from haz.hp.basic import (
    view, get_temp_dir, today_str, dstr,  get_directory_size, get_fp, get_log_stream
    )

from haz.hp.dask import (
    dask_profile_func, dask_threads_func,dataArray_todense, save_bounding_box_as_gpkg
    )
 
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, exclude_lib, cache_base_dir
    )

from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray


#===============================================================================
# helpers------
#===============================================================================
def get_select_meta_fp(basinName,
                     srch_dir=None):
    
    if srch_dir is None:
        srch_dir = os.path.join(out_base_dir,  '03_select', basinName)
        
 
    
    fns = [e for e in os.listdir(srch_dir) if e.startswith(f'{basinName}_nuts3_events_meta')]
    
    assert len(fns)==1, f'got multiple matches\n    {fns}'
    
    return os.path.join(srch_dir, fns[0])


def get_select_haz_stack_fp(basinName, srch_dir=None):
    
    if srch_dir is None:
        srch_dir = os.path.join(out_base_dir,  '03_select', basinName)
    
    #ofp = os.path.join(out_dir, f'{basinName}_nuts3_events_{fn_str}.pkl')
    
    fns = [e for e in os.listdir(srch_dir) if (e.startswith(f'{basinName}_nuts3_events') and e.endswith('.pkl'))]
    
    assert len(fns)==1
    
    return os.path.join(srch_dir, fns[0])
 
def get_slog(name, log):
    if log is None:
        log = get_log_stream()
        
    return log.getChild(name)

#===============================================================================
# runers-----
#===============================================================================
def run_select_events_per_zone(basinName='rhine',
                  nc_fp=None,
                  out_dir=None,temp_dir=None,
                  exclude_d=None,
                  zones_fp=None,
                  use_cache=True,
                  dev=False,log=None,
                  ):
    """identify the worst events for each zone in Nuts 3
    
    
    Params
    --------
    zones_fp: str
        filepath to Nuts3 polygons
        
    nc_fp: str
        filepath to sparse-xarray
        
    exclude_d: dict
        keys of events that should be excluded.
        e.g., rom those rastsers with max=99 (infinite ponding)
        
        
    TODO
    ------
    add tests
    
        
    """
    
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'nuts3', '03_select', basinName)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if temp_dir is None:
        temp_dir = os.path.join(cache_base_dir, 'nuts3', '03_select', basinName)
    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
 
    if nc_fp is None:
        nc_fp = get_sparse_fp(basinName)
        
    if exclude_d is None:
        if basinName in exclude_lib:
            exclude_d = exclude_lib[basinName]
    
    if zones_fp is None:
        from definitions import zones_fp
 
    #
    if log is None:
        log = get_log_stream()
    log = log.getChild(basinName)
 
    
    
    #===========================================================================
    # load------------
    #===========================================================================
    #===========================================================================
    # hazard dat
    #===========================================================================
    log.info(f'loading sparse-xarray from \n    {nc_fp}')
    log.debug(temp_dir)
    
 
    sds = load_sparse_xarray(nc_fp, fix_relative=True)
    
    #=======================================================================
    # make exclusions 
    #=======================================================================
    if not exclude_d is None:
        log.warning(f'dropping {np.array([len(v) for k,v in exclude_d.items()]).sum()} values from exclude_d')
        
 
        sds = sds.swap_dims({'sparse_index':'raster_index'}).drop_sel(
            exclude_d, errors='ignore' #ignore because we  used to have this dropping on nc_to_sparse
            ).swap_dims({'raster_index':'sparse_index'})
            
        #reset sparse index
        sds.coords['sparse_index'] = np.arange(0, len(sds.sparse_index))
        
    #===========================================================================
    # add rasterizeable data
    #===========================================================================
    """because we cant perform raster operations on teh sparse array, need to add some dummy data"""
    dims_l = ['x', 'y', 'spatial_ref']
    d = {e:sds.coords[e] for e in dims_l}
    da_empty = xr.DataArray(coords=d, dims=['x', 'y']).fillna(0.0
                              ).rio.write_crs(sds.rio.crs).rio.write_nodata(-9999
                                    ).rio.write_transform(sds.rio.transform())
                              
                              
    #add to the dataset
    sds['dummy'] = da_empty
    
    #===========================================================================
    # zones
    #===========================================================================
    zones_gdf_raw = gpd.read_file(zones_fp)    
    assert zones_gdf_raw.crs.to_epsg()==epsg_id
    
    #for this basin
    bx = zones_gdf_raw['catchment']==basinName
    assert bx.any()
    log.info(f'selected {bx.sum()}/{len(bx)} nuts3 for basinName={basinName}')
    
    zgdf = zones_gdf_raw.loc[bx, :]
    
    
    """do this later"""
    #drop coasta    
    #===========================================================================
    # bx = zgdf1['COAST_TYPE']==1.0
    # log.info(f'dropping {bx.sum()}/{len(bx)} with COAST_TYPE=1')
    # zgdf = zgdf1[~bx]
    #===========================================================================
    
    log.info(f'looping and selecting on {len(zgdf)}')
    
    
    #===========================================================================
    # loop and calc------
    #===========================================================================
    cnt=0
    res_d = dict()
    err_d=dict()
    empty_d=dict()
    for j, (i, row) in tqdm(enumerate(zgdf.iterrows()), desc='nuts3', total=len(zgdf)):
        log.debug(f'    {j}/{len(zgdf)} %s'%row['NAME_LATN'])
        
        #=======================================================================
        # compute stats from stack
        #=======================================================================
        uuid = hashlib.shake_256(f'{i}_{basinName}_{row}_{nc_fp}'.encode("utf-8")).hexdigest(12)    
 
        ofp_i = os.path.join(temp_dir, f'{i:04d}_{uuid}.pkl')
        
        if (not os.path.exists(ofp_i)) or (not use_cache):
            try:
                dx = get_max_haz_id_from_zone(row.geometry, sds, log=log.getChild(str(i)))
                if dx is None:
                    empty_d[j]=i
                else:
                    dx.to_pickle(ofp_i)                    
                    log.debug(f'    wrote {dx.shape} to \n    {ofp_i}')
                    
            except Exception as e:
                log.error(e)
                err_d[i] = str(e)
                

        else:
            log.info(f'    loading from cache')
            dx = pd.read_pickle(ofp_i)
        
        #=======================================================================
        # wrap
        #=======================================================================
        if not dx is None:
            assert len(dx)>0
            res_d[i]=dx
 
        cnt+=1
        
        if dev:
            if cnt>10:break
            
    log.info(f'got {len(res_d)} nuts w/ flooding. {len(empty_d)} without')
    assert len(res_d)+len(empty_d)+len(err_d)==len(zgdf)
    #===========================================================================
    # write result-----
    #===========================================================================
    rdx = pd.concat(res_d, axis=1, names=['nuts_index', 'metric'])
    
    #join some indexers
    rdx.columns=pd.MultiIndex.from_frame(rdx.columns.to_frame().reset_index(drop=True
                                ).join(zgdf.loc[:, ['id', 'NUTS_ID']], on='nuts_index'))
    
    
    
    fn_str = '_'.join([str(e) for e in rdx.shape])
    
    #to pick
    ofp = os.path.join(out_dir, f'{basinName}_nuts3_events_{fn_str}.pkl')
    rdx.to_pickle(ofp)
    
    #to csv
    ofp1 = os.path.join(out_dir, f'{basinName}_nuts3_events_{fn_str}.csv')
    rdx.to_csv(ofp1)
    log.info(f'wrote {rdx.shape} to \n    {ofp1}')
    #===========================================================================
    # errors
    #===========================================================================
    if len(err_d)>0:
        log.info(f'finished w/ {len(err_d)} errors')
        ergdf = pd.Series(err_d).rename('errors').to_frame().join(zgdf)
        
        ergdf = gpd.GeoDataFrame(ergdf.drop('geometry', axis=1), geometry=ergdf.geometry, crs=zgdf.crs)
        
        ergdf.to_file(os.path.join(out_dir, f'{basinName}_errors_{len(ergdf)}_{today_str}.gpkg'))
        
    #===========================================================================
    # misses
    #===========================================================================
    miss_dx = pd.Series(True, index=pd.Index(empty_d.values(), name='nuts_index'), name='empty'
                        ).to_frame().join(zgdf.loc[:, ['id', 'NUTS_ID']], on='nuts_index'
                                          ).set_index(['id', 'NUTS_ID'], append=True)
    
    #===========================================================================
    # write spatial meta
    #===========================================================================
    zgdf.index.name='nuts_index'
    
    log.info(f'finshed on {len(res_d)}')
    #assemble meta
    gdf1 = rdx.max().unstack('metric')
    
    #add misses
    gdf1 = gdf1.join(miss_dx, how='outer')
    gdf1['empty'] = gdf1['empty'].fillna(False).astype(bool)
    gdf1['wd_sum'].fillna(0.0, inplace=True)
    gdf1['wet_cnt'].fillna(0.0, inplace=True)
    
    gdf2 = gdf1.join(rdx.loc[:, idx[:, 'wd_sum']].droplevel([1,2,3], axis=1).count().rename('layer_count')
                     ).join(zgdf.drop(['NUTS_ID', 'id'], axis=1)).reset_index(drop=False).set_index('nuts_index')
                     
    gdf2['layer_count'].fillna(0.0, inplace=True)
        
    """
    gdf2.dtypes.values
    gdf1.index
    gdf2.index
    view(gdf2)
    """
    
    gdf3 = gpd.GeoDataFrame(gdf2.drop('geometry', axis=1), geometry=gdf2.geometry, crs=zgdf.crs)
 
    
    #write
    ofp = os.path.join(out_dir, f'{basinName}_nuts3_events_meta_{len(gdf1)}_{today_str}.gpkg')    
    gdf3.to_file(ofp)
    log.info(f'wrote {gdf1.shape} to \n    {ofp}')
    
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
    
 
        
        
 
    
def get_max_haz_id_from_zone(poly, sda, log=None,
                             #select_cnt = 5,
                             ):
    """id max event from polygo9n
    
    Params
    ----------
    select_cnt: int
        maximum number of extreme events to return"""
    
    assert poly.is_valid
    
    #===========================================================================
    # clip sparse by poly bbox
    #===========================================================================
    log.debug(f'clip_box({poly.bounds})') 
    wd_sparse_sda = sda['inundation_depth'].rio.clip_box(*poly.bounds)
    
    log.debug(f'bbox slice to get {wd_sparse_sda.shape} from %s' % (str(sda['inundation_depth'].shape)))
    
    #===========================================================================
    # clip by poly
    #===========================================================================
    #densify
    wd_sda = dataArray_todense(wd_sparse_sda)#.dropna(dim='sparse_index', how='all')
    
    #replace zeros with nulls
    """easier to drop dimensions this way"""
    wd_sda = wd_sda.where(wd_sda!=0, np.nan).dropna(dim='sparse_index', how='all')
    log.debug(f'dropping null layers got {wd_sda.shape}')
    
    #full clip (masks cells outside polygon)
    wd_clip_sda = wd_sda.rio.clip([shapely.geometry.mapping(poly)],all_touched=True, drop=False
                              ).dropna(dim='sparse_index', how='all')
    
    #check if any data is left in the clip
    if len(wd_clip_sda)==0:
        log.debug(f'    no valid layers... returning empty')
        return None
    
    log.debug(f'polygon slice to get {wd_clip_sda.shape} from {wd_sda.shape}')
 
    
    """
    import matplotlib.pyplot as plt
    wd_clip_sda.dropna(dim='sparse_index', how='any', thresh=5e5).shape
    
    wd_clip_sda.reset_coords(names=['raster_index', 'realisation'], drop=True
    ).isel(sparse_index=range(0,10)).plot.imshow(x='x',y='y', col='sparse_index', col_wrap=3)
    
    plt.show()
    """
    
    #===========================================================================
    # compute inundation metrics along sparse index
    #===========================================================================
    d = dict()
    d['wet_cnt'] = wd_clip_sda.count(dim=['x','y'], keep_attrs=True)
    d['wd_sum'] = wd_clip_sda.sum(dim=['x','y'], keep_attrs=True)
    
    rds = xr.Dataset(d)
 
 
    #===========================================================================
    # convert and rank
    #===========================================================================
    #convert data array dx
    """
                                                     wet_cnt       wd_sum
    sparse_index day   raster_index realisation                      
    0            7596  1            1               1438  1310.320312
    1            11371 3            2                149   110.601562
    """
    dx = rds.to_dataframe().set_index(['day', 'raster_index', 'realisation'], append=True
                                            ).drop('spatial_ref', axis=1)
                                            
 
    log.info(f'finished w/ {dx.shape}')
    
    return dx
    
    
                                            
 
 
 
    

if __name__=="__main__":
    
    run_select_events_per_zone(basinName='ems', log = get_log_stream(), 
                               dev=False, use_cache=True)
    
    
    
    
    
    
    
    
    
    
    
