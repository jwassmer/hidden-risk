'''
Created on Jan. 14, 2024

@author: cef
'''

#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, psutil, pickle, tempfile, traceback, logging
import multiprocessing
from datetime import datetime

import numpy as np
import numpy.ma as ma
import pandas as pd

from tqdm import tqdm
import geopandas as gpd
import shapely.geometry as sgeo
from pyproj import CRS

import rasterio as rio
from rasterio.enums import Resampling, Compression
from rasterio.warp import calculate_default_transform, reproject, Resampling

import xarray as xr
import rioxarray


from haz.hp.basic import (
    today_str, view, get_log_stream, get_directory_size, get_new_file_logger, get_tqdm_disable,
    )


import config
from haz.rim2019.parameters import  out_base_dir, wet_wsh_thresh
from haz.rim2019.downscale._02_fines import _get_coords

from fdsc.control import Dsc_Session

#===============================================================================
# helpers------
#===============================================================================
 

def get_slog(name, log):
    if log is None:
        log = get_log_stream()
        
    return log.getChild(name)
        

get_hash = lambda x:hashlib.shake_256(x.encode("utf-8")).hexdigest(8)





def run_downscale_fines(
        basinName2,
        fine_ds_dir=None,
        fp_l=None,
        
        raster_index_l=None,
        
        max_fail_cnt=8,
        dev_stack_cnt=None,
        
        processes=None,
        
        log=None, out_dir=None,  tmp_dir=None,use_cache=True, 
        **kwargs):
    """
    
    TODO: check output index
    
    Parms
    ---------
    fine_ds_dir: str
        directory with WSE+DEM DataSource netCDF files for this basinName2
        
    raster_index: list, optoinal
        list of raster indexes to process (mostly for testing)
    
        
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    log = get_slog('fdsc', log)
    
    if out_dir is None: out_dir = os.path.join(out_base_dir, 'downscale', '03fdsc', basinName2)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    if tmp_dir is None: 
        from haz.rim2019.parameters import tmp_dir
        tmp_dir = os.path.join(tmp_dir, 'downscale', '03fdsc', basinName2)
    
    
    
    _kwargs = dict(out_dir=out_dir, use_cache=use_cache, tmp_dir=tmp_dir)
        
    index_coln_l=['raster_index', 'dem_tile_x', 'dem_tile_y']
    #===========================================================================
    # load WSE-DEM fine stack----
    #===========================================================================
    if fp_l is None:
        assert os.path.exists(fine_ds_dir), fine_ds_dir
        
        #get filepaths
        fp_l_raw = [os.path.join(fine_ds_dir, e) for e in os.listdir(fine_ds_dir) if e.endswith('.nc')]
        
        #load meta
        log.info(f'loading meta from {len(fp_l_raw)} fine stacks')
        index_d = dict()
        for i, fp in tqdm(enumerate(fp_l_raw), total=len(fp_l_raw), desc='fine_meta', disable=get_tqdm_disable()):
            with xr.open_dataset(fp,engine='netcdf4') as ds: 
                index_d[i] = {**_get_coords(ds, index_coln_l=index_coln_l), 
                              **{'fn':os.path.basename(fp), 'ctime':os.path.getctime(fp)}} 
                
                
        index_df = pd.DataFrame.from_dict(index_d).T #.set_index('raster_index')
        
        index_df.loc[:, 'raster_index'] = index_df['raster_index'].map(tuple)
        index_dx = index_df.set_index(index_coln_l).sort_values('ctime', ascending=False) #newest at teh top
        bx = index_dx.index.duplicated()
        if bx.any():
            log.warning(f'got {bx.sum()}/{len(bx)} duplicated entries... taking the newest')
            index_dx = index_dx.loc[bx, :]
        
     
        
        #dev break
        if __debug__ and (not dev_stack_cnt is None):
            cnt = len(index_dx)
            index_dx = index_dx.iloc[0:dev_stack_cnt, :]
            log.warning(f'dev_stack_cnt clipped from {cnt} to {len(fp_l_raw)}')
            
     
        
        log.info(f'loading {len(index_dx)} .nc files from fine dataSource \n    {fine_ds_dir}')
        
        fp_l = [os.path.join(fine_ds_dir, e) for e in index_dx['fn'].values]
    
    #===========================================================================
    # loop on each DEM tile
    #===========================================================================
    
    """could maybe do this more elegantly by combining everyting with Xarray,
    then iterating in chunks... but seems simpler to just iterate over the nc files
    """
       
    fail_cnt=0
    res_d=dict()
    fail_d=dict()
    
    
    #===========================================================================
    # single--------
    #===========================================================================
    if processes is None: processes=1
    if processes<=1:

        for i, fp in tqdm(enumerate(fp_l), total=len(fp_l), desc='fdsc', disable=get_tqdm_disable()):            

            
            log.debug(f'\n {i+1}/{len(fp_l)} on {os.path.basename(fp)}\n--------------------\n\n    {fp}')
            
            _kwargs['log'] = log.getChild(f'{i}')
            #=======================================================================
            # load dataSet
            #=======================================================================
            """parallizing with groupby.map doesnt seem to be working
            also makes error reporting more difficult
            should use a simple groupby
            """
            with xr.open_dataset(fp,
                                 engine='netcdf4',
                                 #chunks={'x':-1, 'y':-1, 'raster_index':1}, #single spatal, 1-per layer
                                 #parallel=False,
                                 ) as ds:
                
                log.debug(f'loaded {ds.dims}' + 
                         f'\n    coors: {list(ds.coords)}' + 
                         f'\n    data_vars: {list(ds.data_vars)}' + 
                         #f'\n    crs:{ds.rio.crs}'
                         f'\n    chunks:{ds.chunks}'
                         )
                
                #===================================================================
                # prep
                #=================================================================== 
                #swap in nulls
                ds = ds.where(ds!=-9999, np.nan).round(3)
                
                #load coordinates to memory
                """not sure why this is required"""
                for k,v in ds.coords.items():
                    v.load()
                
                #===================================================================
                # #clip to raster incidies
                #===================================================================
                if not raster_index_l is None:
                    #get just those on the index
                    ri_ar = ds.coords['raster_index'].values
                    bx = np.isin(ri_ar, raster_index_l)
                    if not bx.any():
                        log.warning(f'no indicies found in raster_index_l.... skipping')
                        continue
                    elif bx.all():
                        pass #everything is selected
                    else:
                        log.debug(f'selecting {bx.sum()}/{len(bx)} raster_index')              
                        ds = ds.sel(raster_index=ri_ar[bx]) 
                        
                    
                ri_l =  ds.coords['raster_index'].values.tolist()
                if isinstance(ri_l, int): ri_l = [ri_l] 
                log.debug(f'on {len(ri_l)} raster_index: {ri_l}')           
                
                #===================================================================
                # compute on raster_index stack---------
                #===================================================================
                try:
    
                    #===================================================================
                    # #load persistent layers
                    #===================================================================
        
                    #ds.coords.load()
                    ds['DEM'].load()            
                    ds['WBDY'].load()
                    
                    #===================================================================
                    # call downscaler per raster_index
                    #===================================================================
                
                    kwargs1 = dict(basinName2=basinName2, **_kwargs, **kwargs)
                    
    
     
                    #workaround for single index
                    if isinstance(ds.coords['raster_index'].values.tolist(), int): 
                        rda =  _apply_wse_downscale(ds, **kwargs1)
                        
                        #reformat to match groupby.map
                        s = pd.Series({**{rda.name:rda.item()}, **{k:v.item() for k,v in rda.coords.items()}}, name=i)
                        res_d[i] = s.to_frame().T #.set_index('raster_index', drop=True)
                        
                    #multi-raster_index
                    else:
                    
                        res_d[i] = ds.groupby('raster_index', squeeze=False
                                              ).map(_apply_wse_downscale, **kwargs1
                                                    ).compute().to_dataframe().reset_index()
     
                        
                    #add some meta
                    res_d[i]['ds_fn'] = os.path.basename(fp)
                        
                    log.debug(f'finished DEM tile {i} w/ {res_d[i].shape}')
                        
                except Exception as e:
                    
                    log.error(f'failed on {os.path.basename(fp)} w/ \n    {e}\n    {traceback.format_exc()}')
                    fail_d[i] = {'fp':os.path.basename(fp), 'error':str(e), 
                                 'traceback':str(traceback.format_exc()),
                                 'now':datetime.now()}
                    fail_cnt+=1
                    if fail_cnt>= max_fail_cnt:
                        log.error(f'max_fail_cnt exceeded... terminating')
                        break
                
            #===================================================================
            # wrap DS
            #=========== ========================================================
            ds_coord_keys = list(ds.coords.keys()) #used in meta
 
 
                
 
    #===========================================================================
    # multi--------- 
    #===========================================================================
    else:
        log.info(f'executing \'_worker_run_downscale\' on {processes} processes w/ {len(fp_l)}')
        # Prepare arguments
        
        args = [(i, fp, raster_index_l, config.log_level, {**_kwargs, **kwargs, **{'basinName2':basinName2}}) for i, fp in enumerate(fp_l)]
        
        # Create a pool of workers and execute
        with multiprocessing.Pool(processes=processes) as pool:
            #doesnt work with tqdm... 
            results = list(tqdm(pool.starmap(_worker_run_downscale, args), total=len(fp_l), desc='fdscM'))
            
            #different argument handling
            #results = list(tqdm(pool.imap_unordered(_worker_run_downscale, args), total=len(fp_l), desc='fdscM'))
        
        # Separate the results into two dictionaries 
        for i, (result, result_fail) in enumerate(results):
            if result is not None:
                res_d[i] = result
            if result_fail is not None:
                fail_d[i] = result_fail
                
        #used in meta
        ds_coord_keys = ['x', 'y', 'day', 'realisation', 'raster_index', 'spatial_ref', 'dem_tile_x', 'dem_tile_y']
        
    log.info(f'finished compute')
    #===========================================================================
    # meta-------
    #===========================================================================
    
    meta_d = {
            'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
            #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
            'outdir_GB':get_directory_size(out_dir),
            #'output_MB':os.path.getsize(ofp)/(1024**2)
            }
 
    #get list of keys
    if len(res_d)>0:
        dx = pd.concat(res_d, axis=0)
        dx = dx.set_index(['basinName2', 'ds_fn'] + [e for e in dx.columns if e in ds_coord_keys]).sort_index(level='raster_index')    #get list of keys
        
        ofp = os.path.join(out_dir, f'meta_{basinName2}_{today_str}')
        dx.to_pickle(ofp+'.pkl')
        dx.to_csv(ofp+'.csv')
        
        log.info(f'wrote results {dx.shape} to \n    {ofp}.csv')
    else:
        log.warning(f'no results!')
        dx=None
    
    #fail
    if len(fail_d)>0:
        
        err_df = pd.DataFrame.from_dict(fail_d).T.reset_index(drop=False)
        ofpe = os.path.join(out_dir, f'fail_{basinName2}_{len(err_df):03d}_{today_str}.csv')
        err_df.to_csv(ofpe)
        log.warning(f'wrote fail data {err_df.shape} to \n    {ofpe}')
        if __debug__:
            raise IOError(f'{basinName2} finished w/ {len(err_df)} errors\n{meta_d}\n%s'%err_df['error'])
        
    else:
        err_df = None
 
    #===========================================================================
    # wrap
    #===========================================================================

            
    
    log.debug(meta_d)
    
    
 
    return dx, err_df
 

def _worker_run_downscale(i, fp,   raster_index_l, log_level, kwargs):
    
    #===========================================================================
    # setup config
    #===========================================================================
    import config
    config.log_level=log_level
    
    #===========================================================================
    # setup logger
    #===========================================================================
    out_dir = kwargs['out_dir']
    
    dem_tile = '_'.join(os.path.basename(fp).split('_')[2:4])
    
    logName = f'%s_{i}_{dem_tile}_{today_str}_p{str(os.getpid())}'%kwargs['basinName2']
    log = get_log_stream(name=logName)
    
    """
    import config
    config.log_level
    """
    
    log_fp = os.path.join(out_dir, f'worker_{logName}'+'.log')
    log = get_new_file_logger(logger=log, fp=log_fp)
    log.debug('\n    '+'\n    '.join([str(h) for h in log.handlers]))
    
    kwargs['log'] = log
    
    log.debug(f'on i={i} loading datset from \n    {fp}')
    #=======================================================================
    # load dataSet
    #=======================================================================

    with xr.open_dataset(fp,
                         engine='netcdf4',
                         chunks={'x':-1, 'y':-1, 'raster_index':1}, #single spatal, 1-per layer
                         
                         ) as ds:
        
        log.debug(f'loaded {ds.dims}' + 
                 f'\n    coors: {list(ds.coords)}' + 
                 f'\n    data_vars: {list(ds.data_vars)}' + 
                 f'\n    crs:{ds.rio.crs}'
                 f'\n    chunks:{ds.chunks}'
                 )
        
        #===================================================================
        # prep
        #=================================================================== 
        #swap in nulls
        ds = ds.where(ds!=-9999, np.nan).round(3)
        
        #load coordinates to memory
        """not sure why this is required"""
        for k,v in ds.coords.items():
            v.load()
  
            
        #===================================================================
        # #clip to raster incidies
        #===================================================================
        if not raster_index_l is None:
            #get just those on the index
            ri_ar = ds.coords['raster_index'].values
            bx = np.isin(ri_ar, raster_index_l)
            if not bx.any():
                log.warning(f'no indicies found in raster_index_l.... skipping')
                return None, None
            elif bx.all():
                pass #everything is selected
            else:
                log.debug(f'selecting {bx.sum()}/{len(bx)} raster_index')              
                ds = ds.sel(raster_index=ri_ar[bx]) 
            
        #report
        ri_l =  ds.coords['raster_index'].values.tolist()
        if isinstance(ri_l, int): ri_l = [ri_l] 
        log.debug(f'on {len(ri_l)} raster_index: {ri_l}')           
        
        #===================================================================
        # compute on raster_index stack---------
        #===================================================================
        try:

            #===================================================================
            # #load persistent layers
            #===================================================================

            #ds.coords.load()
            ds['DEM'].load()            
            ds['WBDY'].load()
            
            #===================================================================
            # call downscaler per raster_index
            #===================================================================
 

            #workaround for single index
            if isinstance(ds.coords['raster_index'].values.tolist(), int): 
                rda =  _apply_wse_downscale(ds, **kwargs)
                
                #reformat to match groupby.map
                s = pd.Series({**{rda.name:rda.item()}, **{k:v.item() for k,v in rda.coords.items()}}, name=i)
                result = s.to_frame().T #.set_index('raster_index', drop=True)
                
            #multi-raster_index
            else:
            
                result = ds.groupby('raster_index', squeeze=False).map(_apply_wse_downscale, **kwargs).compute().to_dataframe().reset_index()

                
            #add some meta
            result['ds_fn'] = os.path.basename(fp)
            result_fail = None
                
            log.debug(f'finished DEM tile {i} w/ {result.shape}')
                
        except Exception as e:
            
            log.error(f'failed on {os.path.basename(fp)} w/ \n    {e}\n    {traceback.format_exc()}')
            result=None
            result_fail = {'fp':os.path.basename(fp), 'error':str(e), 'traceback':str(traceback.format_exc()),
                         'now':datetime.now(), 'log_fp':log_fp}
            
        return result, result_fail
 

def _apply_wse_downscale(ds,                          
                         basinName2=None,                         
                         resample_shape=(100,100),
                         log=None, out_dir=None, ofp=None, use_cache=True, tmp_dir=None,
                         ):
    """DataArray apply to downscale. per-DEM tile, per-event,"""
    

     
    
    #===========================================================================
    # setup
    #===========================================================================
    
    #log = get_slog('apply', log)
    if 'basinName2' in ds.attrs:
        assert ds.attrs['basinName2']==basinName2
 
    raster_index = ds.raster_index.item()
    
    if log is None:log = get_log_stream()        
    log = log.getChild(f'apply_{raster_index:03d}')
    
    
    #extract coordinate key:value pairs
    d=dict(basinName2=basinName2)
    for k, v in ds.coords.items():
        if not k in ['x', 'y']:
            d[k] = v.item()
 
    #===========================================================================
    # setup outputs paths
    #===========================================================================
 
    
    base_name = f'{basinName2}_{raster_index:05d}_%i_%i'%(
        ds.coords['dem_tile_x'].item(), ds.coords['dem_tile_y'].item())
    
    uuid = get_hash(f'{resample_shape}')
    
 
    if tmp_dir is None: 
        tmp_dir =  tempfile.gettempdir() 
    tmp_dir = os.path.join(tmp_dir, base_name, f'{today_str}_{uuid}')
    if not os.path.exists(tmp_dir):os.makedirs(tmp_dir)
 
    
    
    
    if ofp is None:        
        ofp = os.path.join(out_dir, f'{base_name}_wse_fdsc_{uuid}.tif')
        
        
    log.debug(f'on {base_name} w/ \n    {ds.dims}')
    
    """
    #write to test pickle
    test_dir = r'l:\10_IO\2307_roads\test\test_downscale_fdsc\test_apply_wse_downscale'
 
    with open(os.path.join(test_dir, basinName2, f'ds.pkl'), 'wb') as f:
        pickle.dump(ds.compute(), f)
 
        
    """
 
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        log.debug(f'building downscale on {raster_index}')
 
        #load
        ds.load()
        
        #=======================================================================
        # build cost friction
        #=======================================================================
 
        
        """we only want predictions OUTSIDE the channel
        this is equvalent to everywhere the waterbodies are null
        nulls in the cost friction are propagated onto the result"""
        ds['WBDY'] = xr.where(
            np.logical_and(ds['WBDY'].isnull(),ds['DEM'].notnull()), #non-water body + non-null
             1.0, np.nan)
        

        #===================================================================
        # filter DEM violators
        #===================================================================
        """usually consider this as the first step of downscaling"""        
        da_nulls = np.logical_or(ds['WSE']<=ds['DEM'], np.logical_or(ds['WSE'].isnull(), ds['DEM'].isnull()))
 
        ds['WSE'] = xr.where(da_nulls, np.nan, ds['WSE'])
 
        if ds['WSE'].isnull().all():
            """seems strange that ALL wse values are less than the DEM
            must be some mismatch between the DEMs for very dry tiles?
            """
            log.warning(f'initial DEM filter removed all wet cells... skipping this tile.')
            return xr.DataArray(None, coords=d, name='wse_fdsc_fn')
        #=======================================================================
        # dump to GeoTiffs---
        #=======================================================================
        fp_d = dict()
        for k in ['WSE', 'DEM', 'WBDY']:
            fp_d[k] = os.path.join(tmp_dir, f'{k}_{base_name}.tif')
            
            #some checks
            if not k=='WBDY':
                assert not ds[k].isnull().all(), k
                assert not (ds[k]==-9999).all(), k
                
                
            ds[k].fillna(-9999).rio.write_nodata(-9999).rio.to_raster(fp_d[k], dtype='float32', compute=False, compress='LZW')
            assert os.path.exists(fp_d[k]), k
            log.debug(f'wrote {ds[k].shape} to \n    {fp_d[k]}')
            
        log.debug(f'dumped rasters to \n    {tmp_dir}')
        
        #=======================================================================
        # create a resample of WSE for clump points
        #=======================================================================
        """
        CostGrow._03_isolated(method='pixel') requires a coarse raster for the clump detection
            originally, we used the raw WSE for this
            however, retreiving this raw coarse at this stage is pretty onerous
            
            instead, we're using an even coarser shape to reproject to
            """
        wse_rsmp_fp = os.path.join(tmp_dir, f'wse_rsmp_{base_name}.tif')
        wse_rsmp_da = xr.where(ds['WSE'].notnull(), 1,-9999
                           ).rio.reproject(ds.rio.crs, shape=resample_shape, resampling=Resampling.max)
                           
        #check
        assert not (wse_rsmp_da==-9999).all().item(), f'resampled WSE is all null'
        
        
        wse_rsmp_da.rio.write_nodata(-9999).rio.to_raster(wse_rsmp_fp, dtype='float32', compute=False, compress='LZW')
        
        log.debug(f'wrote resampled raster with {wse_rsmp_da.shape} to \n    {wse_rsmp_fp}')
 
        
        #===========================================================================
        # execute costGrow-------
        #===========================================================================
        with Dsc_Session(run_name='r', relative=True, out_dir=out_dir, logger=log, 
                         tmp_dir=os.path.join(tmp_dir, 'dsc'),
                         compress=Compression('LZW'),
                         ) as ses:
            
            try:
 
                ofp, meta_lib = ses.p2_costGrow_dp( fp_d['WSE'], fp_d['DEM'],
                                                   cost_fric_fp=fp_d['WBDY'],
                                                   clump_kwargs=dict(
                                                       method='pixel',
                                                       wse_raw_fp=wse_rsmp_fp,
                                                       ),
                                                   decay_kwargs = dict(
                                                       #wse_raw_fp=fp_d['WSE'],
                                                       loss_frac=0.0005,
                                                       ),
                                                   ofp=ofp)
            except Exception as e:
                raise IOError(f'failed \'{base_name}\' p2_costGrow_dp w/ \n{e}')
            
        log.debug(f'finished cosgGrow to \n    {ofp}')
    else:
        log.debug(f'file exists... skipping')
        #meta_lib=dict()
        
    return xr.DataArray(os.path.basename(ofp), coords=d, name='wse_fdsc_fn')
    #return ofp, meta_lib


if __name__=='__main__':
    
    from haz.hp.basic import init_root_logger
    import config
    config.log_level = logging.INFO
    
    
    import config
    
    root_logger = init_root_logger()
    
    run_downscale_fines(
        'elbe_upper',
        fine_ds_dir = r'l:\\10_IO\\2307_roads\\outs\\rim_2019\\downscale\\02fine\\elbe_upper',
        fp_l = [
            r'l:\10_IO\2307_roads\outs\rim_2019\downscale\02fine\elbe_upper\fine_dgm5_4400_3120_20_049_97c8f6ca3abe3fdc.nc'
            ],
        raster_index_l=[410243],
        #dev_stack_cnt=2, #{'tdelta': '114.72 secs', 'outdir_GB': 0.015899572521448135}
        use_cache=False,
        processes=2, #for splitting fine_ds_fp (not raster_index) 
        log=root_logger
        )
    

            
 
        
        