'''
Created on Jan. 21, 2024

@author: cef

collect input DataSet and add WSH
'''


#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, psutil, pickle, tempfile, traceback, math, sys

import subprocess
import multiprocessing
from datetime import datetime

import numpy as np
import numpy.ma as ma
import pandas as pd

from tqdm import tqdm

import fiona 
import geopandas as gpd
import shapely.geometry as sgeo
from pyproj import CRS

from osgeo import gdal
gdal.UseExceptions()

import rasterio as rio
from rasterio.enums import Resampling, Compression
from rasterio.warp import calculate_default_transform, reproject, Resampling

import xarray as xr
import rioxarray

import config

from haz.hp.basic import today_str, view, get_log_stream, get_directory_size, get_new_file_logger, get_tqdm_disable

from haz.rim2019.parameters import  out_base_dir

#===============================================================================
# helpers------
#===============================================================================
 

def get_slog(name, log):
    if log is None:
        log = get_log_stream()
        
    return log.getChild(name)
        

get_hash = lambda x:hashlib.shake_256(x.encode("utf-8")).hexdigest(8)

def get_rio_meta(fp, attn_l = ['crs', 'shape', 'transform', 'bounds', 'res']):
    with rio.open(fp, 'r') as ds:
        return {k:getattr(ds, k) for k in attn_l}


def get_coord_aslist(coord):
    ri_l =  coord.values.tolist()
    if isinstance(ri_l, int): 
        ri_l = [ri_l] 
    
    return ri_l


#===============================================================================
# funcs-----
#===============================================================================






def run_toWSH(
        fdsc_dx_fp=None,
        fdsc_dx=None,
        data_dir=None,
        
        fine_ds_dir=None,
        processes=None, 
        
        encoding =  {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},
        
        
        #debugging
        dem_tile_l=None,
        
        log=None, out_dir=None,  tmp_dir=None,use_cache=True, 
        **kwargs):
    """post-process downscaled rasters by reprojecting, re-organizing files, and computing meta/stats
    
    NOTE: set up for a singel basin, but this could be combined to run on all basins
    
    Parms
    ---------
    fdsc_dx_fp: str
        filepath to meta_dx pickle output from run_downscale_fines.
        index of data_dir 
    
    data_dir: str
        directory with downscalied WSE raster reults (indexed in fdsc_dx)
        
            
    fine_ds_dir: str
        directory to fine (input) DataSets... for re-joining
     
    
        
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    log = get_slog('WSH', log)
    

    
 
    #===========================================================================
    # load the index
    #===========================================================================
    if fdsc_dx is None:
        fdsc_dx = pd.read_pickle(fdsc_dx_fp)
        log.info(f'loaded {fdsc_dx.shape} from \n    {fdsc_dx_fp}')
        
    log.info(f'on {fdsc_dx.shape}')
    """
    view(fdsc_dx)
    """
    bx = np.array([e=='basinName2' for e in fdsc_dx.index.names])
    if bx.sum()>1:
        log.warning(f'got double basinName2 indexer./. dropping one')
        fdsc_dx = fdsc_dx.droplevel(3) #hopefully its always in this position
    
    #extract vars
    basin_index = fdsc_dx.index.get_level_values('basinName2').unique()
    assert len(basin_index)==1
    basinName2 = basin_index.item()
    
    if data_dir is None:
        data_dir = os.path.dirname(fdsc_dx_fp)
    
    if fine_ds_dir is None:
        fine_ds_dir = os.path.join(out_base_dir, 'downscale', '02fine', basinName2)
    
    assert os.path.exists(fine_ds_dir)
    #===========================================================================
    # setup directories
    #===========================================================================
    if out_dir is None: out_dir = os.path.join(out_base_dir, 'downscale', '04wsh', basinName2) #appending basinName below
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if tmp_dir is None: 
        from haz.rim2019.parameters import tmp_dir
        tmp_dir = os.path.join(tmp_dir, 'downscale', '03fdsc', basinName2)
        
    if not os.path.exists(tmp_dir):os.makedirs(tmp_dir)
    
    
    
    skwargs = {**dict(out_dir=out_dir, use_cache=use_cache, tmp_dir=tmp_dir), **kwargs}
        
 
 
    #===========================================================================
    # prep meta from downscaler-----
    #=========================================================================== 
    """
    fdsc_dx.xs(4409, level='raster_index')['wse_fdsc_fn'].values
    view(fdsc_dx.xs(4409, level='raster_index')['wse_fdsc_fn'])
    """
    assert os.path.exists(data_dir), data_dir
    
    #fix nulls
    fdsc_dx['wse_fdsc_fn']=fdsc_dx['wse_fdsc_fn'].replace('None', np.nan)    
    fdsc_dx.loc[pd.isnull(fdsc_dx['wse_fdsc_fn']), 'wse_fdsc_fn']=np.nan
    
    """
    fdsc_dx['wse_fdsc_fn'].sort_values(ascending=True).values
    """

    
    """happens during initial DEM filter of fdsc we may drop an event
    seems better to handle this during write_wsh_ds_stack to maintain the stack consistency
    """
    
    bx = fdsc_dx['wse_fdsc_fn'].isnull()
    if bx.any():
        log.warning(f'got {bx.sum()}/{len(bx)} empty downscales')
        log.debug(fdsc_dx[bx].groupby('raster_index')['basinName'].count())
 
 
 
    
    #add dem tile
    idf = fdsc_dx.index.to_frame().reset_index(drop=True)
    fdsc_dx.loc[:, 'dem_tile'] = idf['dem_tile_x'].astype(str).str.cat(idf['dem_tile_y'].astype(str), '_').values
    fdsc_dx = fdsc_dx.set_index('dem_tile', append=True)
    
    #get filepaths
    bx = np.logical_and(fdsc_dx['wse_fdsc_fn'].duplicated(), fdsc_dx['wse_fdsc_fn'].notna())
    """
    view(fdsc_dx['wse_fdsc_fn'].sort_values())
    """
    if bx.any():
        log.warning(f'got {bx.sum()}/{len(bx)} duplicates... dropping')
        fdsc_dx = fdsc_dx.loc[~bx, :]
        log.debug(fdsc_dx[bx])
        
    fdsc_dx['fp_raw'] = [np.nan if pd.isnull(e) else os.path.join(data_dir, e) for e in fdsc_dx['wse_fdsc_fn'].values]
    

    #load file  meta
    log.info(f'loading meta from {len(fdsc_dx)}  wse_fdsc_fn')
    d = dict()
    for i, fp in tqdm(fdsc_dx['fp_raw'].items(), total=len(fdsc_dx), desc='file meta'):
        if pd.isnull(fp):
            d[i]={'ctime':np.nan, 'size_MB':np.nan}
            continue
        
        assert os.path.exists(fp), f'bad filepath on \n    {i}'
        d[i] = {'ctime':os.path.getctime(fp), 'size_MB':os.path.getsize(fp)/(1024**2)}
        
        
    #add trhe meta
    df = pd.DataFrame.from_dict(d).T
    df.index.set_names(fdsc_dx.index.names, inplace=True)        
    fdsc_dx = fdsc_dx.join(df)
    
    #check for duplicates
    fdsc_dx = fdsc_dx.sort_values('ctime', ascending=False) #newest at teh top
    bx = fdsc_dx.index.duplicated()
    if bx.any():
        log.warning(f'got {bx.sum()}/{len(bx)} duplicated entries... taking the newest')
        fdsc_dx = fdsc_dx.loc[bx, :]
        
    #debug clipping
    if __debug__:
        if not dem_tile_l is None:
            bx = fdsc_dx.index.get_level_values('dem_tile').isin(dem_tile_l)
            assert bx.any()
            log.warning(f'clipping {len(fdsc_dx)} to {bx.sum()} from {len(dem_tile_l)} \'dem_tile\' values')
            fdsc_dx = fdsc_dx.loc[bx, :]
 
    #wrap
    fdsc_dx = fdsc_dx.sort_index()
    

    """
    view(fdsc_dx)
    """ 
    #===========================================================================
    # loop and join (SINGLE)----
    #===========================================================================
    log.info(f'looping on {len(fdsc_dx)} .tif files w/ processes={processes} from \n    {data_dir}')
    index_l = [k for k in fdsc_dx.index.names if not k in ['raster_index', 'day', 'realisation']]
    
    if processes is None: processes=1
    
    if processes <=1:
        meta_lib = dict()
        for i, (i0, gdx) in tqdm(enumerate(fdsc_dx.groupby('ds_fn')), desc='org', disable=get_tqdm_disable()):            
 
            meta_lib[i] = write_wsh_ds_stack(basinName2, fdsc_dx, fine_ds_dir, 
                                             encoding, out_dir, i, index_l, i0, gdx,
                                             use_cache, log=log.getChild(str(i)))

            
            
    #===========================================================================
    # MULTI----------
    #===========================================================================
    else:
        
        args = [(basinName2, fdsc_dx, fine_ds_dir, encoding, out_dir, i, index_l, i0, gdx, use_cache, config.log_level) 
                for i, (i0, gdx) in enumerate(fdsc_dx.groupby('ds_fn'))]
        
        # Use starmap
        log.debug(f'multiprocessing.Pool.starmap on \'write_wsh_ds_stack\' w/ {len(args)}')
        with multiprocessing.Pool(processes=processes) as p:
            meta_lib = dict(enumerate(p.starmap(_worker_write_wsh_ds_stack, args)))
        
            
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(meta_lib)}')
    ofp_d=dict()
    sfx = f'fdsc_stack_{basinName2}_{today_str}'
    
 
    
    meta_dx = pd.DataFrame.from_dict(meta_lib).T.drop(['ofp'], axis=1).set_index(index_l, append=True)
    meta_dx.index = meta_dx.index.droplevel(0)
    
    crs = CRS.from_wkt(meta_dx.iloc[0,:]['crs'])
    #===========================================================================
    # spatial
    #===========================================================================
    #add geometry
    meta_dx['geometry'] = meta_dx['bounds'].apply(lambda x: sgeo.Polygon([(x[0], x[1]), (x[0], x[3]), (x[2], x[3]), (x[2], x[1])]))

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(meta_dx.reset_index().drop(['bounds', 'crs'], axis=1), geometry='geometry', crs=crs)
 
 
    #type conversion
    cn = [ 'dem_tile_x', 'dem_tile_y']
    gdf = gdf.astype({k:float for k in cn}).astype({'tdelta':str})
 
    
    """
    gdf.dtypes
    gdf.columns
    view(gdf.dtypes)
    """
    ofp_d['meta_gdf'] = os.path.join(out_dir, f'meta_{len(gdf)}_{sfx}.gpkg')
    
    
    gdf.drop(['dem_tile'], axis=1).to_file(ofp_d['meta_gdf'])
    log.info(f'wrote {gdf.shape} spatial meta to\n    %s'%ofp_d['meta_gdf'])
    
    #===========================================================================
    # aspatial
    #===========================================================================
    ofp_d['meta_pkl'] = os.path.join(out_dir, f'meta_{len(meta_dx)}_{sfx}.pkl')
    meta_dx.to_pickle(ofp_d['meta_pkl'])
    
    #to csv
    ofp_d['meta_csv'] = os.path.join(out_dir, f'meta_{len(meta_dx)}_{sfx}.csv')
    meta_dx.drop(['geometry', 'crs'], axis=1).to_csv(ofp_d['meta_csv'])
    
    log.info(f'wrote {meta_dx.shape} aspatial meta to\n    %s'%ofp_d['meta_csv'])
 
    #===========================================================================
    # wrap-------
    #===========================================================================
 
    
    meta_d = {
            'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
            #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
            'outdir_GB':get_directory_size(out_dir),
            #'output_MB':os.path.getsize(ofp)/(1024**2)
            }
    
 
        
    
    log.info(meta_d)
    
    return meta_dx, None

def _worker_write_wsh_ds_stack(basinName2, fdsc_dx, fine_ds_dir, encoding,  out_dir, i, index_l, i0, gdx,use_cache,
                               log_level):
    """multlprocess wrapper
    
    
    TODO: fix error handling to return err_d and meta_d so we can report using the main logger
    """
    #===========================================================================
    # setup config
    #===========================================================================
    import config
    config.log_level=log_level
    
    #===========================================================================
    # get meta
    #===========================================================================
    mdf = gdx.index.to_frame().reset_index(drop=True)
    base_name = f'{basinName2}_%s_%i' % (mdf['dem_tile'][0], len(mdf))
    
    #===========================================================================
    # build logger
    #===========================================================================
    logName = f'{base_name}_{i}_{today_str}_p{str(os.getpid())}'
    log=get_log_stream(name=logName)
    log_fp = os.path.join(out_dir, f'worker_{logName}'+'.log')
    log = get_new_file_logger(logger=log, fp=log_fp)
    
    log.debug('\n    '+'\n    '.join([str(h) for h in log.handlers]))
 
    
    log.debug('calling write_wsh_ds_stack')
    
    try:
        meta_d = write_wsh_ds_stack(basinName2, fdsc_dx, fine_ds_dir, encoding, 
                                    out_dir, i, index_l, i0, gdx,use_cache, log=log)
    except Exception as e:
        msg = f'write_wsh_ds_stack failed on {base_name} w/ \n    log_fp:{log_fp}\n    {e}\n    {traceback.format_exc()}'
        log.error(msg)
        raise IOError(msg)
    
    log.debug(f'wrapping worker')
    return meta_d
    

def write_wsh_ds_stack(basinName2, fdsc_dx, fine_ds_dir, encoding,  out_dir, i, index_l, i0, gdx,use_cache,
                       log=None,
                       ):
    #=======================================================================
    # setup
    #=======================================================================
    start_i = datetime.now()
    
 
    
    assert len(np.unique(gdx.index.get_level_values('raster_index'))) == len(gdx), 'non-unique raster_index'
    mdf = gdx.index.to_frame().reset_index(drop=True)
    base_name = f'{basinName2}_%s_%i' % (mdf['dem_tile'][0], len(mdf))
    
    if log is None: 
        log=get_log_stream(name=base_name)
 
        
    log.debug(f' {i+1} on  {len(gdx)} w/ {i0}')
    #===================================================================
    # output
    #===================================================================
    
    uuid = get_hash(f'{gdx}')
    ofp = os.path.join(out_dir, f'fdsc_stack_{base_name}_{uuid}.nc')
    #===================================================================
    # meta
    #===================================================================
    meta_d = {'ds_fn':i0, 
              #'raster_index_len':len(gdx), no... do this after we filter 
        'ofp':ofp, 'ofn':os.path.basename(ofp)}
        #'basinName2':basinName2, 'dem_tile':mdf['dem_tile'][0]
    meta_d.update({k:mdf[k][0] for k in index_l})
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        log.debug(f'loading dataArray for each fdsc geoTiff')
        #===================================================================
        # load each downscale for this tile
        #===================================================================
        fdsc_d = dict()
        empty_d=dict()
        for i1, row in gdx.iterrows():
            keys_d = dict(zip(fdsc_dx.index.names, i1))
            fp = row['fp_raw']
            
            if pd.isnull(fp):
                empty_d[keys_d['raster_index']] =i1
                log.warning(f'empty layer {i1}')
                continue
            
            
            da_i = rioxarray.open_rasterio(row['fp_raw'], parse_coordinates=True, #chunks =False, #load lazily w/ auto chunks
                cache=True, masked=True).squeeze()
                
            #add raster index
            #coords_d = {k:v for k,v in keys_d.items() if k in ['raster_index', 'basinName2']
            fdsc_d[keys_d['raster_index']] = da_i.assign_coords(keys_d)
        
        log.debug(f'loaded {len(fdsc_d)} fdsc rasters')
        #merge
        da = xr.concat(fdsc_d.values(), dim='raster_index').rio.write_crs(da_i.rio.crs)
        da.attrs = {}
        
        meta_d['raster_index_len'] = len(fdsc_d)
        """
        view(gdx)
        """
        #===================================================================
        # merge with fines
        #===================================================================
        fine_fp = os.path.join(fine_ds_dir, keys_d['ds_fn'])
        assert os.path.exists(fine_fp), i
        log.debug(f'open_dataset on {fine_fp}')
        
        with xr.open_dataset(fine_fp, engine='netcdf4') as ds:
            #load coordinates to memory
            """not sure why this is required"""
            for k, v in ds.coords.items():
                v.load()
            
            #promote the raster index if necessary
            if len(ds['WSE'].shape) == 2:
                ds = ds.expand_dims(dim='raster_index')
                
            #drop empties
            """occasionally fdsc's DEM filter removes all wet cells and nothing is returned
            for this, we need to also drop the WSE (resampled) event"""
            if len(empty_d)>0:
                #identify coordinates to drop
                bx_ar = ds.raster_index.isin(list(empty_d.keys())).values 
                assert bx_ar.sum()==len(empty_d)
                
                #drop these coordinate values/layers from the DataSet
                log.warning(f'dropping {bx_ar.sum()}/{len(bx_ar)} raster_index values from teh fine dataset')                
                ds = ds.sel(raster_index=~bx_ar)
 
                
            #check for conformance
            assert ds.rio.bounds() == da.rio.bounds()
            assert ds.rio.crs == da.rio.crs
            
            miss_s=set(ds.coords['raster_index'].values).symmetric_difference(da.coords['raster_index'].values)
            if not miss_s == set():
                """
                view(ds.coords['raster_index'].to_dataframe())
                """
                raise IndexError(f'raster_index mis-match between WSE and WSEf {miss_s} for {base_name}')
            
            if not ds['WSE'].shape == da.shape:                
                raise TypeError(f'shape mismatch')
            

                
            ds['WSEf'] = da
            #===============================================================
            # compute WSH
            #===============================================================
            log.debug(f'building WSHf')
            ds['WSHf'] = (ds['WSEf'] - ds['DEM']).fillna(0)
            #===============================================================
            # write
            #===============================================================
            for k in ['WSEf', 'WSHf']:
                ds[k].encoding.update(encoding)
            
            log.debug(f'writing to \n    {ofp}')
            ds.to_netcdf(ofp, format='netCDF4', engine='netcdf4', mode='w')
            #=======================================================================
            # write GeoTiff
            #=======================================================================
            #===============================================================
            # """doesnt make much sense with all the raster_indexes"""
            # if write_tiff:
            #     log.debug(f'writing to GeoTiff')
            #     ofp1 = os.path.join(os.path.dirname(ofp), 'tifs', os.path.basename(ofp).replace('.nc', '.tif'))
            #     if not os.path.exists(os.path.dirname(ofp1)):
            #         os.makedirs(os.path.dirname(ofp1))
            #
            #     #convert dataset to array (creates 'variable' as a new coord)
            #     da = ds.to_array().fillna(-9999).rio.write_nodata(-9999)
            #
            #     #write this 3D DataArray as a multi-band raster
            #     da.rio.to_raster(ofp1, dtype='float32', compute=False, compress='LZW')
            #
            #     log.info(f'wrote {da.shape} GeoTiff to \n    {ofp1}')
            #===============================================================
            #===============================================================
            # meta
            #===============================================================
            #meta_d['raster_index_len'] = len(get_coord_aslist(ds.coords['raster_index']))
        #===================================================================
        # wrap the fine dtatstack loop
        #===================================================================
        meta_d.update({'tdelta':datetime.now() - start_i, 'bounds':ds.rio.bounds(), 'crs':str(ds.rio.crs)})
        
    #===========================================================================
    # cache
    #===========================================================================
    else:
        with xr.open_dataset(ofp, engine='netcdf4') as ds:
            meta_d.update({'bounds':ds.rio.bounds(), 'crs':str(ds.rio.crs), 'tdelta':datetime.now() - start_i,
                          'raster_index_len':len(ds['raster_index'])},
            )
        
    return meta_d


        
        

if __name__=='__main__':
    run_toWSH(
        r'l:\10_IO\2307_roads\outs\rim_2019\downscale\03fdsc\ems\meta_ems_20240126.pkl',
        #r'l:\10_IO\2307_roads\outs\rim_2019\downscale\03fdsc\elbe_lower\meta_elbe_lower_20240127.pkl',
 
        use_cache=False,
        processes=None,
        
        #dem_tile_l=['4420_3160'],
 
        )
    
    
    
    print('done')
    
    
    
    
    
    