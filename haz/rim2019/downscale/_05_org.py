'''
Created on Jan. 21, 2024

@author: cef

re-organize downscale results
    re-project to original rim2019 CRS
    structure like
        basinName2
            raster_index
                dem_tile
                
    build an index with some meta
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


from haz.hp.basic import today_str, view, get_log_stream, get_directory_size, get_new_file_logger

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



#===============================================================================
# funcs-----
#===============================================================================




def run_organize(
        wsh_dx_fp=None,
        wsh_dx_raw=None,
        data_dir=None,
        dstSRS=None,
 
        processes=None,
 
        
        creationOptions=['COMPRESS=LERC_DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6','MAX_Z_ERROR=.001',
                         'TILED=YES', 'BLOCKXSIZE=400', 'BLOCKYSIZE=400',
                         'SPARSE_OK=FALSE'],
        
        log=None, out_dir=None,  tmp_dir=None,use_cache=True, 
        **kwargs):
    """post-process downscaled rasters by reprojecting, re-organizing files, and computing meta/stats
    
    NOTE: set up for a singel basin, but this could be combined to run on all basins
    
    Parms
    ---------
    wsh_dx_fp: str
        filepath to meta_dx pickle output from run_downscale_fines
        
    wsh_dx: pd.DataFrame
        metadata from run_toWSH for this basin
            columns: Index(['raster_index_len', 'ofn', 'tdelta', 'bounds', 'crs', 'geometry'], dtype='object')
            index: FrozenList(['basinName2', 'ds_fn', 'spatial_ref', 'dem_tile_x', 'dem_tile_y', 'dem_tile'])
    
        
 
    
        
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    log = get_slog('org', log)
    
    if dstSRS is None:
        from haz.rim2019.parameters import epsg_id
        dstSRS = CRS.from_epsg(epsg_id)
    #===========================================================================
    # load the index
    #===========================================================================
    if wsh_dx_raw is None:
        wsh_dx_raw = pd.read_pickle(wsh_dx_fp)
        log.info(f'loaded {wsh_dx_raw.shape} from \n    {wsh_dx_fp}')
        """
        wsh_dx.columns
        wsh_dx.index.names
        """
        
    #clean up
    wsh_dx = wsh_dx_raw.drop(['geometry', 'crs', 'tdelta', 'bounds'], errors='ignore', axis=1
                         ).rename(columns={'ofn':'fn'}).droplevel(['ds_fn', 'spatial_ref'])
    
    #extract vars
    basin_index = wsh_dx.index.get_level_values('basinName2').unique()
    assert len(basin_index)==1
    basinName2 = basin_index.item()
    
    log = log.getChild(basinName2)
    
    if data_dir is None:
        data_dir = os.path.dirname(wsh_dx_fp)
    
    
    #===========================================================================
    # setup directories
    #===========================================================================
    if out_dir is None: out_dir = os.path.join(out_base_dir, 'downscale', '05org', basinName2) #appending basinName below
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if tmp_dir is None: 
        from haz.rim2019.parameters import tmp_dir
        tmp_dir = os.path.join(tmp_dir, 'downscale', '05org', basinName2)
        
    if not os.path.exists(tmp_dir):os.makedirs(tmp_dir)
    
    
    
    #skwargs = {**dict(out_dir=out_dir, use_cache=use_cache, tmp_dir=tmp_dir), **kwargs}
        
 
    assert os.path.exists(data_dir), data_dir
    #===========================================================================
    # prep meta-----
    #===========================================================================
 
 
    """
    view(wsh_dx)
    """
    
 
    #get filepaths
    #===========================================================================
    # bx = wsh_dx['fn'].duplicated()
    # if bx.any():
    #     raise IOError(f'got duplicate ds tiles')
    #     log.warning(f'got {bx.sum()}/{len(bx)} duplicates... dropping')
    #     wsh_dx = wsh_dx.loc[~bx, :]
    #===========================================================================
        
    wsh_dx['fp'] = [os.path.join(data_dir, e) for e in wsh_dx['fn'].values]
    
 
    
    #load file  meta
    log.info(f'loading meta from {len(wsh_dx)}  ')
    d = dict()
    for i, fp in tqdm(wsh_dx['fp'].items(), total=len(wsh_dx), desc='file meta', disable=True):
        assert os.path.exists(fp), f'bad filepath on \n    {i}'
        d[i] = {'ctime':os.path.getctime(fp), 'size_MB':os.path.getsize(fp)/(1024**2)}
        
        
    #add trhe meta
    df = pd.DataFrame.from_dict(d).T
    df.index.set_names(wsh_dx.index.names, inplace=True)        
    wsh_dx = wsh_dx.join(df)
    
    
    wsh_dx = wsh_dx.sort_values('ctime', ascending=False) #newest at teh top
    bx = wsh_dx.index.duplicated()
    if bx.any():
        log.warning(f'got {bx.sum()}/{len(bx)} duplicated entries... taking the newest')
        wsh_dx = wsh_dx.loc[bx, :]
    
    
    assert (wsh_dx['size_MB']>1.0).all(), 'got some very small dataset files'
    #dev break
    #=======================================================================
    # if __debug__ and (not dev_stack_cnt is None):
    #     cnt = len(index_dx)
    #     index_dx = index_dx.iloc[0:dev_stack_cnt, :]
    #     log.warning(f'dev_stack_cnt clipped from {cnt} to {len(fp_l_raw)}')
    #=======================================================================
    
    log.info(f'loading {len(wsh_dx)} ds .nc files from \n    {data_dir}')
    
    #===========================================================================
    # loop and reproject (SINGLE)----
    #===========================================================================
    skwargs = {**dict(out_dir=out_dir, use_cache=use_cache, tmp_dir=tmp_dir, 
                      creationOptions=creationOptions, dstSRS=dstSRS), **kwargs}
    
    if processes ==1 or (processes is None):
        meta_lib = dict()
        for i, (i_t, s) in tqdm(enumerate(wsh_dx.iterrows()), total=len(wsh_dx), desc='org'):
            """
            wsh_dx.index.names
            """
            #=======================================================================
            # setup
            #=======================================================================
            meta_lib[i] = _warp_tile_ds_write_tifs(s, log=log.getChild(str(i)), **skwargs)
    #===========================================================================
    # MULTI
    #===========================================================================
    else:
        raise KeyError('not implemented')
        
        #=======================================================================
        # # Prepare the arguments for starmap
        # args = [(i_t, s,  wsh_dx, dstSRS, creationOptions,  use_cache, dict(out_dir=out_dir)) for i_t, s in wsh_dx.iterrows()]
        # 
        # # Use a multiprocessing pool
        # with multiprocessing.Pool(processes=processes) as pool:
        #     results = pool.starmap(_worker_warp, args)
        # 
        # # Convert the results to a dictionary
        # meta_lib = {i_t: result for i_t, result in results}
        #=======================================================================
        
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(meta_lib)}')
    meta_dx = pd.concat(meta_lib).droplevel(0).swaplevel(0,1).sort_index()
    
    #===========================================================================
    # build VRTs
    #===========================================================================
    """need to do this here as it joins across dem_tiles"""
    ri_s = meta_dx.groupby('raster_index').count()['raster_fp']
    log.info(f'building VRTs on {len(ri_s)} events')
    vrt_d =dict()
    for ri, gdx in meta_dx.groupby('raster_index'):
        log.debug(ri)
 
        fp_l = gdx['raster_fp'].tolist()
        #'%s_%s_%05d'%(keys_d['basinName2'], keys_d['dem_tile'], keys_d['raster_index'])
        #f'rim2019_dgm5_wd_{k}_005m.tif'
        vrt_d[ri] = create_vrt(fp_l, os.path.join(out_dir, f'rim2019_dgm5_wd_{basinName2}_{ri:05d}_005m.vrt'))

    log.debug(f'wrote {len(vrt_d)} vrts til file')
    #===========================================================================
    # build meta dx----
    #===========================================================================
    ofp_d=dict()
    
    
    #===========================================================================
    # spatial
    #===========================================================================
    gdf = meta_dx.reset_index()
    
    gdf['geometry'] = gdf['bounds'].apply(lambda x: sgeo.box(*x))
    gdf['resolution'] = gdf['res'].apply(lambda x:x[0]).round(6)
    gdf['shape_str'] = gdf['shape'].apply(lambda x:str(x)).astype(str)
    #meta_dx['cdate'] = meta_dx['ctime'].apply(lambda x:datetime.fromtimestamp(x).strftime('%Y-%m-%d-%H%M%S'))
    
    gdf = gdf.drop(['crs', 'res', 'shape', 'bounds'], axis=1)
    
 
    #type conversion
    cn = ['dem_tile_x', 'dem_tile_y', 'raster_index', 'mask_cnt', 'pixels', 'max', 'min', 'mean', 'wet_cnt', 'wet_frac']
    gdf = gdf.astype({k:float for k in cn})
    
    """
    view(gdf)
    view(gdf.dtypes)
    view(meta_dx.dtypes)
    """
    gdf = gpd.GeoDataFrame(gdf, crs=dstSRS, geometry='geometry')
    gdf = gdf.drop('geometry', axis=1).set_geometry(gdf['geometry'])        
    
    ofp_d['gdf'] = os.path.join(out_dir, f'meta_05org_rasters_{basinName2}_{len(gdf):04d}_{today_str}.gpkg')
    
    gdf.to_file(ofp_d['gdf'], index=False, engine='fiona', driver='GPKG')
    
    log.info(f'wrote {gdf.shape} to \n    %s'%ofp_d['gdf'])
    
    #===========================================================================
    # aspatial
    #===========================================================================
    ofp_d['meta_dx'] = os.path.join(out_dir, f'meta_05org_{basinName2}_{len(meta_dx):04d}_{today_str}.csv')
    meta_dx.to_csv(ofp_d['meta_dx'])
    log.info(f'wrote {meta_dx.shape} to \n    %s'%ofp_d['meta_dx'])
 
    #===========================================================================
    # meta-------
    #===========================================================================
 
    
    meta_d = {
            'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
            #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
            'outdir_GB':get_directory_size(out_dir),
            #'output_MB':os.path.getsize(ofp)/(1024**2)
            }
    
 
        
    
    log.info(meta_d)
    
    return meta_dx, None


def xxx_worker_warp(i_t, s,  wsh_dx, dstSRS, creationOptions,  use_cache, kwargs):
    
    keys_d = dict(zip(wsh_dx.index.names, i_t))
    
    out_dir = kwargs['out_dir']
    logName = f'%s_%i_{today_str}_p{str(os.getpid())}'%(keys_d['basinName2'], keys_d['raster_index'])
    
    """seems to go to debug even when __debug__=False"""
    log = get_log_stream(name=logName)
    
    log_fp = os.path.join(out_dir, f'worker_{logName}'+'.log')
    kwargs['log'] = get_new_file_logger(logger=log, fp=log_fp)
    
 
    return i_t, _warp_tile_ds_write_tifs(i_t, s,  wsh_dx, dstSRS, creationOptions,  use_cache, **kwargs)
    
    
def _warp_tile_ds_write_tifs( s,  
                    out_dir=None, use_cache=True, tmp_dir=None, 
                    log=None,
                    dkey='WSHf', 
                    dstSRS=None,
                    creationOptions=[],
                    ):
    
    #===========================================================================
    # setup
    #===========================================================================
    start_i = datetime.now()
    
    if log is None: log=get_log_stream()
    
    keys_d = dict(zip(['basinName2', 'dem_tile_x', 'dem_tile_y', 'dem_tile'], s.name))
    
    k = keys_d['dem_tile']
    
    log = log.getChild(str(k))
    
    #log.debug(s)
    fp = s['fp']
    
    #===========================================================================
    # meta
    #===========================================================================
    #meta_d = {**s.to_dict()}
    
    
    #===========================================================================
    # open dataset
    #===========================================================================
    with xr.open_dataset(fp, engine='netcdf4') as ds:
        log.debug(f'loaded DataSet {ds.dims} from \n    {fp}')
        
        #check
        assert ds.attrs['basinName2']==keys_d['basinName2']
        assert len(ds.coords['raster_index'])==s['raster_index_len']
        
        #retrieve dataArray of itnerest
        da = ds[dkey]
        #reproject
        log.debug(f'reprojecting {da.shape} to {dstSRS.to_string()}')
        da_rpj = da.rio.reproject(dstSRS,
                                  resolution=None, #match to outshape?
                                  shape=None,
                                  resampling=Resampling.nearest,
                                  ).compute()
        
        log.debug(f'reprojected to {da_rpj.rio.resolution()}, {da_rpj.shape}')
        
        #dump to GeoTiffs
        res_dx = da_rpj.groupby('raster_index', squeeze=False).map(
            _apply_toRaster, out_dir=out_dir, use_cache=use_cache, log=log, creationOptions=creationOptions,
            ).compute().to_dataframe()
        
 
 
    log.debug(f'finished writing {len(res_dx)} GeoTiffs')
 
    #=======================================================================
    # get stats
    #=======================================================================
    meta_lib=dict()
    for raster_index, row in tqdm(res_dx.iterrows(), desc=f'raster meta {k}', total=len(res_dx), disable=True):
        
        fp = row['raster_fp']
        
        #basic meta
        meta_d=row.to_dict()
        meta_d['size_MB'] =os.path.getsize(fp)/(1024**2)
        
        
        with rio.open(fp, 'r') as ds:
            
            #attribute meta
 
            meta_d.update(
                #{k:getattr(ds, k) for k in ['crs', 'shape', 'transform', 'bounds', 'res']},
                {'crs':ds.crs.to_epsg(),
                 'res':ds.res,
                 'bounds':tuple(ds.bounds)}
                )
            
            #value meta
            mar = ds.read(1, masked=True)
            assert not mar.mask.all()
            meta_d.update(
                {'mask_cnt':mar.mask.sum(), 
                    'pixels':mar.size, 
                    'shape':mar.shape, 
                    'max':mar.max(), 
                    'min':mar.min(), 
                    'mean':mar.mean(),
                    'wet_cnt':np.sum(mar>0),
                    })
            
            meta_d['wet_frac'] = meta_d['wet_cnt']/meta_d['pixels']
            
        meta_lib[raster_index]=meta_d
        
    #awssemble
    meta_dx = pd.DataFrame.from_dict(meta_lib).T #.set_index(['basinName2', 'dem_tile'])
    meta_dx.index.name='raster_index'
    
    assert len(meta_dx)==s['raster_index_len']
    
    #add indexers
    for k,v in keys_d.items():
        meta_dx[k]=v
    
    meta_dx = meta_dx.set_index(list(keys_d.keys()), append=True).swaplevel(0,-1)
            
    """
    view(res_dx)
    view(meta_dx)
    """
    #=======================================================================
    # wrap loop
    #=======================================================================
    meta_d=dict()
    meta_d.update({
                'tdelta_secs':round((datetime.now() - start_i).total_seconds(), 2), 
                #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                'outdir_GB':get_directory_size(out_dir),
                #'in_MB':os.path.getsize(fp)/(1024**2),
                #'out_MB':os.path.getsize(ofp_i) / (1024 ** 2)})
                })
    
 
    log.debug(meta_d)
    return meta_dx  


def _apply_toRaster(da, out_dir=None, use_cache=True, log=None,
                    creationOptions=[],
                    kn_l =['basinName2', 'dem_tile', 'raster_index'],
                     ):
 
    #===========================================================================
    # setup
    #===========================================================================
    if log is None: log=get_log_stream()
    keys_d = {k:v.item() for k,v in da.coords.items() if k in kn_l}
    
    k = '%s_%s_%05d'%(keys_d['basinName2'], keys_d['dem_tile'], keys_d['raster_index'])
 
    out_dir=os.path.join(out_dir, '%05d'%keys_d['raster_index'])
    
    if not os.path.exists(out_dir):os.makedirs(out_dir) 
    
    log = log.getChild('%05d'%keys_d['raster_index'])
    
    #fix creationOPtions
    if isinstance(creationOptions, list):
        create_d = {item.split('=')[0]: item.split('=')[1] for item in creationOptions}
 
        
    #===========================================================================
    # set filepath
    #===========================================================================
 
    
    ofp = os.path.join(out_dir, f'rim2019_dgm5_wd_{k}_005m.tif')
    
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
        
        da.squeeze().fillna(-9999).reset_coords('raster_index', drop=True).rio.write_nodata(-9999
                  ).rio.to_raster(ofp, dtype='float32', compute=False, 
                                  #compress='LZW',
                                  **create_d)
                  
        log.debug(f'wrote {da.shape} to \n    {ofp}')
              
 
    else:
        log.debug('file exists... skipping')
        
        
    #===========================================================================
    # post-check
    #===========================================================================
    assert os.path.getsize(ofp)>1e3, k
    
    return xr.DataArray(ofp, coords=keys_d, name='raster_fp')


 

def create_vrt(fp_l, output_vrt):
    """ create a vrt file
    
    should use python bindings instead:
        vrt_options = gdal.BuildVRTOptions(separate=False, strict=False)
    vrt = gdal.BuildVRT(ofp, list(d.values()), options=vrt_options)
    vrt.FlushCache()
    
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        # Write the list of file paths to the temporary file
        temp.write('\n'.join(fp_l).encode('utf-8'))
        temp_file_name = temp.name

    # Construct the gdalbuildvrt command
    command = f"gdalbuildvrt -input_file_list {temp_file_name} {output_vrt}"

    # Execute the command and echo warnings and progress to stdout
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Print the stdout and stderr
    for line in process.stdout:
        print(line.decode('ISO-8859-1').strip())

    # Delete the temporary file
    os.remove(temp_file_name)
    
    return output_vrt

 

if __name__=='__main__':
    run_organize(
        r'l:\10_IO\2307_roads\outs\rim_2019\downscale\04wsh\ems\meta_3_fdsc_stack_ems_20240126.pkl',
        #r'l:\10_IO\2307_roads\outs\rim_2019\downscale\03fdsc\donau\meta_donau_1098_20240121.pkl',
        use_cache=True,
        processes=None
        )
    
    
    
    
    
    
    
    
    
    