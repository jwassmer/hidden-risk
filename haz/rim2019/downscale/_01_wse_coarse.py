'''
Created on Jan. 10, 2024

@author: cef

construct WSE per hydro_basin x event
    get_wse_ar()
    
    use netcdf DataSets from l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_nuts3_collect\
    
    output as DataSet in same format as 04_nuts3_collect (for potential concat)
        
    use caching

'''

import os, hashlib, psutil, pickle
from datetime import datetime

import numpy as np
import numpy.ma as ma
import pandas as pd

import geopandas as gpd
import shapely.geometry as sgeo
from pyproj import CRS

import rasterio as rio
from rasterio.enums import Resampling, Compression

import xarray as xr


from haz.hp.basic import today_str, view, get_log_stream, dstr

from haz.rim2019.parameters import out_base_dir, wet_wsh_thresh, epsg_id

 

#===============================================================================
# helpers------
#===============================================================================
def get_od(name):
    out_dir = os.path.join(out_base_dir, 'wse_coarse', name)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return out_dir

def get_slog(name, log):
    if log is None:
        log = get_log_stream()
        
    return log.getChild(name)
        

get_hash = lambda x:hashlib.shake_256(x.encode("utf-8")).hexdigest(8) 

def get_haz_event_index(haz_basin_index_fp, haz_data_dir, log):
    assert os.path.exists(haz_basin_index_fp)
    #get default event meta from index location
    #===========================================================================
    # if haz_event_meta_fp is None:
    #     srch_dir =os.path.dirname(haz_basin_index_fp)
    #     fns = [e for e in os.listdir(srch_dir) if e.startswith(f'nuts3_collect_haz_dx')]
    #     assert len(fns)==1
    #     haz_event_meta_fp = os.path.join(srch_dir, fns[0])
    # assert os.path.exists(haz_event_meta_fp)
    #===========================================================================
    #get default data directory from index location
    if haz_data_dir is None:
        haz_data_dir = os.path.join(os.path.dirname(haz_basin_index_fp), 'event_ds')
    assert os.path.exists(haz_data_dir)
    #===========================================================================
    # load indexers
    #===========================================================================
    #load basin index
    haz_basin_index_df = gpd.read_file(haz_basin_index_fp, ignore_geometry=True).set_index('name')
    log.info(f'loaded basin index {str(haz_basin_index_df.shape)} from {os.path.basename(haz_basin_index_fp)}')
    #check ALL filepaths
    for i, row in haz_basin_index_df.iterrows():
        assert os.path.exists(os.path.join(haz_data_dir, row['haz_ds_fp'])), f'bad path on \n{row}'
    
    #check requested indexerse
    
    """
    view(haz_basin_index_df)
    
    haz_basin_index_df['haz_cnt'].astype(int).sum()
    """

    #===========================================================================
    # """no need to load the meta indejx"""
    # #load meta index
    # haz_index_dx = pd.read_pickle(haz_event_meta_fp)
    # log.info(f'loaded event meta {str(haz_index_dx.shape)} from {os.path.basename(haz_event_meta_fp)}')
    #
    # #check requsted
    # assert raster_index in haz_index_dx.index.get_level_values('raster_index')
    #
    # """
    # view(haz_index_df.head(100))
    # view(haz_index_dx.xs('ems', level=0).head(100))
    # """
    #===========================================================================
    log.debug(f'dataset paths valid')
    return haz_data_dir, haz_basin_index_df


def get_haz_ds_fp(basinName2, *args, geometry=False):
    
    haz_data_dir, haz_basin_index_df = get_haz_event_index(*args)
 
    #===========================================================================
    # #get meta for this selection
    #===========================================================================
    assert basinName2 in haz_basin_index_df.index
    s =  haz_basin_index_df.loc[basinName2, :]
    #s.loc['basinName2'] = s.name
    haz_meta_d = s.to_dict()
    haz_meta_d['basinName2'] = s.name
 
    
    #log.info(f'selected haz basin \n    {haz_meta_d}')
    
    ds_fp = os.path.join(haz_data_dir, haz_meta_d['haz_ds_fp'])
    assert os.path.exists(ds_fp), basinName2
    
    #===========================================================================
    # load geometry for thsi basin
    #===========================================================================
    if geometry:
        from definitions import basins_fp
        basin_poly = gpd.read_file(basins_fp).set_index('name').geometry[basinName2]
        haz_meta_d['geometry'] = basin_poly
    
    return ds_fp, haz_meta_d

#===============================================================================
# funcs----
#===============================================================================

def run_coarse_wse_stack(
        basinName2,
        #raster_index,
        
        haz_basin_index_fp = None,
        #haz_event_meta_fp = None,
        
        dem_fp = None,
        
        haz_data_dir = None,
        wsh_thresh = None,
        
        raster_index_l=None,

        ofp=None, log=None, out_dir=None, use_cache=True, tmp_dir=None,
        ):
    """build the coarse WSE from the DEM and WSH for this analysis basin
    
    Pars
    ----------
    haz_basin_index_fp: str
        filepath to the analysis basin index
        links each basin to its (coarse) selected event stack DataSet
        see haz.rim2019.nuts3._04_nuts3_collect()
        
 
        
    haz_data_dir: fp
        directory containing the event stack datasets
        
    dem_fp: fp
        filepath to original DEM (resolution should match)
        loads from parameters if NOne
        
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    
    if wsh_thresh is None:
        wsh_thresh = wet_wsh_thresh        
        
    assert not haz_basin_index_fp is None, 'must specify haz_basin_index_fp'
        
        
    if ofp is None:
        if out_dir is None:
            out_dir=os.path.join(out_base_dir, 'downscale','01coarse',basinName2,'wse')
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        base_name = f'{basinName2}'
        uid = get_hash(f'{base_name}_{haz_basin_index_fp}_{wsh_thresh}')
        ofp = os.path.join(out_dir, f'wetPts_{base_name}_{uid}.gpkg')
        
    log = get_slog('wse', log)
    
    #===========================================================================
    # get WSH datasets
    #===========================================================================
    ds_fp, haz_meta_d = get_haz_ds_fp(basinName2, haz_basin_index_fp, haz_data_dir, log)
    
    #===========================================================================
    # get DEM file
    #===========================================================================
    if dem_fp is None:
        from haz.rim2019.parameters import coarse_dem_fp_d
        dem_fp = coarse_dem_fp_d[haz_meta_d['basinName']]
        
    assert os.path.exists(dem_fp), f'bad path on DEM\n    {dem_fp}'
    
    #=======================================================================
    # load
    #=======================================================================

    
    with xr.open_mfdataset(ds_fp, parallel=True, engine='netcdf4') as ds:
        log.debug(f'loaded {ds.dims}' + 
                 f'\n    coors: {list(ds.coords)}' + 
                 f'\n    data_vars: {list(ds.data_vars)}' + 
                 f'\n    crs:{ds.rio.crs}'
                 f'\n    chunks:{ds.chunks}'
                 )
        
        #get the data array
        da =ds['inundation_depth']
        
        if not raster_index_l is None:
            log.warning(f'slicing to {len(raster_index_l)} raster_index')
            da = da.sel(raster_index=raster_index_l)
        
        #check
        max_df = da.groupby('raster_index').max(['x', 'y']).compute().to_dataframe().drop(['spatial_ref'], axis=1)
        bx = max_df['inundation_depth']<0.001
        if bx.any():
            raise ValueError(f'{basinName2} {bx.sum()}/{len(bx)} WSH events w/ no depths\n    {max_df[bx]}')
 
        
        #loop and write each raster to file
        meta_df = da.groupby('raster_index', squeeze=False).apply(
            _apply_wsh_to_wse, dem_fp=dem_fp, out_dir=out_dir, use_cache=use_cache, log=log,
            basinName2=basinName2
            ).compute().to_dataframe()
        
    log.info(f'finished on {meta_df.shape}')
        
    #===========================================================================
    # wrap
    #===========================================================================
    
    #join meta
    for k,v in haz_meta_d.items():
        meta_df[k] = v
        
    meta_dx = meta_df.set_index(list(haz_meta_d.keys()))
    
    #write meta
    meta_fn = f'meta_wse_{basinName2}_{len(meta_dx):03d}'
    meta_dx.to_pickle(os.path.join(out_dir, f'{meta_fn}.pkl'))
    meta_dx.to_csv(os.path.join(out_dir, f'{meta_fn}.csv'))
    
    log.info(f'wrote meta to \n    {meta_fn}')
    
    return meta_dx
    
    
    
        

def _apply_wsh_to_wse(da,
                      dem_fp=None,
                      basinName2=None,
                      
 
                      ofp=None, log=None, out_dir=None, use_cache=True, 
                      #skwargs=dict(),
                      dem_nulls=None,
                      #blocksize=800,
                      
                      encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},
                      ):
    """build WSE from WSH DataArray and DEM raster
    
    TODO: add some meta
    """
    
    #===========================================================================
    # #write to test pickle
    # with open(r'l:\10_IO\2307_roads\test\test_downscale_wse\test_apply_wsh_to_wse\inundation_depth_001.pkl', 'wb') as f:
    #     pickle.dump(da, f)
    # 
    #===========================================================================
    
    #===========================================================================
    # setup
    #===========================================================================
    log = get_slog('apply_wse', log)
    
    _kwargs = dict(out_dir=out_dir, use_cache=use_cache, log=log)
        
    raster_index = da.raster_index.item()
    
    if dem_nulls is None:
        from haz.rim2019.parameters import coarse_dem_nulls as dem_nulls
 
    
    #===========================================================================
    # setup outputs paths
    #===========================================================================
 
    
    base_name = f'{basinName2}_{raster_index:05d}'
    
    if ofp is None:
        uuid = hashlib.shake_256(f'{os.path.basename(dem_fp)}'.encode("utf-8")).hexdigest(8) 
        ofp = os.path.join(out_dir, f'{base_name}_wse_{uuid}.nc')
    
    #===========================================================================
    # #get some meta from the dataset
    #===========================================================================
    d=dict()
    
    for k in ['day', 'realisation', 'raster_index']:
        d[k] = da.coords[k].values[0]
    
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        log.info(f'building WSE for {da.shape} usind DEM \'{os.path.basename(dem_fp)}\'')
        
        """
        #write to raster
        ofp = os.path.join(out_dir, f'wsh_{base_name}.tif')
        da.encoding=dict()
        
        da.squeeze().reset_coords('raster_index', drop=True).rio.write_nodata(-9999
                  ).rio.to_raster(ofp, dtype='float32', compute=False, compress='LZW')
        
        """
        #drop to numpy array
        assert isinstance(da, xr.DataArray)
        
        wsh_ar = da.squeeze().values  
        wsh_mar= ma.array(wsh_ar,
                         mask = np.logical_or(np.isnan(wsh_ar),wsh_ar==0),
                         fill_value=-9999)       
 
        
        #load DEM (using the windows from teh WSE)        
        with rio.open(dem_fp) as dem_src:
 
 
            #check that the resolution of both 
            if not dem_src.res == tuple([abs(e) for e in da.rio.resolution()]):
                log.warning(f'resolution mismatch' +\
                            f'\n    dem_src.res: {dem_src.res}\n    da.rio.resolution:{da.rio.resolution()}')
            
 
            #load from DEM within bounds of the DataArray
            window = rio.windows.from_bounds(*da.rio.bounds(), transform=dem_src.transform)
            dem_ar_raw = dem_src.read(1, masked=True, window=window)
            
            assert wsh_ar.shape==dem_ar_raw.shape, f'spatially inconsistent... bounds mismatch?'
            
            
            #add water body to mask
            dem_mar = ma.array(dem_ar_raw.data,
                 mask = np.logical_or(dem_ar_raw.mask, #data nulls
                                      np.isin(dem_ar_raw.data,dem_nulls) #water body mask
                                      ),
                 fill_value=-9999)
            
            assert not dem_mar.mask.all()
            
            
            #perform addition (take union of both masks)
            wse_mar = ma.array(wsh_mar.data+dem_mar.data,mask = np.logical_or(wsh_mar.mask, dem_mar.mask), 
                              fill_value=np.nan)
            
            if wse_mar.mask.all():
                raise TypeError(f'got all nulls on {base_name}')
            
        #=======================================================================
        # rebuild xarray
        #=======================================================================
        #re-assemble
        wse_da = xr.DataArray(data=wse_mar.filled()[None, ...], 
                              coords=da.coords, dims=da.dims, name='WSE')
        
        #add rasterio data
        wse_da = wse_da.rio.write_nodata(-9999).rio.write_crs(da.rio.crs)
            
            
        #=======================================================================
        # write
        #=======================================================================
        wse_da.to_netcdf(ofp, mode ='w', format ='netcdf4', engine='netcdf4', compute=True,
                         encoding={'WSE':encoding},
                         )
            
        log.info(f'wrote {wse_da.shape} to \n    {ofp}')
        
        """
        #write to raster
        
        wse_da.rio.to_raster(r'l:\10_IO\2307_roads\scratch\wse_test.tif')
        """
 
            
    else:
        log.debug(f'file exists\n    {ofp}')
        #=======================================================================
        # """dont want to load everything into memory... just write everything"""
        # log.info(f'load_dataarray from existing file \n    {ofp}')
        # wse_da = xr.load_dataarray(ofp, format ='netcdf4', engine='netcdf4') 
        #=======================================================================
 
        
 
    
    return xr.DataArray(os.path.basename(ofp), coords=d, name='wse_da_fn')
    
    
    
if __name__=='__main__':
    run_coarse_wse_stack(
        'rhine_upper',
        #raster_index_l=[20174],
        use_cache=True,
        )
    
    
    
    
    
    
    
    
 
