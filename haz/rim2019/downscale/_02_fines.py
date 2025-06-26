'''
Created on Jan. 13, 2024

@author: cef
'''
#===============================================================================
# IMPORTS------
#===============================================================================
import os, hashlib, psutil, pickle
from datetime import datetime

import numpy as np
import numpy.ma as ma
import pandas as pd

import tqdm
import geopandas as gpd
import shapely.geometry as sgeo
from pyproj import CRS


from osgeo import gdal

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

import xarray as xr
#print(xr.__version__)
import rioxarray


from haz.hp.basic import today_str, view, get_log_stream, get_directory_size, dstr



from haz.rim2019.parameters import  out_base_dir, wet_wsh_thresh

 

#===============================================================================
# helpers------
#===============================================================================
def get_od(name):
    out_dir = os.path.join(out_base_dir, 'fines', name)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return out_dir

def get_slog(name, log):
    if log is None:
        log = get_log_stream()
        
    return log.getChild(name)
        

get_hash = lambda x:hashlib.shake_256(x.encode("utf-8")).hexdigest(8)


def _get_coords(ds, index_coln_l=['raster_index', 'dem_tile_x', 'dem_tile_y']):
    d = {k:v.values.tolist() for k, v in ds.coords.items() if k in index_coln_l}
    d2 = dict()
    for k, v in d.items():
        if k=='raster_index' and isinstance(v, int):
            d2[k] = [v]
        else:
            d2[k] = v

            
    
    return d2

def get_coord_aslist(coord):
    ri_l =  coord.values.tolist()
    if isinstance(ri_l, int): 
        ri_l = [ri_l] 
    
    return ri_l

#===============================================================================
# funcs-----
#===============================================================================





def run_chunk_to_dem_tile(
        basinName2,
        
        wse_ds_dir = None,
        
        dem_index_fp=None,
        dem_data_dir = None,
        
        wbdy_mask_fp = None,
        
        min_wet_cnt=5,
 
        epsg_id=None,
        
        raster_index_l=None, dem_fn_l=None,
        
         log=None, out_dir=None, tmp_dir=None, use_cache=True,
          **kwargs):
    """prepare fine resolution input layers by spatially clipping to the DEM tile.
    outputs a DataSource w/ DEM, WSE, and WBDY
    
    
    Pars
    ---------
    wse_ds_dir: fp
        directory to WSE datasets 
        see haz.rim2019.downscale._01_wse_coarse.run_coarse_wse_stack()
        
    dem_index_fp: fp
        filepath to the DEM index vector layer
        
        
    wbdy_mask_fp: fp
        filepath to the rim2019 burned domain (which contains the burned channel areas)
        
    
    raster_index_l: list, optional
        for slicing the dataset (for DEV)
        
    dem_fn_l: list, optional
        for chunking to specific dem files (for DEV)
        
    min_wet_cnt: int
        minimum number of pixels to include in a tile
        for excluding tiles with very small amounts of flooding
        
        
    Parallelization
    ------------
    not set up well... dask seems to be activating multiple workers but not speeding things up
        see _02_fines_dask
    as we don't use many numpy funcs, this would have been better to use simple multiprocess to divide along dem?
    
    really this is just a big rechunk and combine.. so its difficult to parallelize
    
        
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
 
        
    if dem_index_fp is None:
        from definitions import dgm5_index_fp as dem_index_fp
        
    if dem_data_dir is None:
        from definitions import dgm5_search_dir as dem_data_dir
        
    if epsg_id is None:
        from haz.rim2019.parameters import epsg_id
        
    if wbdy_mask_fp is None:
        """masks are on the rim2019 DEMs"""
        from haz.rim2019.parameters import basinName2_d
        from definitions import coarse_dem_fp_d
        wbdy_mask_fp = coarse_dem_fp_d[basinName2_d[basinName2]]
        
    if wse_ds_dir is None:
        wse_ds_dir = os.path.join(out_base_dir, 'downscale', '01coarse', basinName2, 'wse')
        
 
    log = get_slog('chunk', log)
    
    if out_dir is None: out_dir = os.path.join(out_base_dir, 'downscale', '02fine', basinName2)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    #===========================================================================
    # load DEM index
    #===========================================================================
    dem_index_gdf = gpd.read_file(dem_index_fp)
    assert os.path.exists(dem_data_dir)
    
    log.info(f'loaded {len(dem_index_gdf)} dem tiles')
    
    #get the crs from the first DEM tile
    dem_fp_i = os.path.join(dem_data_dir, dem_index_gdf.iloc[0]['dgm5_Name']+'.asc')
    with rio.open(dem_fp_i, 'r') as dem_ds:
        dem_crs = dem_ds.crs
        
    
    """not sure why this doesn't work.. s hould be 3035
    dem_crs.to_epsg()"""
 
    #===================================================================
    # reproject the waterbody masks
    #===================================================================
    """this introduces a ~10-50m shift"""
    
    wbdy_mask_rprj_fp = os.path.join(out_dir, f'wbdy_mask_{basinName2}_3035.tif')
    if (not os.path.exists(wbdy_mask_rprj_fp)) or (not use_cache):
        log.info(f'reprojecting the waterbodies to match DEM')
        gdal.Warp(wbdy_mask_rprj_fp, wbdy_mask_fp, 
                  options= gdal.WarpOptions(dstSRS=dem_crs),
                   callback=gdal.TermProgress_nocb, # simple progress report to the terminal window
                   )
  
    #=======================================================================
    # load WSE-----
    #=======================================================================    
    #get file list
    fp_l = [os.path.join(wse_ds_dir, e) for e in os.listdir(wse_ds_dir) if e.endswith('.nc')]
    
    log.info(f'loading {len(fp_l)} WSE datasets from \n    {wse_ds_dir}')
 

    with xr.open_mfdataset(fp_l, 
                           engine='netcdf4', format='netcdf', 
                           #combine='by_coords', 
                           parallel=False,
                           ) as ds:
 
        log.debug(f'loaded {ds.dims}' + 
                 f'\n    coors: {list(ds.coords)}' + 
                 f'\n    data_vars: {list(ds.data_vars)}' + 
                 f'\n    crs:{ds.rio.crs}'
                 f'\n    chunks:{ds.chunks}'
                 )
        
        """no... this will be too big for most basins
        #load to memory
        ds = ds.load()"""
        
        #get the data array
        da =ds['WSE'].rio.write_crs(epsg_id)
        
        da.attrs['basinName2'] = basinName2 #add this meta
        
        #dev slicing
        if not raster_index_l is None:
            log.warning(f'slicing to {len(raster_index_l)} raster_index')
            da = da.sel(raster_index=raster_index_l)
        
        #reproject to match the DEM
        """using DEM crs to avoid heavy operations on hi-res.. just reproject output"""
        da_rp = da.rio.reproject(dem_crs)
        

        #=======================================================================
        # identify valid DEM tiles    
        #=======================================================================
        """not so sure about this.... see discussion at the top of haz.rim2019.downscale.pipe_fdsc_nuts3"""
        #pre-select tiles based on this basin-dataArray extents
        bx = dem_index_gdf.geometry.intersects(sgeo.box(*da.rio.bounds()))
        assert bx.any()
        dem_index_df = dem_index_gdf.loc[bx, ['dgm5_Name', 'geometry']]
        log.info(f'selected {bx.sum()}/{len(bx)} DEM tiles that intersect with the DataArray for {basinName2}')
        
        dem_fp_d = identify_valid_dem_tiles(dem_index_df,dem_data_dir,da_rp, 
                                            min_wet_cnt=min_wet_cnt,
                                 out_dir=out_dir, use_cache=use_cache, log=log) 
        
        
        #=======================================================================
        # loop on DEM tiles------      
        #=======================================================================
        #dev clip
        if not dem_fn_l is None:
            dem_fp_d = {k:v for k,v in dem_fp_d.items() if os.path.basename(v).replace('.asc', '') in dem_fn_l}
            assert len(dem_fp_d), 'got no dem tiles after slicing'
            
            log.warning(f'\n\nSLICING to {len(dem_fn_l)} DEM tiles\n\n')
            use_cache=False
        
        res_d = dict()
        log.info(f'looping on {len(dem_fp_d)} valid tiles')
 
        for i, dem_fp in tqdm.tqdm(dem_fp_d.items(), desc='fines'):

            log.debug(f'{i+1}/{len(dem_fp_d)}loading DEM from \n    {dem_fp}\n----------------------\n\n') 
            
            with rio.open(dem_fp, 'r') as dem_ds:
 
                #clip teh datasource
                da_clip = da_rp.rio.clip_box(*dem_ds.bounds)
 
 
            #===================================================================
            # resample and merge
            #===================================================================
            res_d[i] = merge_dem_wse_ds(da_clip,dem_fp,wbdy_mask_rprj_fp,
                                          log=log.getChild(f'{i:03d}'), out_dir=out_dir, use_cache=use_cache,
                                          min_wet_cnt=min_wet_cnt,
                                          **kwargs)
            
        #=======================================================================
        # close dataset
        #=======================================================================
            
            
            
    #=======================================================================
    # meta-------
    #=======================================================================
    sfx = f'fine_data_chunk_{basinName2}_{today_str}'
    log.info(f'finished on {len(res_d)}')
    
 
    
    #join resample meta to DEM meta
    meta_gdf = dem_index_gdf.loc[dem_fp_d.keys(), ['dgm5_Name', 'geometry']].join(pd.DataFrame.from_dict(res_d).T)
    
    #===========================================================================
    # prep for geo
    #===========================================================================
    #convert to geometry
    
    
    meta_gdf['shape_str'] = meta_gdf['shape'].apply(lambda x:str(x)).astype(str)
    meta_gdf = meta_gdf.astype({k:float for k in ['min_wet_cnt', 'wet_cnts_max'] if k in meta_gdf.columns})
 
    meta_gdf['wet_cnt_fail'] = meta_gdf['wet_cnt_fail'].fillna(False)
    
    #===========================================================================
    # #handle fail
    #===========================================================================
    """should just be those that dont satisfy the wet count... not really a failure"""
    
    bx = meta_gdf['wet_cnt_fail']
    if bx.any():
        log.warning(f'{bx.sum()}/{len(bx)} failed the wet count')
        """too hard to get failed data to conform to spatial writer"""
        fail_gdf = meta_gdf.loc[bx, :].dropna(how='all', axis=1)
        meta_gdf1 = meta_gdf.loc[~bx, :]
        
    else:
        fail_gdf=None
        meta_gdf1 = meta_gdf.copy()
 
        
    """
    view(meta_gdf)
    meta_gdf.columns
    """
    ofp_d=dict()
    #=======================================================================
    # #spatial
    #=======================================================================

    
    #clean up some fields for spatial writing
    meta_gdf1['raster_index_cnt'] = meta_gdf1['raster_index'].apply(lambda x:len(x))
    meta_gdf1['raster_index_str'] = meta_gdf1['raster_index'].astype(str)
    meta_gdf = meta_gdf.drop('geometry', axis=1).join(meta_gdf['geometry']).set_geometry('geometry')
    
    meta_gdf1 = meta_gdf1.astype({k:float for k in ['size', 'raster_index_cnt']})
    
 
    ofp_d['meta_gdf'] = os.path.join(out_dir, f'meta_{len(meta_gdf1)}_{sfx}.gpkg')
    
    
    meta_gdf1.drop(['raster_index', 'shape'], axis=1).to_file(ofp_d['meta_gdf'])
    log.info(f'wrote {meta_gdf1.shape} spatial meta to\n    %s'%ofp_d['meta_gdf'])
    
    if not fail_gdf is None:
        ofp_d['fail_gdf'] =os.path.join(out_dir, f'fail_{len(fail_gdf)}_{sfx}.gpkg')
        
        
        #tyep fixing
        fail_gdf['shape_str'] = fail_gdf['shape'].apply(lambda x:str(x)).astype(str)
        
        fail_gdf = fail_gdf.drop('geometry', axis=1).join(fail_gdf['geometry']).set_geometry('geometry')
        
        
        fail_gdf.drop(['shape'], axis=1).to_file(ofp_d['fail_gdf'])
        log.warning(f'wrote {fail_gdf.shape} fail metadtaa to \n    %s'%(ofp_d['fail_gdf']))
        """
        fail_gdf.dtypes
        view(fail_gdf)
        """
    #=======================================================================
    # #aspatial
    #=======================================================================
    
    
    #expand out the raster_index
    """
    view(meta_gdf1.drop(['fp', 'geometry'], axis=1))
    view(dx)
    """
    dx = meta_gdf1.loc[:, ['dgm5_Name', 'fn', 'dem_max', 'wbdy_cnt', 'raster_index']].explode('raster_index').reset_index().rename(columns={'index':'dem_index'})
    dx['basinName2'] = basinName2
    dx = dx.join(dx['dgm5_Name'].str.split('_', expand=True).iloc[:, [1,2]].rename(columns={1:'E', 2:'N'}),how='left')
    
    dx=dx.set_index(['basinName2', 'dem_index','E', 'N', 'dgm5_Name', 'raster_index'], drop=True).sort_index()
    
    #to pickl
    ofp_d['meta_pkl'] = os.path.join(out_dir, f'meta_{len(dx)}_{sfx}.pkl')
    dx.to_pickle(ofp_d['meta_pkl'])
    
    #to csv
    ofp_d['meta_csv'] = os.path.join(out_dir, f'meta_{len(dx)}_{sfx}.csv')
    dx.to_csv(ofp_d['meta_csv'])
    
    log.info(f'wrote {dx.shape} aspatial meta to\n    %s'%ofp_d['meta_csv'])
    
    #=======================================================================
    # gruoped
    #=======================================================================
    ri_meta = dx['fn'].groupby('raster_index').count()
    
    ofp_d['meta_ri_csv'] = os.path.join(out_dir, f'meta_raster_index_{len(ri_meta)}_{sfx}.csv')
    ri_meta.to_csv(ofp_d['meta_ri_csv'])
    
    log.info(f'finished on {len(ri_meta)} raster_index')
    log.debug(ri_meta)
    
    dem_meta = dx['fn'].groupby('dgm5_Name').count()
    log.info(f'and on {len(dem_meta)} dem_index')
    log.debug(dem_meta)
    
    #=======================================================================
    # wrap
    #=======================================================================
    
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'outdir_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'wrote meta to \n{dstr(ofp_d)}')
    
    return meta_gdf
    
def identify_valid_dem_tiles(dem_index_df,dem_data_dir,da_raw,
                             min_wet_cnt=None,
                             
                             ofp=None, log=None, out_dir=None, use_cache=True,
                             ):
    """loop through the DEM tiles and identify those with an event intersect"""
    
    #===========================================================================
    # setup
    #===========================================================================
    basinName2=da_raw.attrs['basinName2']
    if min_wet_cnt is None: min_wet_cnt=1
 
    
    if ofp is None:        
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        base_name = f'{basinName2}'
        uid = get_hash(f'{dem_index_df}_{da_raw.rio.bounds()}')
        ofp = os.path.join(out_dir, f'valids_{base_name}_{uid}.pkl')
        
    log = get_slog('vTiles', log)
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        
        #compress to maximum
        da = da_raw.max('raster_index', keep_attrs =True)
        #===========================================================================
        # loop through each tile
        #===========================================================================
        res_d=dict()
        for i, row in tqdm.tqdm(dem_index_df.iterrows(), total=len(dem_index_df), desc='identify dem tiles'):
            
            #log.info(f'\n{i+1}/{bx.sum()} on DEM tile %s\n----------------'%row[0])
            
            #===================================================================
            # load teh DEM tile (check intersect +  basic clip)
            #===================================================================
            dem_fp = os.path.join(dem_data_dir, row.iloc[0]+'.asc')
            log.debug(f'loading DEM from \n    {dem_fp}')
            
            if not os.path.exists(dem_fp):
                log.error(f'missing DEM tile {os.path.basename(dem_fp)}... skipping')
                continue
            
            with rio.open(dem_fp, 'r') as dem_ds:
        
                #clip teh datasource
                da_clip = da.rio.clip_box(*dem_ds.bounds)
                
                #===============================================================
                # check intersect
                #===============================================================
                if da_clip.notnull().sum().item()<min_wet_cnt:
                    log.debug(f'wet pixel count less than minmum {min_wet_cnt}.. skipping')
 
                    continue
                
            res_d[i] = dem_fp
            
        #=======================================================================
        # wrap
        #=======================================================================
        
        #write
        with open(ofp, 'wb') as f:
            pickle.dump(res_d, f)
            
        log.info(f'wrote {len(res_d)} to \n    {ofp}')
        
        
    #===========================================================================
    # load from cache        
    #===========================================================================
    else:
        log.info(f'file exists... loading from cache')
        with open(ofp, 'rb') as f:
            res_d = pickle.load(f)
            
            
            
    #===========================================================================
    # wrap
    #===========================================================================
    assert len(res_d)>0
    log.debug(f'finished on {len(res_d)}')
    
    return res_d
        
        
        
    
 
            
            
def merge_dem_wse_ds(da_raw,  dem_fp, wbdy_mask_fp,
                     
                     min_wet_cnt=None,
                                  
                  encoding =  {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},

                  
                  ofp=None, log=None, out_dir=None, use_cache=True,
                  write_tiff=False,
                  ):
    
    """clip data and combine into DataSource for a single DEM tile
    
    """
    """
    #write test data
    
    with open(r'l:\10_IO\2307_roads\test\test_downscale_fdsc\merge_dem_wse_ds\da_clip.pkl', 'wb') as f:
        pickle.dump(da_clip, f)
    
    """
    start = datetime.now()
    #===========================================================================
    # setup
    #===========================================================================
 
    
    ri_ar = da_raw.coords['raster_index'].values
    
    if ofp is None:        
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        base_name = os.path.basename(dem_fp).replace('.asc', '')
        uid = get_hash(f'{ri_ar}_{da_raw.shape}_{os.path.basename(wbdy_mask_fp)}_{min_wet_cnt}')
        ofp = os.path.join(out_dir, f'fine_{base_name}_{len(da_raw):03d}_{uid}.nc')
        
    log = get_slog('merge', log)
 
    #log.debug(ri_ar)
    
    
    
    #start meta
    meta_d = {'min_wet_cnt':min_wet_cnt, 'dem_fn':os.path.basename(dem_fp), 'wbd_fn':os.path.basename(wbdy_mask_fp)}
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        
        da_d = dict()
        log.debug(f'on coarse WSE w/ {da_raw.notnull().sum().item()}/{da_raw.size} wet pixels')
 
        #===========================================================================
        # get properties from DEM
        #===========================================================================
        with rio.open(dem_fp, 'r') as ds1:
            crs=ds1.crs
            dem_shape = ds1.shape 
            transform=ds1.transform
            bounds = ds1.bounds
 
        meta_d.update({'shape':dem_shape})
        #===========================================================
        # prep the WSE-----
        #=========================================================== 
        #=======================================================================
        # # clean up WSE on raster index 
        #=======================================================================
        da_slice = da_raw.dropna(how='all', dim='raster_index')
        
        if not min_wet_cnt is None:
            #get only those satisfying the minum wet count
            wet_cnts = da_slice.notnull().sum(dim=['x', 'y']).values
            meta_d['wet_cnts_max'] = np.max(wet_cnts)
            bx_ar = wet_cnts>min_wet_cnt
            meta_d['wet_cnt_fail']= (not np.any(bx_ar))
            
            if meta_d['wet_cnt_fail']:
                
                log.warning(f'no layers satisfied the min_wet_cnt {min_wet_cnt}... skipping')
                return meta_d
            
            log.debug(f'slicing to {bx_ar.sum()}/{len(bx_ar)} that satisfy the min_wet_cnt')
            ri_l = da_slice.coords['raster_index'].values[bx_ar].tolist()
            da_slice2 = da_slice.sel(raster_index=ri_l)
            
        else:
            da_slice2 = da_slice       
        
        #=======================================================================
        # #resample the WSE
        #=======================================================================
        
        log.debug(f'resampling WSE DataArrray from {da_slice2.shape} to {dem_shape}')
        
        da1 = da_slice2.rio.reproject(crs, shape=dem_shape, transform=transform, 
            resampling=Resampling.bilinear) 
            
        da1.attrs={} 
        
        da_d['WSE'] = da1
        
        ri_l = get_coord_aslist(da1['raster_index'])
        log.debug(f'reprojected WSE w/ raster_index={ri_l}')
        """
        import matplotlib.pyplot as plt
        plt.show()
        da1.plot()
        da1.rio.write_crs(crs).rio.to_raster(r'l:\10_IO\2307_roads\scratch\da_resamp1.tif')
        
        """
        
        
        #===========================================================
        # load the DEM data as rioXarray----
        #===========================================================
        da2 = rioxarray.open_rasterio(dem_fp, masked=True).squeeze().reset_coords(drop=True)
        da2.attrs = {} #clear these
        
        da_d['DEM'] = da2
 
        
        #=======================================================================
        # resample and extract the waterbody mask------
        #=======================================================================
        log.debug(f'resampling the waterbody mask {os.path.basename(wbdy_mask_fp)}')
        
        with rio.open(wbdy_mask_fp, mode='r') as wbdy_ds:
            assert wbdy_ds.crs==crs
                        
            window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, wbdy_ds.transform)
            
            
            data_rsmp = wbdy_ds.read(1,
                    out_shape=dem_shape, window=window,
                    resampling=Resampling.nearest,
                    masked=True)
            
            msk_rsmp = wbdy_ds.read_masks(1, 
                        out_shape=dem_shape,window=window,
                        resampling=Resampling.nearest, #doesnt bleed
                        )
                    
            
            # scale image transform
            transform = wbdy_ds.transform * wbdy_ds.transform.scale(
                    (wbdy_ds.width / data_rsmp.shape[-1]),
                    (wbdy_ds.height / data_rsmp.shape[-2])
                )
 
            
            mar = ma.array(np.where(data_rsmp == 9999, 1, 0), 
                               mask=np.where(msk_rsmp==0, True, False), 
                               fill_value=0)
            
 

        #create a rioxarray dataset with all the same spatial info from rastserio
        da_d['WBDY'] = xr.DataArray(data=np.where(mar.filled()==1, 1, np.nan),coords=da2.coords, dims=da2.dims) 
        
 
        
 
        #===================================================================
        # merge
        #===================================================================
        l = list()
        for k, da in da_d.items():

            
            #checks
            if not k=='WBDY':
                assert not da.isnull().all(), k
            
            if len(da.shape)==3:
                s = da.shape[1:]
            elif len(da.shape)==2:
                s = da.shape
            else:
                raise IOError(da.shape)
            
            assert s==dem_shape, k
            assert da.rio.bounds()==bounds, k
 
            #standardize
            da = da.rio.write_crs(crs).rename(k).fillna(-9999).rio.write_nodata(-9999)
 
            l.append(da.to_dataset(name=k))
            
            
            
        
        ds = xr.merge(l)
 
        #=======================================================================
        # post-----
        #=======================================================================
        #add DEm indexers
        x,y = tuple([int(e) for e in base_name.split('_')[1:3]])
        ds = ds.assign_coords(dem_tile_x=[x]).assign_coords(dem_tile_y=[y]).squeeze()
        
        if 'basinName2' in da_raw.attrs:
            ds.attrs['basinName2'] = da_raw.attrs['basinName2']
 
        
        ds['WSE'].encoding.update(encoding)
        ds['DEM'].encoding.update(encoding)
        ds['WBDY'].encoding.update({'zlib': True, 'complevel': 5, 'dtype': 'float32'})
        
        #===================================================================
        # write to netcdf
        #===================================================================
        
        log.debug(f'writing {ds.dims} to \n    {ofp}')
        ds.to_netcdf(ofp, format='netCDF4',engine='netcdf4', mode='w')
        
        """
        too complicated to try and write this in a more logical folder structure
        
        WSE
        - raster_index
            - dem_index
            
        DEM
        - dem_index
        
        
        """
        
        #=======================================================================
        # write GeoTiff
        #=======================================================================
        if write_tiff:
            log.debug(f'writing to GeoTiff')
            ofp1 = os.path.join(os.path.dirname(ofp), 'tifs', os.path.basename(ofp).replace('.nc', '.tif'))
            if not os.path.exists(os.path.dirname(ofp1)):
                os.makedirs(os.path.dirname(ofp1))
            
            #convert dataset to array (creates 'variable' as a new coord)
            da = ds.to_array().fillna(-9999).rio.write_nodata(-9999)
            
            #write this 3D DataArray as a multi-band raster
            da.rio.to_raster(ofp1, dtype='float32', compute=False, compress='LZW')
            
            log.info(f'wrote {da.shape} GeoTiff to \n    {ofp1}')
 
        
        #===========================================================================
        # wrap
        #===========================================================================
        meta_d.update({
                        'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                        #'outdir_GB':get_directory_size(out_dir),
                        'output_MB':os.path.getsize(ofp)/(1024**2)
                        })
        
        log.debug(meta_d)
        
        ds.close() #I don't think this does an ything
        ds=None
        
    else:
        
        meta_d.update({'from_cache':True})
        log.debug('file exists... skipping')
        
    #===========================================================================
    # get meta
    #===========================================================================
    with xr.open_dataset(ofp, engine='netcdf4', cache=True) as ds:
        da = ds['WSE'].load()
        meta_d.update(
            {'shape':da.shape, 'size':da.size, 'raster_index':tuple(get_coord_aslist(da['raster_index'])),
             'fp':ofp, 'fn':os.path.basename(ofp),
             'dem_max':ds['DEM'].max().item(), 'wbdy_cnt':ds['WBDY'].sum().item(),
             'min_wet_cnt':min_wet_cnt,
             }
            )
    if not 'wet_cnt_fail' in meta_d:
        meta_d['wet_cnt_fail']=False
    
    
    
    return meta_d
 
 
if __name__=='__main__':
    run_chunk_to_dem_tile(
        'ems',
        #wse_ds_dir = r'l:\10_IO\2307_roads\outs\rim_2019\downscale\01coarse\ems\wse',
        #raster_index_l=[4409],
        #dem_fn_l = ['dgm5_4380_2800_20'],
        use_cache=True,
        write_tiff=False
        )



        
    