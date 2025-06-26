'''
Created on Oct. 25, 2023

@author: cefect

apply the flood downscaler to some grids
'''
#===============================================================================
# IMPORTS---------
#===============================================================================
import os, hashlib, tempfile

from datetime import datetime

import numpy as np
import numpy.ma as ma
import rasterio as rio
from rasterio.enums import Resampling, Compression

from definitions import wrk_dir

module_out_dir = os.path.join(wrk_dir, 'fdsc')

#from haz.rim2019.coms import _filesearch

from fdsc.control import Dsc_Session
#from fdsc.costGrow import CostGrow

from hp.dirz import _filesearch
from hp.rio import assert_spatial_equal, get_ds_attr, write_clip

from hp.hyd import get_wsh_rlay

from haz.hp.basic import (
    view, get_temp_dir, today_str, 
    init_log,dstr, get_log_stream 
    )

#===============================================================================
# def extract_mask(
#         dem_fp,
#         out_dir=None,
#         ):
#     """extract the water body  mask from the burned DEMs"""
#     
#     if out_dir is None:
#         out_dir = os.path.join(wrk_dir, 'domain')
#     if not os.path.exists(out_dir):os.makedirs(out_dir)
#     
#     with rio.open(dem_fp) as src:
#         ar = src.read(1)
#         
#         # Create a mask for all pixels with the value 9999
#         mask = np.where(ar == 9999, 1, 0)
#     
#     # Write the mask to a new raster file
#     fn_raw = os.path.splitext(os.path.basename(dem_fp))[0]
#     ofp = os.path.join(out_dir, f'{fn_raw}_wbody.tif')
#     with rio.open(ofp, 'w', **src.profile) as dst:
#         dst.write(mask, 1)
#     
#     print(f'wrote to \n    {ofp}')
#         
#     return ofp
#===============================================================================

def get_wbody_domain(
        basinName,
        data_dir = r'l:\10_IO\2307_roads\ins\2019_RIM\burned_domains',
        base_fp=None,
        out_dir=None,
        use_cache=True,
        ):
    """extract the water body mask from teh burned domains. resample to new DEM"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(module_out_dir, 'wbody')
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    #search for the file
    dem_fp = _filesearch(data_dir, f'{basinName}.tif')
    print(f'found burned dem at \n    {dem_fp}')
    
    
    #get resampling parameters
    scale = get_resolution_ratio(dem_fp, base_fp)
    print(f'resacling to {scale}')
    
    #===========================================================================
    # output paths
    #===========================================================================
     
    uuid = hashlib.shake_256(f'{data_dir}_{base_fp}_{scale}'.encode("utf-8")).hexdigest(8) 
    ofp = os.path.join(out_dir, f'{basinName}_wbody_{uuid}.tif')
    
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # load and rescale
        #===========================================================================
        with rio.open(dem_fp) as dataset:
            out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
            
            data_rsmp = dataset.read(1,
                    out_shape=out_shape,
                    resampling=Resampling.nearest,
                    masked=True)
            
            msk_rsmp = dataset.read_masks(1, 
                        out_shape=out_shape,
                        resampling=Resampling.nearest, #doesnt bleed
                        )
                    
            
            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data_rsmp.shape[-1]),
                    (dataset.height / data_rsmp.shape[-2])
                )
            
     
            outres = dataset.res[0]/scale
            
            res_mar = ma.array(np.where(data_rsmp == 9999, 1, 0), 
                               mask=np.where(msk_rsmp==0, True, False), 
                               fill_value=dataset.nodata)
            
            # Create a mask for all pixels with the value 9999
            #build new profile
            prof_rsmp = {**dataset.profile, 
                          **dict(
                              width=data_rsmp.shape[-1], 
                              height=data_rsmp.shape[-2],
                              transform=transform,
                              )}
            
        
        #===========================================================================
        # #write
        #===========================================================================
    
        with rio.open(ofp, 'w',**prof_rsmp) as ds:
            ds.write(res_mar, indexes=1, masked=False)
        
        print(f'wrote w/ resolution = {outres} to \n    {ofp}')
    else:
        print(f'file exists')
        
        
    return ofp

def get_wse_ar(
        wsh_fp,
        dem_fp,
        out_dir=None,
        use_cache=True,
        dem_nulls=[9999, -9999],
        
        ):
    """convert RFM WSH to WSE raster w/ special treatment for DEM nullss. use cache and write to .tif
    
    Params
    ----------
    wsh_fp: str
        filepath to WSH raster
    
    dem_fp:  str
        filepath to DEM raster with special null values
    """
    
    #===========================================================================
    # setup outputs paths
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(module_out_dir, 'wse')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    fn_raw = os.path.splitext(os.path.basename(wsh_fp))[0]
    
    uuid = hashlib.shake_256(f'{dem_fp}'.encode("utf-8")).hexdigest(4) 
    ofp = os.path.join(out_dir, f'{fn_raw}_wse_{uuid}.tif')
    
    
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        #check that the resolution of both 
        #assert_spatial_equal(wse_fp, dem_fp)
        
        #load WSE
        with rio.open(wsh_fp) as wsh_src:
            prof = wsh_src.profile
            prof['nodata']=-9999.0
            
            wsh_ar_raw = wsh_src.read(1, masked=True)
            
            #add nulls to mask

            
            wsh_ar= ma.array(wsh_ar_raw.data,
                 mask = np.logical_or(wsh_ar_raw.mask,
                                      np.logical_or(
                                          np.isnan(wsh_ar_raw.data),
                                          wsh_ar_raw.data==0)),
                 fill_value=prof['nodata'])
            
            """
            with rio.open(os.path.join(out_dir, 'wsh_test.tif'), 'w', **prof) as dst:
                dst.write(wsh_ar_raw, 1, masked=False)
                
            """
            
            
            #load DEM (using the windows from teh WSE)
            
            
            with rio.open(dem_fp) as dem_src:
                window = rio.windows.from_bounds(*wsh_src.bounds, transform=dem_src.transform)
                dem_ar_raw = dem_src.read(1, masked=True, window=window)
                
                #add water body to mask
                dem_ar = ma.array(dem_ar_raw.data,
                     mask = np.logical_or(dem_ar_raw.mask, #data nulls
                                          np.isin(dem_ar_raw.data,dem_nulls) #water body mask
                                          ),
                     fill_value=-9999)
                
                
                #perform addition (take union of both masks)
                wse_ar = ma.array(wsh_ar.data+dem_ar.data,mask = np.logical_or(wsh_ar.mask, dem_ar.mask), 
                                  fill_value=-9999)
                
                """
                wsh_ar.max()
                dem_ar.max()
                wse_ar.max()
                plt.close('all')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()

 
                cax = ax.imshow(dem_ar, interpolation='nearest')
                cax = ax.imshow(wsh_ar, interpolation='nearest')
                
                cax = ax.imshow(wse_ar, interpolation='nearest')
                
                fig.colorbar(cax)
                plt.show()
                """
 
                
        #write
 
        with rio.open(ofp, 'w', **prof) as dst:
            dst.write(ma.filled(wse_ar), 1, masked=False)
            
        print(f'wrote {wse_ar.shape} to \n    {ofp}')
            
    else:
        print('file exists')
        
 
        
    return ofp
        
    
     
    
    
    
def downscale_raster(
        wse_fp,
        basinName,
        dem_fp, #
        
        aoi_fp=None,
        
        out_dir=None, temp_dir=None,
        ):
    """apply downscaler to  a single raster"""
    
    
    #===========================================================================
    # setup output
    #===========================================================================
    fn_raw = os.path.splitext(os.path.basename(wse_fp))[0]
    
    if out_dir is None:
        out_dir = os.path.join(module_out_dir, 'downscale_raster', fn_raw)
 
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(out_dir, 'temp')
        
        #temp_dir = get_temp_dir(temp_dir_sfx='py/temp/downscale_raster/'+today_str)
        
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        
    
 
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'downscaling {fn_raw}')
    
 
    #===========================================================================
    # extract water bodies mask
    #===========================================================================
 
    
    wbdy_fp = get_wbody_domain(basinName, base_fp  = dem_fp)
    
    #===========================================================================
    # execute costGrow
    #===========================================================================
    with Dsc_Session(run_name='r', relative=True, out_dir=out_dir, logger=log, aoi_fp=aoi_fp, tmp_dir=temp_dir
                     ) as ses:
        
        #=======================================================================
        # clip
        #=======================================================================
        demC_fp, wseC_fp = ses.p0_clip_rasters(dem_fp, wse_fp, out_dir=temp_dir)
        
        #=======================================================================
        # build cost friction
        #=======================================================================
        cost_fric_fp = get_costFriction(wbdy_fp, bbox=ses.bbox, log=log, out_dir=temp_dir)
 

        #=======================================================================
        # downscale
        #=======================================================================
        ofp, meta_lib = ses.run_dsc(demC_fp, wseC_fp,
                                    rkwargs=dict(cost_fric_fp=cost_fric_fp,
                                                 #clump_cnt=2, #take both sides of the river
                                                 clump_method='pixel',
                                                 loss_frac=0.0005,
                                                 ), write_meta=False)
        
    return ofp, demC_fp
    

def get_costFriction(
        rlay_fp, bbox=None, out_dir=None, 
        use_cache=True,
        log=None,
        ):
    """clip and prep the cost friction raster"""
    
    #===========================================================================
    # get filepaths
    #===========================================================================
    uuid = hashlib.shake_256(f'{rlay_fp}_{bbox}'.encode("utf-8")).hexdigest(8) 
    ofp = os.path.join(out_dir, f'costFriction_{uuid}.tif')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # clip
        #===========================================================================
        
        rlayC_fp, _ = write_clip(rlay_fp, ofp=tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name, bbox=bbox)
        
        
        
        #1 everywhere and null at pwb
        with rio.open(rlayC_fp) as src:
            prof = src.profile
            mar_raw = src.read(1, masked=True)
            
            """this may need the mask from teh DEM as well"""
            mar = ma.array(np.where(mar_raw.data==0.0, 1.0, np.nan),
                     mask = np.logical_or(mar_raw.mask, mar_raw.data==1.0), fill_value=-9999)
            
        with rio.open(ofp, 'w', **prof) as dst:
            dst.write(mar, 1, masked=False)
            
        log.info(f'build costFriction raster at \n    {ofp}')
        
    else:
        log.debug('file exists')
    
    return ofp
    

def get_resolution_ratio( 
                             fp1,#coarse
                             fp2, #fine
                             ):
    s1 = get_ds_attr(fp1, 'res')[0]
    s2 = get_ds_attr(fp2, 'res')[0]
        
    return s1 / s2



def raster_resample_cache(
        dem_fp,
        out_shape,
         resampling=Resampling.nearest,

        ofp=None, log=None, out_dir=None, use_cache=True,
        ):
    """resample a DEM to match some new shape... with caching
    
    only works well for aggregation
  
    """
 
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    
 
        
    if ofp is None:
        if out_dir is None:
            out_dir=tempfile.gettempdir()
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        base_name = os.path.basename(dem_fp).replace('.asc', '') + f'{out_shape[1]}_{out_shape[2]}'
        #uid = get_hash(f'{base_name}_{haz_basin_index_fp}_{wsh_thresh}')
        ofp = os.path.join(out_dir, f'{base_name}.tif')
        
    if log is None:
        log = get_log_stream()
 
    #===========================================================================
    # build
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # load and rescale
        #===========================================================================
        with rio.open(dem_fp) as dataset:
            #out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
            
            data_rsmp = dataset.read(1,
                    out_shape=out_shape,
                    resampling=resampling,
                    masked=True)
            
            msk_rsmp = dataset.read_masks(1, 
                        out_shape=out_shape,
                        resampling=Resampling.nearest, #doesnt bleed
                        )
                    
            
            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data_rsmp.shape[-1]),
                    (dataset.height / data_rsmp.shape[-2])
                )
            
     
            #outres = dataset.res[0]/scale
            
            res_mar = ma.array(np.where(data_rsmp == 9999, 1, 0), 
                               mask=np.where(msk_rsmp==0, True, False), 
                               fill_value=dataset.nodata)
            
            # Create a mask for all pixels with the value 9999
            #build new profile
            prof_rsmp = {**dataset.profile, 
                          **dict(
                              width=data_rsmp.shape[-1], 
                              height=data_rsmp.shape[-2],
                              transform=transform,
                              )}
            
        
        #===========================================================================
        # #write
        #===========================================================================
    
        with rio.open(ofp, 'w',**prof_rsmp) as ds:
            ds.write(res_mar, indexes=1, masked=False)
        
        log.info(f'wrote w/ resolution = {outres} to \n    {ofp}')
    else:
        log.debug(f'file exists...skipping')
        
        
    return ofp
    


if __name__=="__main__":
 
    #extract_mask(r'l:\10_IO\2307_roads\ins\2019_RIM\burned_domains\ems.tif')
    #get_wbody_domain('ems')
    
    dem_fp = r'l:\10_IO\2307_roads\ins\DEM\DGM5\DGM5_ems_aoi1_4647.tif'
    
    #get the WSH raw
    wse2_fp = get_wse_ar(
        r'L:/10_IO/2307_roads/outs/rim_2019/04_rasters/tifs/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif',
        r'l:\10_IO\2307_roads\ins\2019_RIM\burned_domains\ems.tif'
        )
    
    #downsale teh WSE
    wse1_fp, demC_fp = downscale_raster(
        wse2_fp,'ems',dem_fp,
        aoi_fp=r'l:\02_WORK\NRC\2307_roads\04_CALC\aoi\aoi_ems01_20231025.gpkg',
        )
    
    #get WSH downscaled
    get_wsh_rlay(demC_fp, wse1_fp, out_dir = os.path.dirname(wse1_fp))
    
    print('done')
    