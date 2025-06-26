'''
Created on Jul. 29, 2023

@author: cefect

try2 of unzipping and extracting to netcdf

SWIM2019 'inundation_depth' results are unzipped per basin and extracted to a single netcdf per basin
script is parallelized using concurrent.futures
results are written to os.path.join(lib_dir, basinName, '01_extract2')
layers with all nulls (i.e., all zeros) are skipped

'''
#===============================================================================
# IMPORTS--------
#===============================================================================
import os, zipfile, shutil, hashlib, psutil, logging
import concurrent.futures
from datetime import datetime
import numpy as np
import pandas as pd
 

from osgeo import gdal # Import gdal before rasterio

import xarray as xr
import netCDF4
import rioxarray


from definitions import wrk_dir, epsg_id
from haz.hp import (
    view, get_temp_dir, today_str, get_directory_size, _wrap_rprof,
    init_log,dstr, dask_profile_func, get_log_stream, 
    )
from haz.rim2019.coms import asc_lib_d, load_lookup_df, _filesearch, lib_dir, _filesearch_all
 


#lib_dir =os.path.join(wrk_dir, r'lib\rim2019')

 


def extract_asc_to_nrc2(fp,
                       temp_dir=None,
                       ofp=None,
                       encoding=None,
                       coords=dict(),
                       name='inundation_depth',
                       logger=None,
                       #chunks_d={'x':100, 'y':100, 'band':1}
                       ):
    """unzip asc and convert to nrc
    
    changed to not use dask
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now()
    #setup cache
    if temp_dir is None:
        temp_dir= get_temp_dir(temp_dir_sfx='py/temp/mosaic_from_lookup')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    log = logger.getChild('extract')
    
    log.debug(f'extracting and converting on {fp}')
    
    #set encoding
    if encoding is None:
        #encoding = {'zlib': True, 'complevel': 9} #0.36MB, 19.5 secs
        """could do more trials... not sure how least_significant_digit works"""
        encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2}
    
 
    #===========================================================================
    # setup
    #===========================================================================
    #===========================================================================
    # uuid = hashlib.md5(f'{fp}'.encode("utf-8")).hexdigest()
    # 
    # #temp dir
    # temp_dir_i = os.path.join(temp_dir, uuid)
    # if not os.path.exists(temp_dir_i): 
    #     os.makedirs(temp_dir_i)
    #===========================================================================
    
    #===========================================================================
    # #unzip
    #===========================================================================
    with zipfile.ZipFile(fp, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    log.debug(f'    extracted to {temp_dir}')
        
    # get the asc 
    asc_fp = _filesearch(temp_dir)
    
    #===========================================================================
    # #load
    #===========================================================================
    #print(f'     loading array from {os.path.basename(asc_fp)}')
     
 
    with rioxarray.open_rasterio(asc_fp, 
                                    parse_coordinates=True, #seems like it would be faster if this was false... but easier to apply encoding w/ it as true 
                                    #chunks=None, #finished in 93.554056 #RAM Used (GB): 16.112562176, storage (GB): 1.5712996535003185
                                    #chunks={'band': 1, 'x': 4, 'y': 4}, #finished in 71.865535.   RAM Used (GB): 23.60338432, storage (GB): 1.5712996535003185
                                    #chunks={'band': 1, 'x': 1024, 'y': 1024}, #finished in 2.65667 RAM Used (GB): 14.921428992, storage (GB): 1.5712996535003185
                                    #chunks=chunks_d, #seems to be faster w/o dask
                                    #nodata=0.0,
                                    masked=False,    
                                    ).rio.set_crs(f'EPSG:{epsg_id}', inplace=True) as xds:
 
        #prep
        xds = xds.where(xds != 0)  # set nulls
        
        #=======================================================================
        # check
        #=======================================================================
        skip=False
        if xds.isnull().all():
            log.error(f'got all nulls on \n    {coords}\n    {asc_fp}...skipping')
            skip=True
 
        if (xds<0.0).any():
            log.error(f'got some negatives on \n    {coords}\n    {asc_fp}...skipping')
            skip=True
        
 
        if not skip:
            #=======================================================================
            # add meta
            #=======================================================================
            xds.encoding.update(encoding)  # set encoding/compression    
     
            xds = xds.assign_coords(coords)
            
            xds.name = name  # set the data name (nice when we go to combine)
            
 
            #===========================================================================
            # write
            #===========================================================================
     
            log.debug(f'    writing to {ofp}')  # \n    /w: encoding={encoding}')
            
            xds.to_netcdf(ofp,
                          format='netCDF4',
                          engine='netcdf4',
                          mode='w',
                          #encoding = encoding,
                          #compute=True,
                           )
 
            #===========================================================================
            # wrap
            #===========================================================================
            meta_d = {
                'tdelta':(datetime.now()-start).total_seconds(),
                'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                'disk_GB':get_directory_size(temp_dir),
                'output_MB':os.path.getsize(ofp)/(1024**2)
                }
        else:
            meta_d = {}
    
    log.info(meta_d)
    
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log.debug(f'failed to remove temp_dir w/ \n    {e}')
    
    return None, meta_d




def _prep_row_calc(out_dir, log, i, row, basinName):
    d = row.to_dict()
    d['raster_index'] = i
    fp = d.pop('filepath')
#build the filepath for the cache
    uuid = hashlib.md5(f'{i}_{basinName}_{fp}_{d}'.encode("utf-8")).hexdigest()
    ofp_i = os.path.join(out_dir, f'{i:04d}_{uuid}.nc')
    log.debug(f'on {d} w/ \n    {ofp_i}')
    
    return ofp_i, fp, d





def _process_row(row, i, out_dir, variable, temp_dir, basinName):
    
    log = get_log_stream(name = f'{basinName}.{os.getpid()}')
 
    
    #log.info(f'on {i}/{len(lookup_df1)}')
    ofp_i, fp, d = _prep_row_calc(out_dir, log, i, row, basinName)
    
    # create the file
    if not os.path.exists(ofp_i):
        log.info(f'on {d}')
        ofp_i, meta_d = extract_asc_to_nrc2(fp, ofp=ofp_i,
                                                 temp_dir=os.path.join(temp_dir, str(i)),
                                                 coords=d, name=variable,
                                                 logger=log.getChild(str(i)))
 
    else:
        # pass
        log.info(f'    file exists.. skipping')
        
    #close the logger
    log.debug(f'finished')    
    #[h.close() for h in log.handlers]
    

 
    
#===============================================================================
# ELBE---------
#===============================================================================
def extract_asc_realiz_to_nrc2(zip_fp,
                       temp_dir=None, out_dir=None,
 
                       encoding=None,
                       #coords=dict(),
                       name='inundation_depth',
                       asc_prefix='wstmax',
                       logger=None,
                       unzip=True,
                       raster_indexer_multiplier=1e4,
                       basinName=None, cnt_max_rasters=None,
                       max_workers=None,
                       #chunks_d={'x':100, 'y':100, 'band':1}
                       ):
    """unzip asc and convert to nrc
        for results structured like realization/asc files
    
    Parameters
    ----------
    asc_prefix: str
        ascii file prefix to include
        
    raster_indexer_multiplier: int
        for adding indexers to each layer
        
 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now()
    #setup cache
    if temp_dir is None:
        temp_dir= get_temp_dir(temp_dir_sfx='py/temp/mosaic_from_lookup')
    if not os.path.exists(temp_dir):
        assert unzip
        os.makedirs(temp_dir)
        
    log = logger.getChild('extract')
    
    log.info(f'extracting and converting on {zip_fp} to \n    {temp_dir}')
    
    #set encoding
    if encoding is None:
        #encoding = {'zlib': True, 'complevel': 9} #0.36MB, 19.5 secs
        """could do more trials... not sure how least_significant_digit works"""
        encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2}
    
 
    
    #===========================================================================
    # #unzip
    #===========================================================================
    if unzip:
        with zipfile.ZipFile(zip_fp, mode='r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        log.info(f'    extracted to {temp_dir}')
    else:
        log.warning('unzip=False')
        
    #===========================================================================
    # get asci files
    #===========================================================================
    asc_fps_all = _filesearch_all(temp_dir)
    assert len(asc_fps_all)>0
    assert len(asc_fps_all)<raster_indexer_multiplier, 'may run into indexing problems'
    
    #prefix
    asc_fps_match = [e for e in asc_fps_all if os.path.basename(e).startswith(asc_prefix)]
    assert len(asc_fps_match)>0
    
    log.info(f'extracted {len(asc_fps_all)} asc files. {len(asc_fps_match)} match prefix \'{asc_prefix}\'')

    if not cnt_max_rasters is None:
        log.warning(f'cnt_max_rasters={cnt_max_rasters}')
        asc_fps_match = asc_fps_match[:cnt_max_rasters]
    
    #===========================================================================
    # loop and load
    #===========================================================================
    """could load them all with rioxarray?... simpler to go one by one"""
    meta_lib = dict()
    realisation = int(os.path.basename(zip_fp).replace('.zip', '').replace('M',''))
    
    
    #===========================================================================
    # build params
    #===========================================================================
    log.info(f'building parameters on {len(asc_fps_match)}-----------')
    param_lib = dict()
    for i, fp in enumerate(asc_fps_match):
        log.info(f'    {i+1}/{len(asc_fps_match)} building params from {fp}')     
        uuid = hashlib.md5(f'{i}_{basinName}_{fp}_{zip_fp}'.encode("utf-8")).hexdigest()
        ofp = os.path.join(out_dir, f'{i:04d}_{uuid}.nc')
        
        if os.path.exists(ofp):
            log.warning(f'    file exists... skipping: {ofp}')
            continue
        #=======================================================================
        # get coordinates
        #=======================================================================
        fn = os.path.basename(fp)
        if asc_prefix == 'wstmax':
 
            coords = {
                'day':int(fn.replace(asc_prefix + '_gpu', '').replace('.asc', '')), 
                'realisation':realisation, 
                'raster_index':int(realisation * raster_indexer_multiplier + i)}

        else:
            raise NotImplementedError(asc_prefix)        

        param_lib[i] = (fp, ofp, encoding, name, coords)
        
    #===========================================================================
    # execute loop
    #===========================================================================
    log.info(f'exeecuting w/ {len(param_lib)} and max_workers={max_workers}---------------\n\n')
    #===========================================================================
    # multi-thread
    #===========================================================================
    if not max_workers is None:
        """ProcessPoolExecutor is faster than Threading"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_write_asc_to_nrc, *args) for i, args in param_lib.items()]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    meta_lib[i] = future.result()
                except Exception as e:
                    log.error(f'{i} failed w/ \n    {e}')
                else:
                    log.info(f'{i} succeeded') 
 
    #===========================================================================
    # single-process
    #===========================================================================
    else:
        for i, args in param_lib.items():
            meta_lib[i], _ = _write_asc_to_nrc(*args)
        
            
 
 
    
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(meta_lib)}')
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'disk_GB':get_directory_size(temp_dir),
                    'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    log.info(meta_d)
    
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log.debug(f'failed to remove temp_dir w/ \n    {e}')
    
    return meta_lib

def _write_asc_to_nrc(fp, ofp, encoding, name, coords):
    
    log = get_log_stream(name = f'{os.getpid()}')
    
    with rioxarray.open_rasterio( #seems like it would be faster if this was false... but easier to apply encoding w/ it as true
        fp, parse_coordinates=True, masked=False).rio.set_crs(f'EPSG:{epsg_id}', inplace=True) as xds:
        #===================================================================
        # meta
        #===================================================================
 
        d = {k:getattr(xds, k) for k in ['shape', 'size']}
 
        #===================================================================
        # extract coords
        #===================================================================
        """building in the format of the other basins

            {'realisation': 48, 'day': 17197, 'raster_index': 115}
        
            """

        #===================================================================
        # #prep
        #===================================================================
        xds = xds.where(xds != 0) # set nulls
 
        #=======================================================================
        # check
        #=======================================================================
        skip = False
        if xds.isnull().all():
            log.error(f'got all nulls on \n    {coords}\n    {fp}...skipping')
            skip = True
            d['all_null'] = True
        if (xds < 0.0).any():
            log.error(f'got some negatives on \n    {coords}\n    {fp}...skipping')
            skip = True
            d['negatives'] = True
        if not skip:
 
            #=======================================================================
            # add meta
            #=======================================================================
            xds.encoding.update(encoding) # set encoding/compression
            xds = xds.assign_coords(coords)
            xds.name = name # set the data name (nice when we go to combine)
            #===========================================================================
            # write
            #===========================================================================
            
            
            log.info(f'    for {d} writing to {ofp}') # \n    /w: encoding={encoding}')
            xds.to_netcdf(ofp, 
                format='netCDF4', 
                engine='netcdf4', #not working for some reason...
                mode='w', 
            #encoding = encoding,
                compute=True)
            
        else:
            d['skip']=True
    #===========================================================================
    # wrap file loop
    #===========================================================================
    d.update({
            'fn':os.path.basename(fp),
            'coords':coords, 
            'name':name, 
            'filesize':os.path.getsize(ofp) / (1024 ** 2),
            'ofp':ofp})
    return d




#===============================================================================
# runner-------
#===============================================================================
def convert_basin_to_nrcs2(
        basinName='ems',
                   variable='inundation_depth',
 
                     out_dir=None,
                     temp_dir=None,
                     cnt_max_rasters=None,
                     max_workers=None,
                     asc_dir=None,
                     ):
    """convert the zipped asc to a netcdf. in parallel
    
    
    Parameters
    --------
    
    cnt_max_rasters: int
        maximum number of rasters to process (for debugging)
 
    """

    #===========================================================================
    # setup output
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(lib_dir, basinName, '01_extract2')
 
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = get_temp_dir(temp_dir_sfx='py/temp/convert_basin_to_nrcs/'+today_str)
 
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'starting on {basinName} w/ max_workers={max_workers}\n\n')
    #===========================================================================
    # #get location of zipped asc files
    #===========================================================================
    if asc_dir is None:
        asc_dir = asc_lib_d[basinName]
    assert os.path.exists(asc_dir), asc_dir
    
    log.info(f'on {basinName} at \n    {asc_dir}')
    cnt=0
    if basinName not in ['elbe']:
        #===========================================================================
        # lookup table
        #===========================================================================
        
        lookup_df = load_lookup_df(asc_dir)
     
        lookup_df1 = lookup_df[lookup_df['variable']==variable].drop(['variable', 'filename'], axis=1)
        
        if not cnt_max_rasters is None:
            log.warning(f'randomly selecting {cnt_max_rasters} rows for processing')
            lookup_df1 = lookup_df1.sample(n=cnt_max_rasters)
        
     
        #===========================================================================
        # plot
        #===========================================================================
        """plotting the realisations"""
        #plot_violin_realisations(lookup_df1, os.path.join(out_dir, f'violinplot_day_real_{basinName}.svg'))
     
            
        #===========================================================================
        # mosaic loop
        #===========================================================================
        
        res_d = dict()
        meta_lib= dict()
        log.info(f'converting {len(lookup_df1)} entries')
        #for i, si in enumerate(np.array_split(lookup_df1['filepath'], chunk_cnt)):
        
        #=======================================================================
        # single process
        #=======================================================================
        if max_workers is None:
    
            for i, row in lookup_df1.iterrows():
                log.info(f'on {i}/{len(lookup_df1)}')
                ofp_i, fp, d = _prep_row_calc(out_dir, log, i, row, basinName)              
                
                # create the file
                if not os.path.exists(ofp_i): 
                
                    ofp_i, meta_lib[i] = extract_asc_to_nrc2(fp, ofp=ofp_i,
                                                             temp_dir=os.path.join(temp_dir, str(i)),
                                                             coords=d, name=variable,
                                                             logger=log.getChild(str(i))
                                                             )
                    cnt += 1
                    
                else:
                    # pass
                    log.info(f'    file exists.. skipping')
                    
                # set
                res_d[i] = ofp_i
                
        #===========================================================================
        # parallel
        #===========================================================================
        else:
            log.info(f'executing in parallel w/ max_workers={max_workers}')
            args = (out_dir, variable, temp_dir, basinName)
     
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_row, row, i, *args) for i, row in lookup_df1.iterrows()]
        
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    
    #===========================================================================
    # Elbe-----------
    #===========================================================================
    else:
    
        log.info(f'elbe loader')
        
        #get realiszations
        zip_fps = [os.path.join(asc_dir, e) for e in os.listdir(asc_dir) if e.endswith('.zip')]
        log.info(f'got {len(zip_fps)}... extracting each')
        
 
            
        for i, fp in enumerate(zip_fps):
            log.info(f'loading {i+1}/{len(zip_fps)} from {os.path.basename(fp)}----------\n\n\n')
            
            extract_asc_realiz_to_nrc2(fp, 
                                        temp_dir=os.path.join(temp_dir, os.path.basename(fp).replace('.zip', '')),
                                        name=variable,
                                        logger=log.getChild(str(i)),
                                        basinName=basinName,
                                        out_dir=out_dir,
                                        cnt_max_rasters=cnt_max_rasters,
                                        max_workers=max_workers,
                                                         )
            cnt+=1
 
 
 
        
        
    
    #===========================================================================
    # wrap
    #===========================================================================
    #===========================================================================
    
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log.error(f'failed to remote temp_dir\n    {temp_dir}\n    {e}')
    
    print(f'finished {basinName} on {cnt}  to \n    {out_dir}')
    
    return out_dir




if __name__=="__main__":
    
    basinKey = 'elbe'
    convert_basin_to_nrcs2(basinKey, cnt_max_rasters=None, max_workers=7) #101.167107
    
    """much slower for ems and donau"""
    #===========================================================================
    # dask_kwargs = dict(
    #     threads_per_worker=16, n_workers=1,cnt_max_rasters=2, #145.60
    #     #threads_per_worker=4, n_workers=4,cnt_max_rasters=2, #147.431024 
    #     )
    # dask_profile_func(convert_basin_to_nrcs2, basinKey,**dask_kwargs)
    #===========================================================================
    
    
    
    
    
    print('finished')
    
                      
                      
                           














