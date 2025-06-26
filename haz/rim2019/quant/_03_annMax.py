'''
Created on Jul. 29, 2023

@author: cefect

reduce to annual maxima
 
'''
import os, warnings, psutil
from datetime import datetime

import numpy as np
import pandas as pd
import sparse
import dask
  
 
import xarray as xr
 
 

from haz.hp import (
    view, get_temp_dir, today_str, init_log, dask_profile_func, dstr, dask_threads_func, get_directory_size
    )
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, exclude_lib
    )

from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray
 

      
@dask.delayed
def _get_max(coo_ar, id_l):
    #get the slice
    coo_ar_i = coo_ar[id_l, :, :]
    
    assert coo_ar_i.nnz>0, f'for {id_l} got all zeros'
    
    if len(id_l)==1:
        return coo_ar_i
    else:        
    
        return coo_ar_i.max(axis=0).reshape((1, coo_ar_i.shape[1], coo_ar_i.shape[2]))
  

def annual_max_fromSparse(basinName=None,
                      nc_fp=None, 
                      out_dir=None,
 
 
                      #data_var='inundation_depth',
                      exclude_d=None,
 
                      ):
    """reduce to annual maxima
    
    Pars
    --------
 
    """
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(lib_dir, basinName, '03_annMax')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
    if nc_fp is None:
        nc_fp = get_sparse_fp(basinName)
        
    if exclude_d is None:
        if basinName in exclude_lib:
            exclude_d = exclude_lib[basinName]
 
    log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
 
    #===========================================================================
    # load------------
    #===========================================================================
    log.info(f'loading sparse-xarray from \n    {nc_fp}')
    
    sds = load_sparse_xarray(nc_fp)
    
    #=======================================================================
    # make exclusions-----------
    #=======================================================================
    if not exclude_d is None:
        log.warning(f'dropping {np.array([len(v) for k,v in exclude_d.items()]).sum()} values')
        
 
        sds = sds.swap_dims({'sparse_index':'raster_index'}).drop_sel(
            exclude_d, errors='ignore' #ignore because we  used to have this dropping on nc_to_sparse
            ).swap_dims({'raster_index':'sparse_index'})
            
        #reset sparse index
        sds.coords['sparse_index'] = np.arange(0, len(sds.sparse_index))
    
    #=======================================================================
    # prep
    #=======================================================================
 
    #add a simulation day
    real_day = (sds.realisation.values-1)*(365*100) #each realisation was for 100 years
    event_day_ar = real_day + sds.day.values #add teh actual event day
    
    assert len(event_day_ar)==len(sds.coords['raster_index'])        
    
    sda1 = sds.assign_coords(event_day=('sparse_index', event_day_ar)) #add a event_day along the 'sparse_index' dimension
    
    #add an event_year
    event_year_ar = event_day_ar//365
    sda1 = sda1.assign_coords(event_year=('sparse_index', event_year_ar)) 
    
    #===========================================================================
    # get new index
    #===========================================================================
 
    ds_empty = xr.Dataset(coords=sda1.coords)    
    dsE_max = ds_empty.groupby('event_year').first() #cant do this with sparse array
 
                            
    years_w_events = dsE_max.coords['event_year'].values
     
    assert len(years_w_events)==len(set(event_year_ar)), f'event year mismatch' 
    assert set(years_w_events).difference(event_year_ar)==set()     
     
    log.info(f' reduced from {len(sds.sparse_index)} sparse_index to {len(dsE_max.event_year)} event_years')
    
    #=======================================================================
    # calc annual max on sparse
    #=======================================================================
    #get the indexer
    #ri_da = da_amax.sparse_index
    ri_da = sda1.sparse_index.copy().swap_dims({'sparse_index':'event_year'})
    coo_ar = sda1['inundation_depth'].data
    
    log.info(f'computing the annual maxima on {len(np.unique(sda1.event_year))} years for {str(coo_ar.shape)}')
    
    # dok_ar = sparse.DOK.from_coo(coo_ar)
 
    jobs_l = list()
    for gval, _ in sda1.groupby('event_year'):
        # get the sparse_index values for this
        id_l = ri_da.sel(event_year=gval).values.tolist()
        if isinstance(id_l, int):
            id_l = [id_l]
            
        log.info(f'    queing on {gval} got {id_l}')        
        jobs_l.append(_get_max(coo_ar, id_l))
 
    # execute
    log.info(f'dask.compute on {len(jobs_l)}')
    result = dask.compute(jobs_l)
    
    #collect
    res_coo_ar = sparse.concatenate(result[0], axis=0)
    log.info(f'finished in {(datetime.now()-start).total_seconds():.4f} secs w/ {res_coo_ar.shape}')
    
    
    #===========================================================================
    # write
    #===========================================================================
 
    write_sparse_xarray(dsE_max, res_coo_ar, os.path.join(out_dir, f'{basinName}_annMax'), 
                        log=log, sparse_datavar='inundation_depth_annMax')
 
 
    
        
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now()-start).total_seconds(),
        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
        'disk_GB':get_directory_size(out_dir),
        #'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    
    log.info(f'finished w/ {dstr(meta_d)} to \n    {out_dir}')
    sds.close()
        
    return out_dir
        
 
      
if __name__=="__main__":
    
    basinName = 'rhine'
    
    kwargs = dict( 
           #====================================================================
           # count_file_limiter=None,
           #====================================================================
           
        )
    
    annual_max_fromSparse(basinName=basinName)
 
    #dask_threads_func(annual_max_fromSparse, basinName=basinName, n_workers=14, **kwargs)
    # 
    # kwargs = dict( 
    #     threads_per_worker=14, n_workers=1 
    #     )
    #===========================================================================
 
    #dask_profile_func(annual_max_tozarr, basinName=basinName, **kwargs)
    
 
