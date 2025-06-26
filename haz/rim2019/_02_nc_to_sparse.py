'''
Created on Jul. 31, 2023

@author: cefect

converting exported netcdf files to a sparse-xarray hybrid
'''
#===============================================================================
# IMPORTS----------
#===============================================================================
import os, hashlib, logging, shutil, psutil
from datetime import datetime

import numpy as np

import dask
from dask.diagnostics import ProgressBar

from osgeo import gdal # Import gdal before rasterio
import xarray as xr

 
import sparse


import definitions

from haz.hp.dask import (
    view, get_temp_dir, today_str, init_log, dask_profile_func, dask_threads_func, dstr, get_fp
    )
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, epsg_id, basin_chunk_d, exclude_lib
    )

def get_sparse_fp(basinName, subdir='02_sparse2'):
    srch_dir = os.path.join(lib_dir, basinName, subdir)

    return get_fp(srch_dir, ext='.nc')
    

 
def ncdf_tosparse(basinName=None,                      
                      count_file_limiter=None,
                      out_dir=None,
                      #temp_dir=None,
                      #chunks_d=None, #raster_index must be -1
                      #encoding=encoding_d,
                      data_var='inundation_depth',
                      #exclude_d=None,
 
 
 
                      ):
    """computations were too slow... trying to rebuild the dataset w/ better chunking
    
    using rechunker tool and zarr now
    
    Parameterse
    ---------
    exclude_d: dict
        indexers of simulations to exclude
    """
    
    #===========================================================================
    # setup dirs
    #===========================================================================
    start = datetime.now()
    nrc_dir = os.path.join(lib_dir, basinName, '01_extract2')
    
    assert os.path.exists(nrc_dir), nrc_dir    
    
    if out_dir is None:
        out_dir=os.path.join(lib_dir, basinName, '02_sparse2')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    #===========================================================================
    # 
    # if exclude_d is None:
    #     if basinName in exclude_lib:
    #         exclude_d = exclude_lib[basinName]
    #===========================================================================
    #===========================================================================
    # logger
    #===========================================================================
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name=basinName)
    log.info(f'starting on {basinName} from {nrc_dir}\n\n')
    

    #===========================================================================
    # identify files
    #===========================================================================
    fp_l_raw = [os.path.join(nrc_dir, e) for e in os.listdir(nrc_dir) if e.endswith('.nc')]
    log.info(f'found {len(fp_l_raw)} matching files')
    
    if not count_file_limiter is None:
        log.warning(f'only taking first {count_file_limiter}')
        fp_l = fp_l_raw[:count_file_limiter]
    else:
        fp_l = fp_l_raw
        
    #===========================================================================
    # loop and load
    #===========================================================================
    log.info(f'loading {len(fp_l)} from {nrc_dir}')
 
    #===========================================================================
    # load-----------
    #===========================================================================
    
    log.info(f'open_mfdataset on {len(fp_l)} from {nrc_dir}  ')
    
    """not working... need to merge raster_index also"""
    with xr.open_mfdataset(fp_l, 
                           parallel=False, 
                           #combine='nested',
                            combine='nested', 
                            concat_dim =['raster_index'], #load a bunch of files and combine
                            #chunks=chunks_d, #doesn't seem to be working
                            #chunks=chunks_d,
                            #compat='override', 
                            #coords=['x', 'y', 'raster_index', 'band'],
                           
                            ).squeeze().drop(['band', 'spatial_ref'], errors='ignore') as ds:
 
        log.info(f'loaded {ds.dims}' + 
                     f'\n    coors: {list(ds.coords)}' + 
                     f'\n    data_vars: {list(ds.data_vars)}' + 
                     f'\n    crs:{ds.rio.crs}'
                     f'\n    chunks:{ds.chunks}'
                     )
        
        #prep
 
        #pull out just the data
        #da = 
        
        #===========================================================================
        # chunks
        #===========================================================================
        """better to just use some default chunks as we are converting to sparse
        
        should be 1 chunk per raster
            not the most optimal.. but should be fine for the conversion
        """
                
        #=======================================================================
        # convert to sparse
        #=======================================================================
 
        log.info(f'dumping {ds[data_var].shape} to sparse')
        coo_ar = ds[data_var].fillna(0.0).data.map_blocks(sparse.COO).compute()
        
        #check
        for i, coo_ar_i in enumerate(coo_ar):
            assert coo_ar_i.nnz>0, f'got all zeros on {i}'
        
        log.info(f'built sparse in {(datetime.now()-start).total_seconds():.2f} secs w/ {coo_ar.nbytes/(1024**2):.4f} MB')
        
        # reset data
        ds[data_var].data = coo_ar
 
        #=======================================================================
        # make exclusions-----------
        #=======================================================================
        """moved to annMax_03
        makes manipulating the exclusions much faster... don't have to re-compile
        if not exclude_d is None:
            log.warning(f'dropping {np.array([len(v) for k,v in exclude_d.items()]).sum()} values')
            
 
            ds = ds.drop_sel(exclude_d)"""
 
            
        #=======================================================================
        # add layer index
        #=======================================================================
        """raster index comes from the lookup table.
        for slicing on teh sparse index, we need a clean index"""
        ds1= ds.assign_coords(sparse_index=('raster_index',np.arange(0,len(ds.raster_index), 1))
                          ).swap_dims({'raster_index':'sparse_index'}) 

                          
        #=======================================================================
        # write
        #=======================================================================
        ds1.attrs['sparse_datavar'] = data_var
        dims_str ='-'.join([str(abs(v)) for v in coo_ar.shape])
 
        ofp = write_sparse_xarray2(ds1, os.path.join(out_dir, f'{basinName}_toSparse_{dims_str}'), log=log)
        #=======================================================================
        # write metadata
 #==============================================================================
 #        #=======================================================================
 #        dims_str ='-'.join([str(abs(v)) for v in da.shape])
 #        ofn = f'{basinName}_rechunk6_{dims_str}'
 #        ofn1 =ofn+'.npz'
 #        
 #        #build an empty version of the datasource
 #        empty_ds = xr.Dataset(coords=ds1.coords, attrs=ds1.attrs
 #                              ).assign_attrs({'sparse_filename':ofn1, 'sparse_datavar':data_var})
 # 
 #        
 #        #write it
 #        
 #        ofp_meta = os.path.join(out_dir, ofn+'.nc')
 #        log.info(f'writing empty datSource to \n    {ofp_meta}\n    {ds1.coords}')
 #        empty_ds.to_netcdf(ofp_meta, mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
 #==============================================================================
 
        
    #=======================================================================
    # write
    #=======================================================================
    
    #===========================================================================
    # ofp = os.path.join(out_dir, ofn1)
    # 
    # log.info(f'writing to \n    {ofp}')
    # sparse.save_npz(ofp, sparse_da, compressed=True)
    #===========================================================================
 
        
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now()-start).total_seconds(),
        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
        #'disk_GB':get_directory_size(temp_dir),
        'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    
    log.info(f'finished w/ \n {dstr(meta_d)} \n    {out_dir}')
    return ofp

def write_sparse_xarray2(
        ds_sparse, ofp, log=None,sparse_datavar=None,sparse_index=None
        ):
    """
    write an xarray Dataset with sparse data as two files
    
    (some data, y, x)
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if log is None: log=logging.getLogger('write') 
        
    #ofp = ofp + '_' + '-'.join([str(abs(v)) for v in coo_ar.shape])
    
    if sparse_datavar is None: sparse_datavar=ds_sparse.attrs['sparse_datavar']
    
    #get the shape of the data source
    dm = ds_sparse.squeeze().dims
   
    
    if sparse_index is None:
        sparse_index = list(set(dm.keys()).difference(['x', 'y']))[0]
    
 

    #dshape = [dm[sparse_index], dm['y'], dm['x']]
    
    #===========================================================================
    # check
    #===========================================================================
    assert np.array_equal(
            np.arange(0, len(ds_sparse.coords[sparse_index])),
            ds_sparse.coords[sparse_index].values), f'got discontinous sparse index'
 
    
    #===========================================================================
    # write sparse
    #===========================================================================
    ofp1 = ofp+'.npz'
    
    sparse.save_npz(ofp1, ds_sparse[sparse_datavar].data, compressed=True)
    
    #===========================================================================
    # write xarray
    #===========================================================================
    ds_empty = ds_sparse.drop(sparse_datavar) 
 
    ds_empty.assign_attrs({'sparse_filename':ofp1, 'sparse_datavar':sparse_datavar, 'sparse_index':sparse_index}
                    ).to_netcdf(ofp+'.nc', mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
    
    log.info(f'wrote { ds_sparse[sparse_datavar].shape} to \n    {os.path.dirname(ofp)}')
    
    return ofp1

def write_sparse_xarray(
        ds, coo_ar, ofp, log=None,sparse_datavar=None,sparse_index=None
        ):
    """
    write an xarray Data Source but with sparse data
    
    (some data, y, x)
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if log is None: log=logging.getLogger('write') 
        
    ofp = ofp + '_' + '-'.join([str(abs(v)) for v in coo_ar.shape])
    
    if sparse_datavar is None: sparse_datavar=ds.attrs['sparse_datavar']
    
    #get the shape of the data source
    dm = ds.squeeze().dims
   
    
    if sparse_index is None:
        sparse_index = list(set(dm.keys()).difference(['x', 'y']))[0]
    
    #===========================================================================
    # check
    #===========================================================================\

    dshape = [dm[sparse_index], dm['y'], dm['x']]
    
    assert dshape==list(coo_ar.shape), f'shape mismatch'
    
    assert len(ds.data_vars)==0, 'expected no data'
    
    #===========================================================================
    # write sparse
    #===========================================================================
    ofp1 = ofp+'.npz'
    sparse.save_npz(ofp1, coo_ar, compressed=True)
    
    #===========================================================================
    # write xarray
    #===========================================================================
 
    ds.assign_attrs({'sparse_filename':ofp1, 'sparse_datavar':sparse_datavar, 'sparse_index':sparse_index}
                    ).to_netcdf(ofp+'.nc', mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
    
    log.info(f'wrote {coo_ar.shape} to \n    {os.path.dirname(ofp)}')
    
    return ofp

def load_sparse_xarray(
        nc_fp,
        sparse_index=None,
        fix_relative=True
        ):
    """load and re-assemble sparse-xarray data"""
 
 
    
    #===========================================================================
    # load the dataset
    #===========================================================================
    print(f'loading DataSet from \n    {nc_fp}')
    ds =  xr.open_dataset(nc_fp)  
    
    print(f'loaded {ds.dims}' + 
                     f'\n    coors: {list(ds.coords)}' 
                     )
        
 
    dm = ds.squeeze().dims
    if sparse_index is None:
        sparse_index = list(set(dm.keys()).difference(['x', 'y']))[0]
    #=======================================================================
    # load the sparze
    #=======================================================================
    if fix_relative:
        sparse_fp = os.path.join(os.path.dirname(nc_fp), os.path.basename(ds.attrs['sparse_filename']))
    else:
        sparse_fp = os.path.join(os.path.dirname(nc_fp), ds.attrs['sparse_filename'])
    assert os.path.exists(sparse_fp), f'no sparse file found \n    {sparse_fp}'
    
    print(f'sparse.load_npz from \n    {sparse_fp}')
    sparse_da = sparse.load_npz(sparse_fp)
    
    print(f'loaded {sparse_da.shape} w/ {sparse_da.nbytes/(1024**2):.4f} MB')
    
    #===========================================================================
    # merge
    #===========================================================================
    sdims = [sparse_index, 'y', 'x']
    #dims_l = [ds.dims[k] for k in sdims]
    
    sda = xr.DataArray(sparse_da, dims=sdims, coords=ds.coords)
    
    #promote to dataset
    sds = sda.to_dataset(name=ds.attrs['sparse_datavar']).rio.write_crs(epsg_id)
    
 
    #===========================================================================
    # check
    #===========================================================================
    assert np.array_equal(
            np.arange(0, len(ds.coords[sparse_index])),
            ds.coords[sparse_index].values), f'got discontinous sparse index'
    
    
    print(f'built sparse DataArray w/ {sda.shape} ')
 
    #ds.close()
    return sds #this is crashing my debugger :(
    
#===============================================================================
# def load_sparse_xarray2(
#         nc_fp,
#         sparse_index=None,
#         ):
#     """load and re-assemble sparse-xarray data"""
#  
#  
#     
#     #===========================================================================
#     # load the dataset
#     #===========================================================================
#     print(f'loading DataSet from \n    {nc_fp}')
#     ds =  xr.open_dataset(nc_fp)  
#     
#     print(f'loaded {ds.dims}' + 
#                      f'\n    coors: {list(ds.coords)}' 
#                      )
#         
#     dm = ds.squeeze().dims
#     if sparse_index is None:
#         sparse_index = list(set(dm.keys()).difference(['x', 'y']))[0]
#     #=======================================================================
#     # load the sparze
#     #=======================================================================
#     sparse_fp = os.path.join(os.path.dirname(nc_fp), ds.attrs['sparse_filename'])
#     assert os.path.exists(sparse_fp), f'no sparse file found \n    {sparse_fp}'
#     
#     print(f'sparse.load_npz from \n    {sparse_fp}')
#     sparse_da = sparse.load_npz(sparse_fp)
#     
#     print(f'loaded {sparse_da.shape} w/ {sparse_da.nbytes/(1024**2):.4f} MB')
#     
#     #===========================================================================
#     # merge
#     #===========================================================================
#     sdims = [sparse_index, 'y', 'x']
#     #dims_l = [ds.dims[k] for k in sdims]
#     
#     sda = xr.DataArray(sparse.COO([], shape=sparse_da.shape), dims=sdims, coords=ds.coords)
#     
#     print(f'built sparse DataArray w/ {sda.shape} ')
#  
#     #ds.close()
#     return sda,  sparse_da#this is crashing my debugger :(
#  
#  
#===============================================================================
    

if __name__=="__main__":
    basinName='donau'
    #===========================================================================
    # ncdf_tosparse(basinName=basinName, count_file_limiter=None,
    #               exclude_d={'raster_index':[3,4,7]}, #for dev
    #               )
    #===========================================================================
 
    #dask_threads_func(ncdf_tosparse, basinName=basinName, count_file_limiter=None)
     
    kwargs = dict(
        basinName=basinName, threads_per_worker=15, n_workers=1 #total_time=129.11 secs, max_mem=2572.58 MB, max_cpu=362.5 %
        )
  
    dask_profile_func(ncdf_tosparse, **kwargs)
    
    
    """test reloading"""
    #load_sparse_xarray(r'l:\10_IO\2307_roads\lib\rim2019\ems\02_rechunk6\ems_rechunk6_60-2600-1400.nc')
    
    
    
    
    
    
    
    
    
    
    
    
