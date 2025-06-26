'''
Created on Jul. 31, 2023

@author: cefect

computing quantiles from sparse annual maxima
'''

import os, warnings, psutil, math
from datetime import datetime

import numpy as np
import pandas as pd
 

from osgeo import gdal # Import gdal before rasterio
 
 
import xarray as xr
import rioxarray
 

import sparse

from haz.hp.dask import (
    view, get_temp_dir, today_str, init_log, dask_profile_func, dstr, dask_threads_func, dataArray_toraster
    )


 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, epsg_id, basin_chunk_d
    )

from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray


#@dask.delayed
def delay_quantile(coo_ar, quant_l):
    
    target_shape = (len(quant_l), coo_ar.shape[1], coo_ar.shape[2])
 
    
    #sparse array w/ some data
    if coo_ar.nnz > 0:
        
        ar = coo_ar.todense()
 
        quant_ar = np.quantile(ar, quant_l, axis=0, method='linear')
        
        res_coo_ar = sparse.COO.from_numpy(quant_ar, fill_value=coo_ar.fill_value)
        
 
    else:
        #create an empty sparse array
        res_coo_ar = sparse.COO([], shape=target_shape, cache=False, fill_value=coo_ar.fill_value)
        
    #check
    assert res_coo_ar.shape==target_shape
    
    return res_coo_ar

def _sparse_map_func(func, chunks_d, coo_ar, **kwargs):
    jobs_l=list()
    i0, j0 = 0, 0
    for i in range(chunks_d['y'], coo_ar.shape[1], chunks_d['y']):
        for j in range(chunks_d['x'], coo_ar.shape[2], chunks_d['x']):
            #log.debug(f'{i0}-{i}, {j0}-{j}')
            assert i > i0
            assert j > j0, f'bad j vals: {j}, {j0}'
            jobs_l.append(func(coo_ar[:, i0:i, j0:j], {'i':i, 'j':j}, **kwargs))
            j0 = j
        
        i0 = i
        j0=0
        
    return jobs_l

def q_sparse(qraw, total_cnt, avail_cnt):
    """get a quantile adjusted for some missing data (low values)
    
    see test_quantile.test_quantile_adjust() for validation
    """
    assert qraw<=1.0
    assert isinstance(avail_cnt, int)
    
    miss_cnt = total_cnt-avail_cnt
 
    
    assert (total_cnt-qraw*total_cnt)<=avail_cnt, f'requested percentile {qraw} is too low given the missing fraction {miss_cnt/total_cnt:.4f}'
    
 
    q_adj = 1+total_cnt*(qraw-1)/avail_cnt
    
    
    assert q_adj<=1.0, q_adj
    assert q_adj>=0.0, q_adj
    return q_adj
        
        

def quantile_annMax_sparse(basinName=None,
                      nc_fp=None,
                      quant_l_raw=[
                                0.995, 
                                 0.999,
                                 0.998,
                                 ],
                      xchunk='auto', 
 
 
                      #data_var='inundation_depth',
                      simulated_years=5000,
                      encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},
                      
                      out_dir=None,
                      #compressor=None,
                      ):
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, basinName, 'quant')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
    if nc_fp is None:
        nc_fp = get_sparse_fp(basinName, subdir='03_annMax')
 
    log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    #===========================================================================
    # load
    #===========================================================================
    log.info(f'loading sparse-xarray from \n    {nc_fp}')
    
    sds = load_sparse_xarray(nc_fp)
    sda = list(sds.data_vars.values())[0] #get the first (should be only) dataArray
    #===========================================================================
    # convert to sparse quantiles
    #===========================================================================
    """because we dont include non-event years in the data"""
 
    quant_d = {q:q_sparse(q, simulated_years, len(sds.event_year)) for q in quant_l_raw}
    
    log.info(f'converted quantiles for n={simulated_years}\n    raw={quant_l_raw}\n    sparse={list(quant_d.values())}')
 
    #===========================================================================
    # compute quantile-------
    #===========================================================================
    #===========================================================================
    # w/ dask's chunk' map_blocks
    #===========================================================================
 
    if xchunk=='auto':
        """~1million cells"""
        xchunk = int(math.sqrt(1e6//sda.shape[0]))
 
    #rechunk
    log.info(f'rechunking {sda.shape} by {xchunk}')
    sda1 = sda.chunk({'x':xchunk, 'y':xchunk, 'event_year':-1})
 
    #map_blocks
    log.info(f'map_blocks on {sda1.chunks}')     
    quant_coo_ar = sda1.data.map_blocks(delay_quantile, list(quant_d.values()),
                    meta=sparse.COO([], shape=(len(quant_d), sda1.data.shape[1], sda1.data.shape[2])),
                    ).compute() 
 
    
    log.info(f'concat sparse quantile to get {quant_coo_ar.shape} w/ {quant_coo_ar.nnz} reals and {quant_coo_ar.nbytes/(1024**2):.4f} MB')
    
    #===========================================================================
    # #check
    #===========================================================================
    for i, qc_ar in enumerate(quant_coo_ar): #iterate over resulting quantile
        q = list(quant_d.keys())[i]
        assert qc_ar.nnz>0.0, f'got zero max on qraw={q} qadj={quant_d[q]}'
    
    #===========================================================================
    # compute maximum-------
    #===========================================================================
    quant_d[1.0] = 1.0
    #coo_ar_max = sda1.max(axis=0)
    coo_ar_max = sda1.max(axis=0).compute().data.reshape([1, sda1.shape[1], sda1.shape[2]])
    
    sda1.close()
    #add this
    quant_coo_ar = sparse.concatenate([quant_coo_ar, coo_ar_max], axis=0)
    
    
    #===========================================================================
    # convert to xarray
    #===========================================================================
    res_da =xr.DataArray(quant_coo_ar.todense(), 
                 dims=['quantile', 'y', 'x'], 
                 coords={k:sda.coords[k] for k in ['y', 'x']}
                 ).assign_coords({'quantile':list(quant_d.keys())}).rio.write_crs(f'EPSG:{epsg_id}')
    
    
    assert np.array_equal(res_da.x.values, sda.x.values)
    assert np.array_equal(res_da.y.values, sda.y.values)
    #===========================================================================
    # write to netcdf
    #===========================================================================
    shape_str = '-'.join([str(e) for e in res_da.shape])
    
    #promote to dataset
    name = list(sds.data_vars.keys())[0]
    res_ds = res_da.to_dataset(name=name).assign_attrs(
        {'basinName':basinName, 'date':today_str, 'nc_fp':nc_fp, 'quant_d':str(quant_d), 'simulated_years':simulated_years})
    
    #res_ds.encoding.update(encoding)
    
    ofp_nc = os.path.join(out_dir, f'quant_{basinName}_q{len(quant_d)}_{shape_str}_{today_str}.nc')
    log.info(f'writing {res_da.shape} to \n    {ofp_nc}') 
    res_ds.to_netcdf(ofp_nc, mode ='w', format ='netcdf4', engine='netcdf4', compute=True,
                     encoding={name:encoding})
    
    #=======================================================================
    # write to_raster---------
    #=======================================================================
    log.info(f'writing each quantile for {res_da.shape}')
    for quant, _ in quant_d.items():
        #get the slice
        qdxa_i = res_da.sel(quantile=quant).squeeze().drop('quantile') 
        
        #set spatials
        qdxa_i = qdxa_i.rio.write_crs(f'EPSG:{epsg_id}').rio.write_nodata(-9999)
                  
        #write        
        ofp = os.path.join(out_dir, f'quant_aMax_{basinName}_{shape_str}_q{quant:.4f}_{today_str}'.replace('.', '') + '.tif') 
        log.info(f'queing write of {qdxa_i.shape} q={quant} to \n    {ofp}')
        
        #assign spatils and write
        """not giving me a delayed object for some reason"""
        qdxa_i.rio.to_raster(ofp, dtype='float32', compute=True, compress='LZW')
    
 
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now()-start).total_seconds(),
        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
        #'disk_GB':get_directory_size(temp_dir),
        'output_MB':os.path.getsize(ofp_nc)/(1024**2)
        }
    
    log.info(f'finished w/ \n {dstr(meta_d)} \n    {out_dir}')
    
    return out_dir
    #===============================================================================
    # xarray map_blocks
    #===============================================================================

    #===========================================================================
    # coords ={k:sda1.coords[k].values for k in ['y', 'x']}
    # coords['quantile'] = quant_l_raw
    # 
    # """template is strange
    # xr.map_blocks(delay_quantile, sda1, args=(list(quant_d.values())),
    #               template=xr.DataArray(sparse.COO([], shape=(len(quant_d), sda1.data.shape[1], sda1.data.shape[2])),
    #                                     dims=('quantile', 'y', 'x'), coords=coords)
    #               )"""
    #===========================================================================

    #===========================================================================
    # manual quantile loop on 1 dim
    #===========================================================================
 #==============================================================================
 #    #get xchunk
 #    if xchunk=='auto':
 #        """~1million cells"""
 #        xchunk = int(1e6//(coo_ar.shape[0]*coo_ar.shape[1]))
 #    
 #    
 #    chunk_cnt = math.ceil(coo_ar.shape[2]/xchunk)
 # 
 #     
 #    log.info(f'computing for {chunk_cnt} chunks w/ xchunk={xchunk}')
 #    jobs_l = list()
 # 
 #    #split x domain into chunks
 #    for i, xar_i in enumerate(np.array_split(np.arange(0, coo_ar.shape[2]), chunk_cnt)):
 # 
 #        log.info(f' {i}/{chunk_cnt} on {xar_i}')
 #        jobs_l.append(delay_quantile(coo_ar[:,:, xar_i], quant_l=list(quant_d.values())))
 #        
 # 
 #    assert xar_i[-1]+1==coo_ar.shape[2], f'loop terminated with bad xi={xar_i[-1]}, {coo_ar.shape[2]}'
 #    log.info(f'finished on {len(jobs_l)}')
 #     
 #    #collect    
 #    quant_coo_ar = sparse.concatenate(jobs_l, axis=2)
 #==============================================================================
    
    #===========================================================================
    # manual range loops
    #===========================================================================
    #===========================================================================
    # raise IOError('better to just work on 1 dimension (i.e., y).. makes collecting much easiser')
    # #check divisibility
    # """not sure how this works if dimensions are non-divisible"""
    # assert coo_ar.shape[1]%chunks_d['y']==0, f'non divisible axis'
    # assert coo_ar.shape[2]%chunks_d['x']==0, f'non divisible axis'
    # 
    # 
    # jobs_l= _sparse_map_func(delay_quantile, chunks_d, coo_ar, quant_l=quant_l)
    # 
    # result = dask.compute(jobs_l)
    # 
    # log.info(f'compute fniished in  {(datetime.now()-start).total_seconds():.2f} w/ %.2f GB ram'%(psutil.virtual_memory () [3]/1000000000))
    # 
    # #===========================================================================
    # # collect
    # #===========================================================================
    # ar_l, index_l = tuple(zip(*result[0]))
    # 
    # for k, row in pd.DataFrame(index_l).iterrows():
    #     ar = ar_l[k]
    #         #print(i,j)
    #     raise IOError('stopped here... need to collect into 2D')
    # return
    #===========================================================================
    #===========================================================================
    # stride_tricks
    #===========================================================================
    #===========================================================================
    # """only works with numpy"""
    # my_array = coo_ar
    # block_size = (60, 100, 100)
    # for i in range(0, my_array.shape[1], block_size[1]):
    #     for j in range(0, my_array.shape[2], block_size[2]):
    #         block = np.lib.stride_tricks.as_strided(my_array[:, i:i+block_size[1], j:j+block_size[2]], shape=block_size, strides=my_array.strides)
    #         print(block.shape)
    #===========================================================================
    
    
    #===========================================================================
    # block loop
    #===========================================================================
    #===========================================================================
    # """complicated.. need to make sure blocks are well rounded"""
    # xi_ar, yi_ar = np.arange(0,len(ds.x.values), 1), np.arange(0,len(ds.y.values), 1)
    # 
    # for xi in xi_ar:
    #     for yi in yi_ar:
    #         print(xi, yi)
    #===========================================================================
    
    #===========================================================================
    # dump into dask
    #===========================================================================
    #===========================================================================
    # """blows up memory"""
    # log.info(f'loading into dask')
    # dask_ar = dask.array.from_array(coo_ar, chunks=(1, 100, 100), 
    #                       asarray=False,
    #                       fancy=False,
    #                       inline_array =False,
    #                       )
    # 
    # log.info(f'loaded into dask in {(datetime.now()-start).total_seconds():.2f} as {dask_ar.nbytes/(1024**2):.4f} MB')
    #===========================================================================
    
    #===========================================================================
    # use dask to help with the chunking
    #===========================================================================
    """couldnt find a good function for this"""
#===============================================================================
#     block_size=(100, 100)
#     my_array = dask.array.empty(coo_ar.shape, dtype=float, chunks=(1, 100, 100))
#     indices = dask.array.indices(coo_ar.shape, dtype=int)
#     
#     indices = dask.array.indices(my_array.shape, dtype=int)
# 
#     for i in range(0, my_array.shape[0], block_size[0]):
#         for j in range(0, my_array.shape[1], block_size[1]):
#             block_indices = indices[:, i:i+block_size[0], j:j+block_size[1]]
#             print(block_indices.compute())
#===============================================================================
            
    
 
    
    #=======================================================================
    # simple concat nulls the builtin quantile
    #=======================================================================
    """blows up memory"""
    
    #=======================================================================
    # add missing
    #=======================================================================
    #===========================================================================
    # if not simulated_years is None:        
    #     years_missing = np.setdiff1d(np.arange(0,simulated_years, 1), ds.event_year.values)
    #     
    #     assert set(years_missing).intersection(ds.event_year.values)==set()
    #     
    #     log.info(f'got {len(ds.event_year)}/{simulated_years} years with events (missing {len(years_missing)})')
    #     
    #     #build the block of blanks
    #     blank_da = da.isel(event_year=0).squeeze().drop('event_year').where(False)
    #     
    #     assert blank_da.compute().isnull().all(), 'failed to get all blanks'
    #     
    #     """blows up the memory
    #     log.info(f'building da_missing')
    #     da_missing = blank_da.expand_dims({'event_year':years_missing}
    #                                       ).chunk(chunk_d).compute()
    #     """
    #     
    #     blank_da.expand_dims({'event_year':years_missing}
    #                                       ).chunk(chunk_d)
    #                                       
    # 
    # 
    #     da_complete = xr.concat([da, da_missing], dim='event_year')#.sortby('event_year')
    #     
    #     #da_complete = xr.merge([da, da_missing], compat ='broadcast_equals')#.sortby('event_year')
    #     
    # else:
    #     da_complete=da
    #===========================================================================
    
 #==============================================================================
 #    log.info(f'concated to get {da_complete.shape}')
 #    
 #    
 #    #=======================================================================
 #    # compute quantile
 #    #=======================================================================
 #    """not sure why these are different
 #    da_missing.chunks
 #    
 #      
 #    da_complete.dims    
 #    da_complete.chunks
 #    da_complete.data.chunks
 #    da_complete.encoding['chunks']
 #    """  
 #    #rechunk?
 #    """
 #    log.info(f'rechunking')
 #    qxda0 = da_complete.chunk({'event_year':-1}).compute()
 #    
 #    log.info(f'fillna')
 #    qxda1 = qxda0.fillna(0.0).compute()
 #    
 #    log.info(f'computing quantiles')
 #    qxda = qxda1.quantile(
 #        q=quant_l,dim='event_year', keep_attrs =False, skipna=False).compute()
 #        
 #    """
 #    log.info(f'computing rechunk, fillna, and quantile')
 #    #non-sparse
 #    #"""A.too-slow
 #    log.info(f'rechunk')
 #    #qxda1 = da_complete.chunk({'event_year':-1}).compute()
 #    qxda1=da_complete
 #    log.info(f'fillna and quantile')
 #    qxda = qxda1.fillna(0.0).quantile(
 #        q=quant_l,dim='event_year', keep_attrs=True, 
 #        skipna=False, #stalls if this is true?
 #        ).compute()
 #    
 #    
 #    """A.over-estimates
 #    qxda = da_complete.chunk({'event_year':-1}).quantile(
 #        q=quant_l,dim='event_year', keep_attrs=True, 
 #        skipna=True,  
 #        ).compute()"""
 #        
 #    """C.all nulls
 #    qxda = da_complete.chunk({'event_year':-1}).quantile(
 #        q=quant_l,dim='event_year', keep_attrs=True, 
 #        skipna=False, 
 #        ).compute()"""
 #        
 # 
 #    
 #    assert not qxda.isnull().all()
 #==============================================================================
      
if __name__=="__main__":
    
    #===========================================================================
    # basic---------
    #===========================================================================
    
    kwargs = dict( 
           basinName = 'ems',   #{'tdelta': 2.905838, 'RAM_GB': 11.840299008, 'output_MB': 0.3228721618652344} 
           #basinName = 'weser',   #{'tdelta': 16.457108, 'RAM_GB': 12.534173696, 'output_MB': 0.9339733123779297} 
        )
     
    quantile_annMax_sparse( **kwargs)
    
    
    #===========================================================================
    # threaded---------
    #===========================================================================
  #=============================================================================
  #   kwargs = dict( 
  #          #basinName = 'ems',  n_workers=12, #{'tdelta': 4.431748, 'RAM_GB': 11.609587712, 'output_MB': 0.3228721618652344} 
  #           #basinName = 'weser',  n_workers=14, #{'tdelta': 16.767357, 'RAM_GB': 12.510744576, 'output_MB': 0.9339733123779297} 
  #       )
  # 
  #   dask_threads_func(quantile_annMax_sparse,  **kwargs)
  #=============================================================================
 #==============================================================================
    
    
    
    # 
    # kwargs = dict( 
    #        simulated_years=5000,  n_workers=2,threads_per_worker=7
    #     )
    #===========================================================================
        
    #dask_profile_func(quantile_annMax_zarr, basinName=basinName,**kwargs)
                      
    print(f'finished')
                      
                      
                      
                      
                      
                      
                      
                      
                      
