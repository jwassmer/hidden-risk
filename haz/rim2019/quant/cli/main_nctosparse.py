'''
Created on Jul. 30, 2023

@author: cefect
'''

import argparse
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert_basin_to_nrcs2')
    parser.add_argument('--basinName', type=str, default='ems', help='name of the basin')
    #parser.add_argument('--data_var', type=str, default='inundation_depth', help='data_var to convert')
    #parser.add_argument('--out_dir', type=str, default=None, help='output directory')
    #parser.add_argument('--temp_dir', type=str, default=None, help='temporary directory')
    parser.add_argument('--threads_per_worker', type=int, default=None, help='maximum number of rasters to process at once')
    parser.add_argument('--n_workers', type=int, default=None, help='maximum number of workers to use')
    
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
 
    from haz.rim2019.nc_to_sparse_02 import dask_threads_func, ncdf_tosparse, dask_profile_func
    
    dask_profile_func(ncdf_tosparse, **vars(args))
    
    
    #dask_profile_func(ncdf_rechunk_tozarr, **vars(args))