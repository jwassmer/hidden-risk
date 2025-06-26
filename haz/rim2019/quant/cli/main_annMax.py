'''
Created on Jul. 30, 2023

@author: cefect


maybe not needed... rusn pretty fast
'''

import argparse
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='annual_max_fromSparse')
    parser.add_argument('--basinName', type=str, default='ems', help='name of the basin')
    #parser.add_argument('--data_var', type=str, default='inundation_depth', help='data_var to convert')
    #parser.add_argument('--out_dir', type=str, default=None, help='output directory')
    #parser.add_argument('--temp_dir', type=str, default=None, help='temporary directory')
    #parser.add_argument('--threads_per_worker', type=int, default=None, help='maximum number of rasters to process at once')
    parser.add_argument('--n_workers', type=int, default=14, help='maximum number of workers to use')
 
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
 
    from haz.rim2019.annMax_03 import annual_max_fromSparse, dask_profile_func, dask_threads_func
    
    dask_threads_func(annual_max_fromSparse, **vars(args))
    
    
    #dask_profile_func(ncdf_rechunk_tozarr, **vars(args))