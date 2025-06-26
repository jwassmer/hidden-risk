'''
Created on Jan. 10, 2024

@author: cef

compile rasters for each hyd basin
'''

import argparse
from haz.rim2019.downscale._01_dem import run_get_dem_dgm5_all

 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='haz.rim2019.downscale._01_dem')
    parser.add_argument('--out_dir', type=str,)
    parser.add_argument('--search_dir', type=str,default=None)
    parser.add_argument('--dem_index_fp', type=str,default=None)
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
    
    #set up optional kwargs
    skwargs=dict()
    if not args.search_dir is None:
        skwargs['build_vrt']=dict(search_dir=args.search_dir)
        
    if not args.dem_index_fp is None:
        skwargs['identify_dem_tiles']=dict(dem_index_fp=args.dem_index_fp)
        
    
    
    run_get_dem_dgm5_all(out_dir=parser.out_dir,
                         skwargs= skwargs
                         )
    
