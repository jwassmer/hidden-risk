'''
Created on Jul. 29, 2023

@author: cefect
'''


import argparse


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert_basin_to_nrcs2')
    parser.add_argument('--basinName', type=str, default='ems', help='name of the basin')
 
    parser.add_argument('--out_dir', type=str, default=None, help='output directory')
    parser.add_argument('--asc_dir', type=str, default=None, help='directory of asci files (None reads from definitions)')
 
    parser.add_argument('--max_workers', type=int, default=6, help='maximum number of workers to use')
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
    
    from haz.rim2019.extract_to_nc_01 import convert_basin_to_nrcs2
    
    convert_basin_to_nrcs2(**vars(args))
