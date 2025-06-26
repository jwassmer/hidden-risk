'''
Created on Oct. 10, 2023

@author: cefect

rim2019 parameteres
'''
import os
from definitions import wrk_dir, src_dir


#===============================================================================
# #base directories-------
#===============================================================================
cache_base_dir = os.path.join(wrk_dir, r'cache\rim_2019')

out_base_dir = os.path.join(wrk_dir, r'outs\rim_2019')

lib_dir = os.path.join(wrk_dir, r'lib\rim2019')

tmp_dir = os.path.join(wrk_dir, 'temp')


#===============================================================================
# project parmaeters--
#===============================================================================
epsg_id = 4647 # spatial (RFM 2019)

#RIM 2019 model DEMS
"""nodata =-9999, burned channel = 9999"""
coarse_dem_nulls = [9999, -9999]

#===============================================================================
# run variables
#===============================================================================



def get_chunk(xdim):
    return {'x': xdim, 'y': xdim,'band':-1, 'raster_index':-1}

basin_chunk_d = {
    'donau':get_chunk(100),  
    'weser':get_chunk(400),
    'rhine':get_chunk(50),
    'ems':get_chunk(700),
    }

index_d = {'03_annMax':'event_year', '02_sparse2':'raster_index'}

""" simulation files to exclude
do not filter by sparse_index as this changes based on teh shape
"""
exclude_lib = {
    #from those rastsers with max=99 (infinite ponding)
    'donau':{
        #'sparse_index':[961, 1009, 1347, 552, 1040, 291, 812, 779, 415, 1768, 2075, 2346]
        'raster_index':[2040, 2149, 2855, 1172, 2229, 570, 1709, 1674, 857, 3747, 4407, 5000]
        },
    'rhine':{
        'raster_index':[7619, 4624, 8844, 8142, 9118, 12374] #08599 
        }
    }

basinName2_d = {
    'elbe_lower':'elbe',
    'elbe_upper':'elbe',
    'rhine_lower':'rhine',
    'rhine_upper':'rhine',
    'ems':'ems','donau':'donau','weser':'weser'}

#===============================================================================
# downscaling------
#===============================================================================
#threshold for considering a cell dry (and ignoring it)
wet_wsh_thresh = 0.1

blocksize=800

#===============================================================================
# hi-res DEM
#===============================================================================
from definitions import coarse_dem_fp_d