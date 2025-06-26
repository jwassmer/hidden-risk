'''
Created on Jan. 10, 2024

@author: cef

event specific DEM
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

import geopandas as gpd

from haz.rim2019.coms import get_analysis_polygon

from definitions import test_data_dir as test_data_dir_master
test_data_dir = os.path.join(test_data_dir_master, 'test_downscale_dem')

 
#===============================================================================
# fixtures---------
#===============================================================================
#===============================================================================
# @pytest.fixture(scope='function')
# def bound_poly(basinName, raster_index):
#     bound_fn = {'ems':{
#             81:'bounds_ems_0000081_22463564263593a6.gpkg',
#             118:'bounds_ems_0000118_8d3de34e97bf0af3.gpkg',
#             1:'bounds_ems_0000001_009dd43f23b1391f.gpkg'
#             }}[basinName][raster_index]
#     
#     fp = os.path.join(test_data_dir, 'test_02_event_dem', bound_fn)
#     return gpd.read_file(fp).geometry[0]
#===============================================================================

@pytest.fixture(scope='function')
def wet_pts(basinName, raster_index):
    bound_fn = {'ems':{
            81:'wetPts_ems_0000081_22463564263593a6.gpkg',
            118:'wetPts_ems_0000118_8d3de34e97bf0af3.gpkg',
            1:'wetPts_ems_0000001_009dd43f23b1391f.gpkg'
            }}[basinName][raster_index]
    
    fp = os.path.join(test_data_dir, 'test_02_event_dem', bound_fn)
    return gpd.read_file(fp)


#===============================================================================
# shared parameters----
#===============================================================================
basin_raster_par= ('basinName, raster_index', [
    ('ems', 1), 
    ])
 
 
#===============================================================================
# tests---------
#===============================================================================
 

@pytest.mark.parametrize('basinName', ['ems'])
@pytest.mark.parametrize('raster_index', [1,118,81])
def test_01_event_wet_points(basinName, raster_index, tmpdir): 
    from haz.rim2019.downscale._02_dem import _01_event_wet_points as func
    
    func(basinName, raster_index, out_dir=tmpdir, use_cache=False)
    
    

@pytest.mark.parametrize(*basin_raster_par)
def test_identify_dem_tiles(wet_pts): 
    from haz.rim2019.downscale._02_dem import identify_dem_tiles
    
    identify_dem_tiles(wet_pts)
    
 
#===============================================================================
# @pytest.mark.parametrize('filename_l', 
#                          [['dgm5_4120_3300_20','dgm5_4120_3280_20']],
#                          )
# @pytest.mark.parametrize('search_dir', [os.path.join(test_data_dir, 'test_build_vrt')])
# def test_build_vrt(search_dir, filename_l, tmpdir): 
#     from haz.rim2019.downscale._02_dem import build_vrt
#     
#     build_vrt(filename_l, search_dir=search_dir, out_dir=tmpdir)
#===============================================================================
   
 
@pytest.mark.parametrize('vrt_fp', [os.path.join(test_data_dir, 'test_write_vrt_to_tif', 'DGM5_002_4d5ccb94f4079fd0.vrt')])
@pytest.mark.parametrize('creationOptions', [
    #['COMPRESS=LZW'], #{'tdelta': '19.30 secs', 'RAM_GB': 9.911947264, 'output_MB': 9.238180160522461}
    #[], #{'tdelta': '19.74 secs', 'RAM_GB': 11.0032896, 'output_MB': 1220.7040529251099}
    ['COMPRESS=LERC_DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6','MAX_Z_ERROR=.001'] #{'tdelta': '19.02 secs', 'RAM_GB': 9.9853312, 'output_MB': 1.9930086135864258}
    ]) 
def test_write_vrt_to_tif(vrt_fp, creationOptions, tmpdir):
    from haz.rim2019.downscale._02_dem import write_gdal_to_tif
    
    write_gdal_to_tif(vrt_fp,  creationOptions=creationOptions, out_dir=tmpdir, NUM_THREADS=4)
    

@pytest.mark.dev
@pytest.mark.parametrize(*basin_raster_par)
def test_02_event_dem(wet_pts, tmpdir): 
 
    from haz.rim2019.downscale._02_dem import _02_event_dem as func
    
    func(wet_pts, out_dir=tmpdir, use_cache=False)
    
 

 
    