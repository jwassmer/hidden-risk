'''
Created on Jan. 10, 2024

@author: cef
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd



from haz.rim2019.coms import get_analysis_polygon

from definitions import test_data_dir as test_data_dir_master
test_data_dir = os.path.join(test_data_dir_master, 'test_downscale_dem')

#small analysis basins for testing
index_fp = os.path.join(test_data_dir_master, 'aoiT01_20240110.gpkg')
#===============================================================================
# fixtures---------
#===============================================================================
@pytest.fixture(scope='function')
def aoi_polygon(basinName):
    return get_analysis_polygon(basinName, index_fp=index_fp)

 
#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.parametrize('basinName', ['elbe_lower', 'ems'])                            
def test_select_basin(basinName):
    
    result = get_analysis_polygon(basinName)
    
    
 
@pytest.mark.parametrize('basinName', ['ems_test'])
def test_identify_dem_tiles(aoi_polygon): 
    from haz.rim2019.downscale._01_dem import identify_dem_tiles
    
    identify_dem_tiles(aoi_polygon)
    
 


@pytest.mark.parametrize('filename_l', 
                         [['dgm5_4120_3300_20','dgm5_4120_3280_20']],
                         )
@pytest.mark.parametrize('search_dir', [os.path.join(test_data_dir, 'test_build_vrt')])
def test_build_vrt(search_dir, filename_l, tmpdir): 
    from haz.rim2019.downscale._01_dem import build_vrt
    
    build_vrt(filename_l, search_dir=search_dir, out_dir=tmpdir)
   

@pytest.mark.parametrize('vrt_fp', [os.path.join(test_data_dir, 'test_write_vrt_to_tif', 'DGM5_002_4d5ccb94f4079fd0.vrt')])
@pytest.mark.parametrize('creationOptions', [
    #['COMPRESS=LZW'], #{'tdelta': '19.30 secs', 'RAM_GB': 9.911947264, 'output_MB': 9.238180160522461}
    #[], #{'tdelta': '19.74 secs', 'RAM_GB': 11.0032896, 'output_MB': 1220.7040529251099}
    ['COMPRESS=LERC_DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6','MAX_Z_ERROR=.001'] #{'tdelta': '19.02 secs', 'RAM_GB': 9.9853312, 'output_MB': 1.9930086135864258}
    ])
def test_write_vrt_to_tif(vrt_fp, creationOptions, tmpdir):
    from haz.rim2019.downscale._01_dem import write_vrt_to_tif
    
    write_vrt_to_tif(vrt_fp, creationOptions=creationOptions, out_dir=tmpdir, NUM_THREADS=4)
    
    
 
@pytest.mark.parametrize('basinName', ['ems_test'])
@pytest.mark.parametrize('search_dir', [os.path.join(test_data_dir, 'test_build_vrt')])
def test_run_get_dem_dgm5(aoi_polygon, search_dir, tmpdir):
    from haz.rim2019.downscale._01_dem import run_get_dem_dgm5
    
    run_get_dem_dgm5(aoi_polygon, out_dir=tmpdir, use_cache=False,
                     skwargs={
                         'build_vrt':dict(search_dir=search_dir)
                         })
    

@pytest.mark.dev
@pytest.mark.parametrize('search_dir', [
    os.path.join(test_data_dir, 'test_build_vrt'),
    #None,
    ])
def test_run_get_dem_dgm5_all(search_dir, tmpdir):
    from haz.rim2019.downscale._01_dem import run_get_dem_dgm5_all
    
    run_get_dem_dgm5_all(
        index_fp=index_fp,
        debug_cnt=None,
        out_dir=tmpdir, use_cache=False,
                     skwargs={
                         'build_vrt':dict(search_dir=search_dir),
                         'identify_dem_tiles':dict(dem_index_fp=os.path.join(test_data_dir, 'dgm5_kacheln_20x20km_laea_4647_test.gpkg'))
                         })
    
    
    