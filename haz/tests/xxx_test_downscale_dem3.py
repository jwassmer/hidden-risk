'''
Created on Jan. 10, 2024

@author: cef

tests for prep fine DEM per hydro-basin
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
 

@pytest.fixture(scope='function')
def aoi_poly(basinName2):
    fn = {'ems':'aoi_poly_ems.pkl'}[basinName2] 
    
    fp = os.path.join(test_data_dir, 'test_warp_from_poly', fn)
    
    with open(fp, 'rb') as f:
        return pickle.load(f)
 


#===============================================================================
# shared parameters----
#===============================================================================
 
 
 
#===============================================================================
# tests---------
#===============================================================================
 
@pytest.mark.dev
@pytest.mark.parametrize('basinName2', ['ems'])
@pytest.mark.parametrize('dem_index_fp', [
    os.path.join(test_data_dir, 'dgm5_kacheln_20x20km_laea_4647_test.gpkg')
    ])
def test_run_dem_fine_to_DataSet(basinName2, dem_index_fp, tmpdir): 
    from haz.rim2019.downscale._02_dem import run_dem_fine_to_DataSet as func
    func(basinName2, dem_index_fp=dem_index_fp, use_cache=False, out_dir=tmpdir)
    
    

@pytest.mark.parametrize('filename_l', 
                         [['dgm5_4120_3300_20','dgm5_4120_3280_20']],
                         )
@pytest.mark.parametrize('search_dir', [os.path.join(test_data_dir, 'test_build_vrt')])
def test_build_vrt(search_dir, filename_l, tmpdir): 
    from haz.rim2019.downscale._02_dem import build_vrt
     
    build_vrt(filename_l, search_dir=search_dir, out_dir=tmpdir, use_cache=False)
    
    


@pytest.mark.parametrize('vrt_fp', [
    os.path.join(test_data_dir, 'test_warp_from_poly', 'DGM5_004_beb4658826a0e4d3.vrt')
    ])
def test_vrt_to_disc(vrt_fp,tmpdir): 
    from haz.rim2019.downscale._02_dem import vrt_to_disc as func
     
    func(vrt_fp, out_dir=tmpdir, use_cache=False)
    

@pytest.mark.parametrize('fp', [
    os.path.join(test_data_dir, 'test_warp_from_poly', 'DGM5_004_beb4658826a0e4d3.tif') #{'tdelta': '275.08 secs', 'RAM_GB': 19.328847872, 'output_MB': 85.91256332397461}
    ])
@pytest.mark.parametrize('basinName2', ['ems'])
def test_warp_from_poly(fp, aoi_poly, tmpdir): 
    from haz.rim2019.downscale._02_dem import warp_from_poly as func
     
    func(fp, aoi_poly, out_dir=tmpdir, use_cache=False)

 
    