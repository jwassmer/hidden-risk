'''
Created on Jan. 13, 2024

@author: cef
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd


from definitions import test_data_dir as test_data_dir_master
test_data_dir = os.path.join(test_data_dir_master, 'test_downscale_fines')

#from haz.rim2019.parameters import coarse_dem_fp_d

#===============================================================================
# fixtures---------
#===============================================================================
@pytest.fixture(scope='function')
def wse_ds_dir(basinName2):    
    return os.path.join(test_data_dir, 'chunk_to_dem_tile', basinName2)
    

@pytest.fixture(scope='function')
def wse_da(basinName2, raster_index):
    bound_fn = {'ems':{
            1:'wse_001.pkl', 
            }}[basinName2][raster_index]            
    
    fp = os.path.join(test_data_dir, 'test_apply_wse_downscale', bound_fn)
    
    with open(fp, 'rb') as f:
        return pickle.load(f)
    

@pytest.fixture(scope='function')
def da_clip():
    with open(r'l:\10_IO\2307_roads\test\test_downscale_fines\build_dem_wse_fine_datasource\da_clip.pkl', 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope='function')
def dem_index_fp():
    return os.path.join(test_data_dir, 'dgm5_kacheln_20x20km_laea_4647_test.gpkg')

 
#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('basinName2', ['ems'])

def test_run_chunk_to_dem_tile(basinName2, wse_ds_dir,dem_index_fp,  tmpdir):
    from haz.rim2019.downscale._02_fines import run_chunk_to_dem_tile as func
    
    func(basinName2, wse_ds_dir=wse_ds_dir, 
         dem_index_fp=dem_index_fp,
         out_dir=tmpdir)



@pytest.mark.parametrize('dem_fp',
                          [os.path.join(test_data_dir, r'build_dem_wse_fine_datasource\dgm5_4120_3260_20.asc')]
                          )
@pytest.mark.parametrize('wbdy_mask_fp',
                          [os.path.join(test_data_dir, 'build_dem_wse_fine_datasource', 'burned_domains_ems_test_clip_0115.tif')]
                          )
@pytest.mark.parametrize('min_wet_cnt',[1])
def test_merge_dem_wse_ds(da_clip, dem_fp, wbdy_mask_fp, min_wet_cnt, tmpdir):
        from haz.rim2019.downscale._02_fines import merge_dem_wse_ds as func
        func(da_clip, dem_fp, wbdy_mask_fp, min_wet_cnt=min_wet_cnt, out_dir=tmpdir)

 
#===============================================================================
# @pytest.mark.parametrize('basinName2', ['ems'])
# def test_apply_wse_downscale(basinName2, wse_da, tmpdir):
#     from haz.rim2019.downscale._03_fdsc import _apply_wse_downscale as func
#     
#     func(wse_da, basinName2=basinName2, out_dir=tmpdir)
#===============================================================================