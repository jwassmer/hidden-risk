'''
Created on Jan. 10, 2024

@author: cef
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd


from definitions import test_data_dir as test_data_dir_master
test_data_dir = os.path.join(test_data_dir_master, 'test_downscale_wse')

from haz.rim2019.parameters import coarse_dem_fp_d


#===============================================================================
# fixtures---------
#===============================================================================

@pytest.fixture(scope='function')
def wsh_da(basinName2, raster_index):
    bound_fn = {'ems':{
            1:'inundation_depth_001.pkl',
 
            }}[basinName2][raster_index]
            
            
    
    fp = os.path.join(test_data_dir, 'test_apply_wsh_to_wse', bound_fn)
    
    with open(fp, 'rb') as f:
        return pickle.load(f)
    
@pytest.fixture(scope='function')
def dem_fp(basinName2):
    return coarse_dem_fp_d[basinName2]
 
#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('basinName2', ['ems'])
def test_run_coarse_wse_stack(basinName2, tmpdir):
    from haz.rim2019.downscale._01_wse_coarse import run_coarse_wse_stack as func
    
    func(basinName2, out_dir=tmpdir
                   
                   )
    


@pytest.mark.parametrize('basinName2, raster_index', [('ems', 1)])
def test_apply_wsh_to_wse(basinName2, wsh_da,dem_fp, tmpdir):
    from haz.rim2019.downscale._01_wse_coarse import _apply_wsh_to_wse as func
    
    func(wsh_da, dem_fp=dem_fp, basinName2=basinName2, out_dir=tmpdir, use_cache=False)