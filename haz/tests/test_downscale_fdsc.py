'''
Created on Jan. 14, 2024

@author: cef
'''


import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd


from definitions import test_data_dir as test_data_dir_master
test_data_dir = os.path.join(test_data_dir_master, 'test_downscale_fdsc')


#===============================================================================
# fixtures---------
#===============================================================================
 

@pytest.fixture(scope='function')
def fine_ds_dir(basinName2):
   
    return os.path.join(test_data_dir, 'test_run_downscale_fines', basinName2)


@pytest.fixture(scope='function')
def fine_ds(basinName2):
    
    fp = os.path.join(test_data_dir, 'test_apply_wse_downscale', basinName2, 'ds.pkl')
    
    with open(fp, 'rb') as f:
        return pickle.load(f)
    
 

#===============================================================================
# tests---------
#===============================================================================
@pytest.mark.dev
@pytest.mark.parametrize('basinName2, raster_index_l', [
    ('ems', [1, 81]),
    ])
@pytest.mark.parametrize('processes', [
    #None, #84.00s
    2, #51.94
    ])
@pytest.mark.parametrize('dev_stack_cnt', [2])
def test_run_downscale_fines(basinName2, raster_index_l, fine_ds_dir, processes, dev_stack_cnt, tmpdir):
    from haz.rim2019.downscale._03_fdsc import run_downscale_fines as func
    
    func(basinName2, fine_ds_dir=fine_ds_dir, raster_index_l=raster_index_l, out_dir=tmpdir,
         processes=processes, dev_stack_cnt=dev_stack_cnt)
    
    


@pytest.mark.parametrize('basinName2', ['ems'])
def test_apply_wse_downscale(basinName2, fine_ds, tmpdir):
    from haz.rim2019.downscale._03_fdsc import _apply_wse_downscale as func
    
    func(fine_ds, basinName2=basinName2, out_dir=tmpdir, tmp_dir=os.path.join(tmpdir, 'temp'))