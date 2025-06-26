'''
Created on Dec. 20, 2023

@author: cef

basic proejct setup tests
'''

import pytest, os


def test_params():
    """check module resolution is taking the correct parameters file"""
    from parameters import src_dir, logcfg_file
    
    assert not 'FloodDownscaler2' in src_dir
    
    assert os.path.exists(logcfg_file)
    
 
def test_submod():
    try:
        import fdsc
    except ImportError:
        pytest.fail("Failed to import submodule \'FloodDownscaler2\'")
    
    
