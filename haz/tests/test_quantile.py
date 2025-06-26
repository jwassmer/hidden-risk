'''
Created on Mar. 27, 2023

@author: cefect
'''

import pytest, copy, os, random, re, pickle
import numpy as np
import pandas as pd

xfail = pytest.mark.xfail

from haz.rim2019.quant._04_quantile import q_sparse
 
#===============================================================================
# helpers
#===============================================================================
 
#===============================================================================
# fixtures------------
#===============================================================================
 
        
#===============================================================================
# tetss------
#===============================================================================
 
 
@pytest.mark.parametrize('n', [random.randint(100, 1000) for _ in range(3)])
#@pytest.mark.parametrize('n', [100])
@pytest.mark.parametrize('miss_frac', [
    #0, 
    0.5, 
    0.1, 0.01]) 
@pytest.mark.parametrize('qraw', [0.9, 0.99, 
                                   0.5
                                   ])                            
def test_quantile_adjust(n, miss_frac, qraw):
    """validating our sparse quantile converter"""
    
    #get the data 
    ser = pd.Series(np.sort(np.random.rand(n))).sort_values().reset_index(drop=True)
    
    #get the sparse data
    miss_cnt = int(n*miss_frac) #number of records missing
    avail_cnt = n-miss_cnt
    ser_sparse = ser.sort_values(ascending=False).iloc[:avail_cnt]
    assert round((n-len(ser_sparse))/n, 2)==miss_frac
    
    
    #get the adjusted quantile    
    """
    total_cnt =n
    abs(avail_cnt - qraw*total_cnt)/avail_cnt
    
    """
    
    q_adj = q_sparse(qraw, n,avail_cnt)
    
    #compute the quantile using the adjustment
    percentile_sparse = np.quantile(ser_sparse.values, q_adj, method='linear')
    
    #compute the quantile on the full data
    percentile_full = np.quantile(ser.values, qraw, method='linear')
    
    #check
    print(f'n={n}, miss_frac={miss_frac} for q={qraw:.4f} ({q_adj:.4f}) got {percentile_sparse:.4f} and {percentile_full:.4f}')
    assert round(percentile_sparse,2)==round(percentile_full, 2)
    
    
 
    