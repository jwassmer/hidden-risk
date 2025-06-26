'''
Created on Jul. 26, 2023

@author: cefect

common functions for rim2019
'''

import os
from datetime import datetime
import numpy as np
import pandas as pd

import geopandas as gpd
import shapely.geometry as sgeo
from pyproj import CRS

from definitions import wrk_dir
from haz.rim2019.parameters import *
#from haz.hp import view, get_temp_dir, today_str, get_directory_size, _wrap_rprof
from haz.hp.basic import today_str




 
def get_analysis_polygon(
        name,
        index_fp=None,
        ):
    """retrieve the polygon associated with an analysis basin name"""
    
    if index_fp is  None:
        from .parameters import basins_fp as index_fp
        
    # load polygon vector file (index_fp) from disk using geopandas
    gdf = gpd.read_file(index_fp)
    
    assert gdf.crs==CRS.from_user_input(epsg_id), 'EPSG mismatch'
    
    # select the feature named 'name'
    feature = gdf[gdf['name'] == name]
    
    # return the polygon from this feature
    polygon = feature.geometry.values[0]
    
 
    assert isinstance(polygon, sgeo.polygon.Polygon)
    
    return polygon
 

def load_lookup_df(asc_dir):
    """retrieve and load the lookup table from the directory"""
    fns = [e for e in os.listdir(asc_dir) if e.endswith('.csv')]
    assert len(fns) == 1, f'bad match on lookup:\n    {fns}'
    lookup_fp = os.path.join(asc_dir, fns.pop(0))
    print(f'found lookup table:\n    {lookup_fp}')
    #load
    df_raw = pd.read_csv(lookup_fp, index_col=0)
    print(f'loaded w/ {str(df_raw.shape)}')
    
    #===========================================================================
    # clean
    #===========================================================================
    #harmonize column names
    df = df_raw.rename(columns={
        'realisations':'realisation',
        'files':'filename',
        'original_name':'filename'
        }).drop(['scenario', 'basin'], axis=1)
    
    
    
    
    #check
    cols = df.columns.to_list()
    miss = set(['variable', 'realisation', 'day', 'filename']).symmetric_difference(cols) 
    assert miss==set(), f'column name mismach on {lookup_fp}\n    data:{cols}\n    miss:{miss}' 
    
    """
    view(df)
    """
    
    #===========================================================================
    # post
    #===========================================================================
    #drop realization zero
    bx = df['realisation']=='M0'
    if bx.any():
        print(f'found {bx.sum()}/{len(bx)} anomlous M0 values... dropping')
        df = df[~bx]
    
    #make realisations sortable
    df.loc[:, 'realisation'] = df['realisation'].str.split('M', expand=True)[1].astype(int)
    
    #add filepaths
    fp_l = [os.path.join(asc_dir, f'{v}.zip') for v in df.index.values]
    
    for i, fp in enumerate(fp_l):
        assert os.path.exists(fp), f'{i} bad filepath {fp}'
    df['filepath'] = fp_l
    
    #make index floats
    df.index = df.index.astype(int)
 
    #add raster type column
    #===========================================================================
    # lookup_df['data_type'] = lookup_df['original_name'].str.split('_', expand=True)[0]
    # 
    # print(lookup_df['data_type'].value_counts())
    #===========================================================================
    
    return df.sort_values(by=['realisation', 'day'], ascending=True)


 

def _get_out_dir(out_dir, subkey):
    
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, subkey, today_str)
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print(f'set out_dir\n    {out_dir}')
    return out_dir
        
                               
                            


#===============================================================================
# def plot_violin_realisations(df, ofp, 
#                              subplot_kwargs=dict(figsize=(10, 20)), **kwargs):
#     
#     fig, ax = plt.subplots(**subplot_kwargs)
#     
#  
#     
#     _ = sns.violinplot(data=df, x='day', y='realisation', cut=0, ax=ax, **kwargs)
#  
#     ax.figure.savefig(ofp, transparent=True)
#     print(f'wrote violin plot to \n    {ofp}')
#     
#     return ofp
#===============================================================================




                    
                    

    
    
    
    
    

