'''
Created on Aug. 4, 2023

@author: cefect

extra data analysis  on rasters

'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'draft'
cm = 1 / 2.54

if env_type == 'journal': 
    usetex = True
elif env_type == 'draft':
    usetex = False
elif env_type == 'present':
    usetex = False
else:
    raise KeyError(env_type)

 
 
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')

def set_doc_style():
 
    font_size=8
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : font_size})
     
    for k,v in {
        'axes.titlesize':font_size,
        'axes.labelsize':font_size,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+2,
        'figure.autolayout':False,
        'figure.figsize':(17.7*cm,18*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v

#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='pdf',add_stamp=False,add_subfigLabel=True,transparent=True
        )            
#===============================================================================
# draft
#===============================================================================
elif env_type=='draft':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=True,transparent=True
        )          
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present': 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=False,transparent=False
        )   
 
    font_size=12
 
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(34*cm,19*cm), #GFZ template slide size
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# IMPORTs-------
#===============================================================================
import os, warnings, psutil, math, logging, itertools
from datetime import datetime

import numpy as np
import pandas as pd
 

from osgeo import gdal # Import gdal before rasterio
 
 
import xarray as xr
import rioxarray
import geopandas as gpd
 

import sparse

from definitions import asc_lib_d

from haz.hp import (
    view, get_temp_dir, today_str, init_log, dask_profile_func, dstr, dask_threads_func,
    dataArray_todense
    )
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, epsg_id,  index_d, dataArray_toraster
    )

from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray


def stats_on_sparse(basinName=None,
                      nc_fp=None,
                      stat_l=['max', 'mean'],
                      subdir='03_annMax',
                      indexName=None,
                      out_dir=None,
                      log=None,
 
                      ):
    """get stats from sprase stack"""
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()

 
    if nc_fp is None:
        nc_fp = get_sparse_fp(basinName, subdir=subdir, )
 
    #log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    if log is None:log = logging.getLogger().getChild('stats_on_sparse')
    
    if indexName is None:
        indexName = index_d[subdir]
    
    #===========================================================================
    # load
    #===========================================================================
    log.info(f'loading sparse-xarray from \n    {nc_fp}')
    
    sds = load_sparse_xarray(nc_fp, fix_relative=True)
    sda = list(sds.data_vars.values())[0] #get the first (should be only) dataArray
    
    #===========================================================================
    # compute stats
    #===========================================================================
 
 
    log.info(f'computing stats on {sda.shape}')
    d = dict()
    coo_ar = sda.data
    
    for stat in stat_l:
        log.info(f'computing {stat}')
        if not stat in ['nanmean']:        
            d[stat] = getattr(coo_ar, stat)(axis=(1,2)).todense()
 
    df = pd.DataFrame.from_dict(d)
    df.index =sda.coords[indexName]
    df.index.name=indexName
    
    
    df['real_count'] =  [coo_ar[i,:,:].nnz for i in range(sda.data.shape[0])]
    #===========================================================================
    # #add some indexers
    #===========================================================================
    d=dict()
    for k in [indexName, 'day', 'realisation', 'sparse_index']:
        if k in sda.coords:
            d[k] = sda.coords[k].values
 
 
    df.index = pd.MultiIndex.from_frame(pd.DataFrame.from_dict(d))
 
    
    log.info(f'finished w/ {df.shape}')
    
    sds.close()
    sda.close()
    
    return df


def run_stats_plot(
        basinName=None,
        subdir='03_annMax',
        f1_kwargs=dict(),
        pick_fp=None,
        out_dir=None,
        figsize=(16,6),
        line_kwargs = dict(
            markersize=5, linestyle='', fillstyle='none'
            ),
        log=None,
        ):
        
        
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, basinName, 'stats')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
 
    if log is None:
        log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    log.info(f'on {basinName}')
    #===========================================================================
    # compute stats
    #===========================================================================
    if pick_fp is None:
        df = stats_on_sparse(basinName=basinName, log=log, subdir=subdir, **f1_kwargs)
        
        pick_fp = os.path.join(out_dir, f'{basinName}_{subdir}_stats_{df.size}.pkl')
        df.to_pickle(pick_fp)
        
        #write to csv
        csv_ofp = os.path.join(out_dir, f'{basinName}_{subdir}_stats_{df.size}.csv')
        df.to_csv(csv_ofp)
        
        
        log.info(f'wrote {df.shape} to \n   {pick_fp}\n    {csv_ofp}')
        
    df = pd.read_pickle(pick_fp).drop('max', axis=1)
    
    
    
    #===========================================================================
    # plot stats
    #===========================================================================
    markers = itertools.cycle(('x', '+', 'o'))
    
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    #use the first indexer as the x
    xar = df.index.get_level_values(0)
    for stat, ser in df.items():
        if not stat in ['real_count']:
            ax1.plot(xar,ser.values,  label=stat, marker=next(markers), **line_kwargs)
        else:
            ax2.plot(xar,ser.values,  label=stat,   color='black', marker=next(markers), **line_kwargs)
        
    #===========================================================================
    # post
    #===========================================================================
    text = dstr({'count':len(df)})
    anno_obj = ax1.text(0.1, 0.9, text, transform=ax1.transAxes, va='center',fontsize=None)
    
    
    ax1.set_xlabel(xar.name)
    ax1.set_ylabel('depth (m)')
    ax2.set_ylabel('wet cell count')
    ax1.set_title(f'raster stats for {basinName}')
    ax1.grid()
    
    #===========================================================================
    # legend
    #===========================================================================
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
 
    ax1.legend(handles1 + handles2, labels1 + labels2)
 
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'{basinName}_rasterStats_{len(df)}_{today_str}.svg')
    fig.savefig(ofp, dpi =300,  transparent=True)
    log.info(f'saved figure to \n    {ofp}')
    
    return ofp
        

def export_raster(
        basinName=None,
        nc_fp=None,
        subdir='02_sparse2',
        log=None,
        indexName=None,
        out_dir=None,
        slice_l=[{}],
        #aoi_fp=None,
        ):
    """export a specific raster"""
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()

 
    if nc_fp is None:
        nc_fp = get_sparse_fp(basinName, subdir=subdir)
 
    #log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    
    if indexName is None:
        indexName = index_d[subdir]
        
        
         
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'export', today_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if log is None:log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    #===========================================================================
    # load
    #===========================================================================
    log.info(f'loading sparse-xarray from \n    {nc_fp}')
    
    sds = load_sparse_xarray(nc_fp)
    sda = list(sds.data_vars.values())[0] #get the first (should be only) dataArray
    
    #clipping polygon
    #===========================================================================
    # if not aoi_fp is None:
    #     aoi_vlay = gpd.read_file(aoi_fp)
    #===========================================================================
 
    #===========================================================================
    # slice
    #===========================================================================
    out_d = dict()
    log.info(f'slicing {len(slice_l)}')
    for i, slice_d in enumerate(slice_l):
        log.info(f'slicing w/ {slice_d}')
        
        try:
            #densify
            """better to index by raster_index as this does not change"""
            da_i = dataArray_todense(sda.swap_dims({'sparse_index':'raster_index'}).loc[slice_d])
            
            #===================================================================
            # #clip
            # xds.rio.clip(geometries)
            #===================================================================
            
            #write
            key_str = '-'.join([str(e) for e in slice_d.values()])
            
            out_d[i] = os.path.join(out_dir, f'{basinName}_{indexName}_{key_str}_{today_str}.tif')
            dataArray_toraster(da_i, out_d[i])
 
            log.info(f'wrote {da_i.shape} to \n    {out_d[i]}\n')
        except Exception as e:
            log.error(f'failed to write {slice_d} w/ \n    {e}')
        
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished w/ {len(out_d)} to \n    {out_dir}')
    
    return out_dir
 
        
        
    
 
def all_run(
        out_dir=None,
        basin_l = None,
        ):
    
    #===========================================================================
    # defautls
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'stats')
        
    if basin_l is None:
        basin_l = list(asc_lib_d.keys())

    log = init_log(fp=os.path.join(out_dir, today_str + '.log'))
    
    #===========================================================================
    # execute
    #===========================================================================
    res_d = dict()
    for b in basin_l:
        res_d[b] = run_stats_plot(basinName=b, out_dir=out_dir, log=log.getChild(b))
        
    #===========================================================================
    # wrap
    #===========================================================================
        
    print(f'finished w/ {dstr(res_d)}')
    
    return res_d
 



if __name__=="__main__":
    
    #all_run()
 
    #===========================================================================
    # plots
    #===========================================================================
    kwargs = dict( 
           basinName = 'elbe',  
           #pick_fp=r'l:\10_IO\2307_roads\outs\rim_2019\donau\stats\donau_02_sparse2_stats_7941.pkl', 
           subdir='02_sparse2' 
        )
          
    run_stats_plot( **kwargs)
    
    
    #===========================================================================
    # exports
    #===========================================================================
    #===========================================================================
    # l = [{'raster_index':e} for e in [1578, 8662, 1581, 1987, 3330]]
    # export_raster(basinName='rhine', slice_l=l, subdir='02_sparse2')
    #===========================================================================
    
    
    
    
    