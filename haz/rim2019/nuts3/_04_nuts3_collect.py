'''
Created on Oct. 10, 2023

@author: cefect

collect all of the nuts3 selected events and organize


'''
import os, warnings, psutil, hashlib, pickle, gc
 
from datetime import datetime

import numpy as np
import pandas as pd
idx = pd.IndexSlice
from pandas.testing import assert_frame_equal

import sparse
import dask

import shapely.geometry 

import geopandas as gpd
import xarray as xr
import rioxarray

from definitions import asc_lib_d
from haz.rim2019.parameters import epsg_id

from haz.hp.basic import (
    view, get_temp_dir, today_str, get_new_file_logger,   dstr,   get_directory_size, get_log_stream
    )

from haz.hp.dask import dataArray_todense
 
 
from haz.rim2019.coms import  (
    out_base_dir, lib_dir, exclude_lib, cache_base_dir
    )

from haz.rim2019.nuts3._03_nuts3_event_selec import get_select_meta_fp, get_select_haz_stack_fp

from haz.rim2019._02_nc_to_sparse import get_sparse_fp, load_sparse_xarray, write_sparse_xarray
 

        
        
        
    

def run_basin_nuts3_event_stack( 
        basins_fp=None,
                         out_dir=None,temp_dir=None,
                         nuts_select_dir=None,
                         
                         event_cnt=5,min_wet_cnt=10,
 
                          use_cache=True,
                          dev=False, log=None,
                          basin_l=None,
                          encoding = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'least_significant_digit':2},
                          ):
    """collect and prepare per-basin nuts3 derived event stack
    
    to identify the event stack per basin
        we use extreme events per nuts 3 
            see select_events_per_zone()
    
    
    Writes
    ------------
    event stack: DataSet
        collected, slicked, and masked water depth events per-basin
    
    analysis basin index: GeoDataFrame
        links each basin to its event stack DataSet
        
    event index: pd.DataFrame
        metadata on each event
        
    nuts3 index: pd.DataFrame
        metadata and selected events per-nuts3
        
    nuts_select_dir: str
        directory of run_select_events_per_zone outputs (above basin level)
    
 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'nuts3', '04_collect')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if log is None:
        log = get_log_stream()
        log = get_new_file_logger(fp=os.path.join(out_dir, today_str + '.log'), logger=log)
 
    
    if basins_fp is None:
        from definitions import basins_fp
        
    if basin_l is None:
        basin_l=list(asc_lib_d.keys())
        
    if nuts_select_dir is None:
        nuts_select_dir = os.path.join(out_base_dir, 'nuts3', '03_select')
        
    assert os.path.exists(nuts_select_dir), f'bad path specified for run_select_events_per_zone outputs'
        
    #===========================================================================
    # load indexers
    #===========================================================================
    #polygon meta of nuts3 (needed for intersections)
    nuts3_gdf = get_merged_meta(log=log, use_cache=False, 
                                #basin_l=basin_l, 
                                nuts_select_dir=nuts_select_dir,
                                out_dir=out_dir)
    
    #basins polygons
    basins_gdf = gpd.read_file(basins_fp).rename(columns={'layer':'basinName'})
    """
    basins_gdf.plot()
    view(nuts3_gdf.drop('geometry', axis=1))
    
    view(nuts3_gdf.drop('geometry', axis=1).set_index('basinName', append=True).xs('rhine', level='basinName'))
    """
    
    #===========================================================================
    # loop on basins----
    #===========================================================================
    log.info(f'looping on {len(basins_gdf)} basins')
    haz_res_lib, nuts_res_lib=dict(), dict()
    meta_lib=dict()
    odi = os.path.join(out_dir, 'event_ds')
    if not os.path.exists(odi):os.makedirs(odi)

    for i, row in basins_gdf.iterrows():
        
        
        #=======================================================================
        # precheck
        #=======================================================================
        basinName2 = row['name'] #subdivided basins
        
        
        if not row['basinName'] in basin_l:
            log.warning(f'skipping %s'%row['basinName'])
            continue
        
        #=======================================================================
        # setup
        #=======================================================================
        
        log.info(f'\non {i} {basinName2}\n------------------------\n\n')
        
 
        #=======================================================================
        # select valid nuts for this basin by label and intersect
        #=======================================================================
        """because there are some nuts that show up in multiple basins
        and that we further sub-divided the basins
        we use this 2-level selection to get the relevant nuts3"""
        #by basin
        bx_basin = nuts3_gdf['basinName']==row['basinName'] 
        
 
        #by centroid 
        bx_geo = nuts3_gdf.geometry.centroid.intersects(row.geometry)
 
        #too inclusive
        #bx_geo = nuts3_gdf.geometry.intersects(row.geometry)
        
        bx = np.logical_and(bx_basin, bx_geo)
        
        log.info(f'{basinName2} got the following selection from {len(bx)}\n    basin:{bx_basin.sum()}\n    geom:{bx_geo.sum()}\n    AND:{bx.sum()}')
        
        nuts3_gdf_i =nuts3_gdf.loc[bx, :].sort_index()
        
        """
        import matplotlib.pyplot as plt
        plt.show()
        nuts3_gdf_i.plot()
        
        view(nuts3_gdf_i.dro)
        
        nuts3_gdf_i.to_file(os.path.join(out_dir, f'{basinName2}_nuts3_gdf_i_{len(nuts3_gdf_i)}.gpkg'))
        """
        
        #=======================================================================
        # retrieve hazard stack data (and slice a bit)
        #=======================================================================
        """rdx output from run_select_events_per_zone for this basin
        does ? contain the empty nuts ??
        
        """
        haz_meta_fp = get_select_haz_stack_fp(row['basinName'], srch_dir=os.path.join(nuts_select_dir, row['basinName']))
        log.debug(f'loading hazard meta from \n    {haz_meta_fp}')
        stack_dx = pd.read_pickle(haz_meta_fp).swaplevel(i=1, j=-1, axis=1).sort_index(axis=1)
        
        log.debug(f'%s got %i raster_index and %i nuts'%(
            row['basinName'], len(stack_dx.index.unique('raster_index')), len(stack_dx.columns.unique('nuts_index'))))
        
        """
        view(stack_dx.head(100))
        """
        
        #slice to selected nuts
        """some basins should select most nuts (some are dropped because they dont fit into the new zones)
        some basisns select ~half (these were split)"""
        
        """2024-01-25: stack_dx only has wet nuts... not sure about this assertion
        
        I think this should have failed on the rhine basins... but they were oringall run w/ __debug__=False
        """
        #=======================================================================
        # miss_s = set(nuts3_gdf_i['NUTS_ID']).difference(stack_dx.columns.unique('NUTS_ID'))
        # if not miss_s==set():
        #     raise IOError(f'{basinName2}  missing {len(miss_s)} nuts from stack\n    {miss_s}')
        #=======================================================================
        
        """no... for split hydro_basins we will often not ahve all the nuts
        if not set(stack_dx.columns.unique('NUTS_ID')).difference(nuts3_gdf_i['NUTS_ID'])==set():
            raise IOError(f'missing some nuts in the stack')"""
        
        
        bx = stack_dx.columns.get_level_values('NUTS_ID').isin(nuts3_gdf_i['NUTS_ID'])
        if not bx.all():
            log.warning(f'{basinName2} selected {bx.sum()}/{len(nuts3_gdf_i)} nuts in both the spatial meta and the event selection data')
 
        
        stack_dx_i = stack_dx.loc[:, bx]
        log.info(f'    selecting %i/%i nuts3 from the stack'%(
            len(stack_dx_i.columns.unique('nuts_index')), len(stack_dx.columns.unique('nuts_index'))
            ))
        log.debug(f'loaded stack_dx w/ {stack_dx.shape}')
        
        """
        view(nuts3_gdf_i.drop('geometry', axis=1))
        view(stack_dx.head(100))
        """
        
        #=======================================================================
        # prepare stack
        #=======================================================================
        #get the most extreme per nut
        nuts_d, haz_dx = _basin_collect_event_extremes(stack_dx_i, log=log.getChild(basinName2),
                                                       event_cnt=event_cnt, min_wet_cnt=min_wet_cnt)
        
        #=======================================================================
        # crop and write selected rasters
        #=======================================================================
        #load teh sparse data
        nc_fp = get_sparse_fp(row['basinName'])    
    
        #clip rasters per stack and basin
        uuid = hashlib.shake_256(f'{row}_{nc_fp}_{haz_dx.index}'.encode("utf-8")).hexdigest(12) 
        ofp_i = os.path.join(odi, f'{basinName2}_{uuid}.nc')
        
        if (not os.path.exists(ofp_i)) or (not use_cache):
            haz_ds = _basin_haz_stack_clip(row.geometry, haz_dx, nc_fp,log=log.getChild(basinName2), min_wet_cnt=min_wet_cnt)
            
            #add some meta
            haz_ds = haz_ds.assign_coords({'basinName':row['basinName'], 'basinName2':row['name']})
            
 
            log.info(f'writing dataset {basinName2} ({haz_ds.dims}) w/ encoding\n    {encoding}')
            haz_ds['inundation_depth'].encoding.update(encoding) #donau raw=2.4GB
            haz_ds.to_netcdf(ofp_i, format='netCDF4',engine='netcdf4',mode='w')
            log.info(f'wrote DataSet ({haz_ds.dims}) to \n    %s'%ofp_i)
        else:
            log.info(f'haz_ds already exists... loading from file')
            
            haz_ds = xr.load_dataset(ofp_i)
        
        #=======================================================================
        # meta
        #=======================================================================
 
        """this happens for basins where we discard some events below the min_wet_cnt during _basin_haz_stack_clip()"""
        if not set(haz_ds['raster_index'].values).symmetric_difference(haz_dx.index.unique('raster_index'))==set():
            log.warning(f'raster_index request mismatch on {basinName2}') 
        
        meta_d = row.to_dict()
        
        #raster indicies in stack
        ri_ar = haz_ds['raster_index'].values
        
        #slice haz_dx
        bx = haz_dx.index.get_level_values('raster_index').isin(ri_ar)
        assert bx.any()
        if not bx.all():
            """for split basins"""
            log.warning(f'{basinName2} got {bx.sum()}/{len(bx)} events')
            haz_dx = haz_dx[bx]
            
        haz_dx['basinName2'] = basinName2
        
        meta_d['haz_ds_fp']=os.path.basename(ofp_i)
        meta_d['haz_cnt']=  len(haz_dx)
        meta_d['nuts_cnt'] = len(nuts_d)
        meta_d['basinName2'] = basinName2
 
        #=======================================================================
        # wrap
        #=======================================================================
        haz_ds.close()
        haz_ds=None
        
        meta_lib[basinName2] = meta_d 
        haz_res_lib[basinName2] = haz_dx
        nuts_res_lib[basinName2] = pd.concat(nuts_d, names=['nuts_index'])
        gc.collect()
        
 
    #===========================================================================
    # write------
    #===========================================================================
    log.info(f'finished on {len(meta_lib)}')
    
    #===========================================================================
    # spatial
    #===========================================================================
    meta_gdf1 = pd.DataFrame.from_dict(meta_lib).T.reset_index(drop=True)
    meta_gdf2 = gpd.GeoDataFrame(meta_gdf1.drop('geometry', axis=1), geometry=meta_gdf1.geometry, crs=basins_gdf.crs)
    
    fn_str = '_'.join([str(e) for e in meta_gdf2.shape])
    ofp = os.path.join(out_dir, f'nuts3_collect_meta_{fn_str}_{today_str}.gpkg')
    
    """
    meta_gdf2.columns
    """
    
    try:
        if os.path.exists(ofp):os.remove(ofp)
        
    except Exception as e:
        log.error(f'failed to write spatial meta w/\n    {e}')
    meta_gdf2.to_file(ofp)
    log.info(f'wrote {meta_gdf2.shape} to \n    {ofp}')
    
    #===========================================================================
    # #haz lib. stats per raster_index
    #===========================================================================
    
    haz_dx = pd.concat(haz_res_lib, names=['basinName2'])
    
    """
    view(haz_dx.head(100))
    
                                                                nuts_count  ...  rank_mean
    basinName2 sparse_index day   raster_index realisation              ...           
    ems        0            7596  1            1                     3  ...        0.0
               36           3583  118          49                    3  ...        1.0
           
    """
    
    fn_str = '_'.join([str(e) for e in haz_dx.shape])
    ofp = os.path.join(out_dir, f'nuts3_collect_haz_dx_{fn_str}_{today_str}.pkl')
    haz_dx.to_pickle(ofp)
    log.info(f'wrote {haz_dx.shape} to \n    {ofp}')
    
    haz_dx.to_csv(ofp.replace('.pkl', '.csv'))
    
    log.info(f'w/ \n%s'%haz_dx['nuts_count'].groupby('basinName2').sum())
    
    
    #===========================================================================
    # #nuts lib. stats per nuts
    #===========================================================================
    nuts_dx = pd.concat(nuts_res_lib, names=['basinName2'])
    
    """
    view(nuts_dx.head(100))
    
    mdf = nuts_dx.index.to_frame().reset_index(drop=True)
    
    view(nuts_dx.xs('weser', level='basinName2').index.unique('raster_index').to_frame())
    
    metric                                                             rank  ...  wet_cnt
    basinName2 nuts_index sparse_index day   raster_index realisation        ...         
    ems        1          0            7596  1            1               0  ...   1438.0
                          36           3583  118          49              1  ...    906.0
                          26           24458 81           34              2  ...    406.0
    """
    
    fn_str = '_'.join([str(e) for e in nuts_dx.shape])
    ofp = os.path.join(out_dir, f'nuts3_collect_nuts_dx_{fn_str}_{today_str}.pkl')
    nuts_dx.to_pickle(ofp)
    nuts_dx.to_csv(ofp.replace('.pkl', '.csv'))
    log.info(f'wrote {nuts_dx.shape} to \n    {ofp}')
    
    #count of unique events per basin
    s1 = nuts_dx.index.to_frame().reset_index(drop=True).groupby('basinName2').nunique()['raster_index']
    log.info(f'w/ \n{s1}\nTOTAL=%i'%s1.sum())
    
    
    
        
    #===========================================================================
    # wrap
    #===========================================================================
 
    
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'outdir_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 


def _basin_collect_event_extremes(stack_dx, log=None, 
              
                                  event_cnt=5,
                                  select_metric='wet_cnt',
                                  min_wet_cnt=10,
                                  ):
    """
    for the nuts3, identify the collective extreme event set
        basically, we use the nuts3 sub-spatial discretization to identify the event set
        
        
    Params
    -------
    event_cnt: int
        number of events to select from each nuts3
        
    select_metric: str
        metric to use for selection
        
    Returns
    -----------
    dict
        nuts indexed selection of events
        nuts_index: pd.DataFrame of selected events
        
    pd.DataFrame
        hazard indexed selection of events for this basin (with some stats)
    
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    
    log.info(f'on {stack_dx.shape}')
    
    """
    view(stack_dx.head(100))
    """
    
    #===========================================================================
    # loop and collect 5 worst
    #===========================================================================
    res_d = dict()
    for nuts_index, gdx in stack_dx.T.groupby('nuts_index'):
        """
        gdx
        """
        #make the selection for this nuts
        sel_df1 = gdx.T.droplevel([0,1,2], axis=1).sort_values(select_metric, ascending=False
                        ).iloc[:event_cnt, :].dropna(how='any', axis=0)
                        
        #filter tiny floods
        bx = sel_df1['wet_cnt']>min_wet_cnt
        if not bx.any():
            log.warning(f'{nuts_index} failed to get any events staisfying min_wet_cnt={min_wet_cnt}...skipping')
            continue
                        
        #add rank
        sel_df2 = sel_df1[bx].reset_index().reset_index().set_index(sel_df1.index.names).rename(columns={'index':'rank'})
        
        #check
        if not not sel_df2.isna().any().any():
            raise AssertionError(f'got some nulls on {nuts_index}')
        
        assert len(sel_df2)>0
        
        res_d[nuts_index] = sel_df2
        
    #===========================================================================
    # #collect
    #===========================================================================
    #join all of the selected metadata (really just care about the index here
    cat_dx = pd.concat(res_d.copy(), names=['nuts_index'], axis=1)
    
    #compute event-based stats
    #shows the number of times an event was selected as the worst by a nuts3
    res_dx = pd.concat([
        cat_dx.count(axis=1).rename('nuts_count'),
        cat_dx.drop('rank', axis=1, level=1).T.groupby('metric').sum().T,
        cat_dx.xs('rank', axis=1, level=1).mean(axis=1).rename('rank_mean'),
        cat_dx.xs('rank', axis=1, level=1).max(axis=1).rename('rank_max'),
        ], axis=1).sort_values('nuts_count', ascending=False)
    
    """
    view(cat_dx.head(100))
    view(res_dx)
    """
    
    #===========================================================================
    # wrap
    #===========================================================================
    return res_d, res_dx

def get_merged_meta(
        out_dir=None,
        use_cache=True,
        log=None,
        basin_l=None,
        
        nuts_select_dir=None,
        ):
    """merge all of the meta gpgkgs into 1"""
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, '04_nuts3_collect')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if log is None:
        log = get_log_stream()
        
    if basin_l is None:
        basin_l=list(asc_lib_d.keys())
        
    #===========================================================================
    # search and load each
    #===========================================================================
    fp_d = dict()
    for k in basin_l:
        fp_d[k] = get_select_meta_fp(k, srch_dir = os.path.join(nuts_select_dir, k))
        
    log.info(f'found {len(fp_d)} meta files')
    #===========================================================================
    # #setup cache
    #===========================================================================
    uuid = hashlib.shake_256(f'{fp_d}'.encode("utf-8")).hexdigest(8)    
 
 
    ofp = os.path.join(out_dir, f'nuts3_select_meta_merge_{len(fp_d):03d}_{uuid}.gpkg')
    
    
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # load and merge
        #===========================================================================
        
        d=dict()
        for k, fp in fp_d.items():
            gdf = gpd.read_file(fp)
            
            #gdf['basinName'] = k
            #gdf.index.name='nuts3_id'
            
            d[k] = gdf
        
    
        
        #===========================================================================
        # #merge and write
        #===========================================================================
        merge_df = pd.concat(d.values()).reset_index(drop=True).rename(columns={'catchment':'basinName'})
        
        merge_gdf = gpd.GeoDataFrame(merge_df.drop('geometry', axis=1), geometry=merge_df.geometry, crs=gdf.crs)
        
        merge_gdf.to_file(ofp)
        
        """
        view(merge_df.drop('geometry', axis=1).head(100))
        
        import matplotlib.pyplot as plt
        plt.show()
        merge_gdf.plot()
        
        view(merge_gdf)
        """
        
        log.info(f'built and wrote {merge_gdf.shape} to file\n    {ofp}')
    else:
        merge_gdf = gpd.read_file(ofp)
        log.info(f'loaded {merge_gdf.shape} from cache\n    {ofp}')
        
    #===========================================================================
    # wrap
    #===========================================================================
    return merge_gdf
        




def _basin_haz_stack_clip(poly, stack_dx, nc_fp,   log=None, min_wet_cnt=10):
    """clip and prep teh selected stack
    
    Params
    ---------
    poly: POLYGON
        footprint of hydro_basin
        
    stack_dx: pd.DataFrame
        index of events selected for this hydrobasin (poly)
    
                                                         nuts_count  ...  rank_mean
        sparse_index day   raster_index realisation              ...           
        217          25556 482          2                    75  ...   0.520000
        2429         27406 3325         10                   60  ...   1.050000
        186          21591 448          2                    39  ...   2.307692
        
    """
    
 
    
    log.info(f'on {stack_dx.shape}')
    
    """
    view(stack_dx.head(100))
    """
    #helrper to check if we still have all the events
    get_ri_dif = lambda x: set(x['raster_index'].values).symmetric_difference(stack_dx.index.unique('raster_index'))
    #===========================================================================
    # load sparse araray
    #===========================================================================
    sds = load_sparse_xarray(nc_fp)
    sda = list(sds.data_vars.values())[0] #get the first (should be only) dataArray
    
    #check index presence
    if not set(stack_dx.index.unique('raster_index')).difference(sda['raster_index'].values)==set():        
        raise IOError(f'missing some selected raster_indexs in teh sparse data\n    {nc_fp}')
    
    #check index consistenncy
    nc_df = sda['sparse_index'].to_dataframe().reset_index(drop=True).sort_values('sparse_index').drop('spatial_ref', axis=1)
    assert nc_df['raster_index'].is_unique
    
    st_df = stack_dx.index.to_frame().reset_index(drop=True).set_index('sparse_index').sort_index() 
    
    nc_intersect_df = nc_df.loc[st_df.index, :].set_index('sparse_index').loc[:, st_df.columns.values].sort_index()
    
    """stopped here... not sure why the day columns are different... or whether it matters"""
    try:
        assert_frame_equal(st_df, nc_intersect_df)
    except Exception as e:
        log.error(e)
        
    
    
    
    
    #===========================================================================
    # #event slice
    #===========================================================================
    log.info(f'slicing by %i events'%len(stack_dx.index.unique('sparse_index')))
    sda1 = sda.swap_dims({'sparse_index':'raster_index'}).loc[{'raster_index':stack_dx.index.unique('raster_index')}]
    
 
    
    assert get_ri_dif(sda1)==set(), f'lost {len(get_ri_dif(sda1))} raster_index'
    #===========================================================================
    # bbox slice
    #===========================================================================
    #bbox
    sda2 = sda1.rio.clip_box(*poly.bounds)
 
    #===========================================================================
    # clip by poly
    #===========================================================================
    """because some basins are sub-divided... need a true polygon clip"""
    #densify
    """Im assuming that dense arrays will be fine as we are only dealing with the most extreme
    alternatively, could keep things sparse and only mask the polygon when writing the raster"""
    log.debug(f'densify {sda2.shape}')
    sda3 = dataArray_todense(sda2).reset_coords('sparse_index', drop=True)
 
    log.debug(sda3.shape)
    
    #full clip (masks cells outside polygon)
    sda4 = sda3.rio.clip([shapely.geometry.mapping(poly)],all_touched=True, drop=False)
    
    #===========================================================================
    # check
    #===========================================================================
    #raster_index
    #ri_ar = sda4['raster_index'].values
    if not get_ri_dif(sda4)==set():
        raise IndexError(f'lost {len(get_ri_dif(sda4))} indicies')
 
    
    #check empties
    max_wsh_ser = (sda4.fillna(0.0)>0.0).sum(['x', 'y']).to_dataframe(name='wsh')['wsh']
    bx = max_wsh_ser<min_wet_cnt
    if bx.any():
        log.warning(f'got {bx.sum()}/{len(bx)} events w/ min_wet_cnt<{min_wet_cnt}... dropping these')
        ri_ar = max_wsh_ser[~bx].index
        
        sda4 = sda4.loc[{'raster_index':ri_ar}]
        log.debug(sda4.shape)
        
        """could clip these"""
        #raise IOError(f'got {bx.sum()}/{len(bx)} dry events')
        """
        view(max_wsh_ser)
        """
    """
    sda4.reset_coords(names=['raster_index', 'realisation'], drop=True
    ).isel(sparse_index=range(0,9)).plot.imshow(x='x',y='y', col='sparse_index', col_wrap=3)
    
    import matplotlib.pyplot as plt
    plt.show()
    """
    
    
    
    if len(sda4)==0:
        raise AssertionError('bad clip')
    
 
    
    log.debug(f'clip results:\n    og:{sda.shape}\n    event:{sda1.shape}\n    bbox:{sda2.shape}')
    
    return sda4.to_dataset(name=sda.name)
    
     
    
 
    
    
        
        
if __name__=="__main__":
    
    run_basin_nuts3_event_stack(
        log = get_log_stream(),
        #basin_l=['rhine'],
        use_cache=True,
        #=======================================================================
        # basin_l=['donau', 'weser', 
        #          #'ems'
        #          ],
        #=======================================================================
        )
    
    
    
    
    
    
    