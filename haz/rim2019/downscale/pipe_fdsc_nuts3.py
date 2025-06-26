'''
Created on Dec. 20, 2023

@author: cef

downscaling pipeline on nuts3 selection

computational discretization (spatial):
    see also haz.rim2019.downscale._03_fdsc.run_downscale_per_basin()

    1) lets assume hydro_basins are small enough to fit into memory
        2024-01-11:  Maybe Not... this is too hard to implement. see xxx_01_dem.py
    
    2) select DEM tiles per-flood event
        makes event parallelization difficult
        also requires loading all hi-res into memory
    
    3) iterate on DEM tiles
        don't really like this... lots of blank tiles... difficult to merge
        fourth try... doing this one
 
        
    4) build a hi-res DataSet for WSE and DEM
        makes for cleaner iteration and blocking
        but we don't want to resample ALL WSE and write to disc.. do this in line
        just need to convert the DEM to a DataSet
            this is too nasty
    
    
0) load WSH per hydro basin
    use netcdf DataSets from l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_nuts3_collect\
    

    
2) construct WSE per hydro_basin x event
    get_wse_ar()
    
    use netcdf DataSets from l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_nuts3_collect\
    
    output as DataSet in same format as 04_nuts3_collect (for potential concat)
        
    use caching
    
1) build DEm per WSH event
    select tiles from E:\05_DATA\Germany\BKG\DGM5\20230717\kacheluebersicht\dgm5_kacheln_20x20km_laea_EPSG_3035.shp
        filenames found on \\rzv230c.gfz-potsdam.de\hydro_public\DATEN\GIS\DGM_DEM\DGM5_Germany\
    concat and write to GeoTIFF (try 800x800 blocking)
    use caching
    
3) downscale the WSE.nc + DEM pairs (get a new high-res WSE)
    downscale_raster()
    
    output new Datasets (keep separate as resolution is different)
        these may be quite large... consider writing using subfolders: hydro_basin/event/WSE/downscaled_wse.nc
        
        NOTE: not sure which folder hierarchy Xarray likes best
    
    
    
4) construct high-res WSH
    get_wsh_rlay()
    
    output in same format as WSE dataset for eventual concat
    
5) export high-res WSH as GeoTiff

    

    
'''

import os, hashlib, psutil, pickle, shutil, logging,gc
from datetime import datetime

import pandas as pd

from haz.hp.basic import today_str, get_directory_size, get_log_stream, get_new_file_logger, view

#from haz.rim2019.scripts import get_wse_ar, downscale_raster

from haz.rim2019.downscale._01_wse_coarse import run_coarse_wse_stack
from haz.rim2019.downscale._02_fines import run_chunk_to_dem_tile
from haz.rim2019.downscale._03_fdsc import run_downscale_fines
#from haz.rim2019.downscale._04_org import run_organize
from haz.rim2019.downscale._04_toWSH import run_toWSH
from haz.rim2019.downscale._05_org import run_organize

 

from haz.rim2019.parameters import out_base_dir


def run_downscale_pipeline(
        basinName2,
        
        skip_l=None,
        
        log=None, out_dir=None, use_cache=True, tmp_dir=None,
        #log_level=logging.DEBUG,
        
        skwargs=dict()
        
        ):
    """full downscaling pipeline on DGM5 DEM and nuts3 events
    
    Params
    ---------
    skip_l: list
        optional steps to skip
    """
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now()
 
 
    if out_dir is None:
        out_dir = os.path.join(out_base_dir, 'downscale')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if tmp_dir is None:
        from haz.rim2019.parameters import tmp_dir
        tmp_dir = os.path.join(tmp_dir, 'downscale', basinName2)
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f'failed to remove tmp_dir\n    {tmp_dir}\n    {e}')
    if not os.path.exists(tmp_dir):os.makedirs(tmp_dir)
 
    if skip_l is None:
        skip_l = list()
    else:
        log.warning(f'skipping {len(skip_l)} steps\n    {skip_l}')
    assert isinstance(skip_l, list)
        
    #===========================================================================
    # #logging
    #===========================================================================
    #simple stream default
    if log is None:
        log = get_log_stream()
        
    #create a DEBUG file logger for this simulation
    log = get_new_file_logger(fp=os.path.join(out_dir, f'{basinName2}_{today_str}.log'), logger=log)
 
    
    log.info(f'on {basinName2}')
    log.debug(skwargs)
    #===========================================================================
    # defaults
    #===========================================================================
 
    
    #functino kwargs
    k_l = ['run_coarse_wse_stack', 'chunk_to_dem_tile', 'run_downscale_fines', 'run_toWSH', 'run_organize']
    for k in k_l:
        if not k in skwargs:
            skwargs[k] = dict()
            
    #check the skip keys
    for k in skip_l:
        assert k in k_l, k

    
    res_d, err_d=dict(), dict()
    ikwargs = dict(log=log, use_cache=use_cache, tmp_dir=tmp_dir)
    #===========================================================================
    # execute pipeline-------
    #===========================================================================
    
    #===========================================================================
    # get the coarse WSE---------
    #===========================================================================
    k = k_l[0]
    log.info(f'\n\n{k}\n=================================================\n\n')
    wse_ds_dir = os.path.join(out_dir,  '01coarse', basinName2, 'wse')    
    res_d[k] = run_coarse_wse_stack(basinName2, 
                         out_dir = wse_ds_dir,
                         **skwargs[k], **ikwargs)
    
    gc.collect()
    #=======================================================================
    # get fine data stack (chunked to DEM tiles)--------
    #=======================================================================
    
    k=k_l[1]
    log.info(f'\n\n{k}\n=================================================\n\n')
    fine_ds_dir = os.path.join(out_dir,  '02fine', basinName2)
    if not k in skip_l:
        res_d[k] = run_chunk_to_dem_tile(basinName2,
                                     wse_ds_dir=wse_ds_dir,
                                     out_dir = fine_ds_dir,
                                     **skwargs[k], **ikwargs
                                     )
        gc.collect()
    else:
        log.warning(f'skipping {k}')
    
    #===========================================================================
    # downscale WSE on DEM--------
    #===========================================================================
    k=k_l[2]
    log.info(f'\n\n{k}\n=================================================\n\n')
    fdsc_data_dir = os.path.join(out_dir,  '03fdsc', basinName2)
    fdsc_dx, err_d[k] = run_downscale_fines(basinName2,
                                 fine_ds_dir=fine_ds_dir,
                                 out_dir = fdsc_data_dir,
                                 **skwargs[k], **ikwargs
                                 )
    
    res_d[k] = fdsc_dx
    gc.collect()
    #===========================================================================
    # WSH-------
    #===========================================================================
    k=k_l[3]
    log.info(f'\n\n{k}\n=================================================\n\n')
    wsh_dir = os.path.join(out_dir, '04wsh', basinName2)
    wsh_dx, err_d[k] = run_toWSH(fdsc_dx=fdsc_dx, data_dir = fdsc_data_dir, out_dir=wsh_dir,
                                 **skwargs[k], **ikwargs)
    res_d[k] = wsh_dx
    gc.collect()
    #===========================================================================
    # ORG----
    #===========================================================================
    k=k_l[4]
    log.info(f'\n\n{k}\n=================================================\n\n')
    res_d[k], err_d[k] = run_organize(wsh_dx_raw=wsh_dx,data_dir=wsh_dir,
                                      **skwargs[k], **ikwargs)
 
    
    #add raster-index meta
    try:    
        """nice summary meta for keeping track of event counts"""
        d = {
            'tile_cnt':res_d[k].groupby('raster_index').count()['raster_fp'],
            'wet_frac_mean':res_d[k]['wet_frac'].groupby('raster_index').mean(),
            'mean':res_d[k]['mean'].groupby('raster_index').mean(),        
            }
        
        res_d = {**{'raster_index_smry':pd.DataFrame.from_dict(d)}, **res_d}
    except Exception as e:
        log.error(f'failed to buidl raster-index meta w/\n    {e}')
    
    """
    view(res_d[k])
    view(res_d[k].groupby('raster_index').count())
    """
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                        'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                        #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                        'outdir_GB':get_directory_size(out_dir),
                        'tmpdir_GB':get_directory_size(tmp_dir),
                        #'output_MB':os.path.getsize(ofp)/(1024**2)
                        }

    #write meta
    fp_d = {'res': os.path.join(out_dir, f'{basinName2}_downscale_pipe_smry_{today_str}.xlsx')}
    with pd.ExcelWriter(fp_d['res']) as writer:
        for tabnm, df in res_d.items():
            df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
            
    log.info(f'wrote {len(res_d)} tabs to \n    %s'%fp_d['res'])
            
    #write errors
    err_d = {k:v for k,v in err_d.items() if not v is None}
    if len(err_d)>0:
        #err_df = pd.concat(err_d)
        #err_df.to_csv(os.path.join(f'{basinName2}_errors_{len(err_d)}_{today_str}.csv'))
        raise IOError(f'{basinName2} finished w/ {len(err_d)} errors\n{meta_d}')
    
        
 
 
        
    log.info(f'finished {basinName2} to \n    {out_dir}')
    

            
    
    log.info(meta_d)
    
    return res_d
    
    
    
if __name__=='__main__':
    #===========================================================================
    # params
    #===========================================================================
    import config
    processes=4
    config.log_level = logging.INFO
 
    #===========================================================================
    # execute
    #===========================================================================
    run_downscale_pipeline(
        'ems',
        skwargs=dict(
            run_coarse_wse_stack=dict(
                haz_basin_index_fp=r'l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_collect\nuts3_collect_meta_7_7_20240126.gpkg'
                ),
            run_downscale_fines=dict(
                processes=processes
                #raster_index_l=[1, 32]
                ),
            run_toWSH=dict(
                processes=processes
                )
            )
        )
    
    print('done')

