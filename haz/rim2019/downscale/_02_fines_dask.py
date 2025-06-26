'''
Created on Jan. 20, 2024

@author: cef
'''

import webbrowser

from dask.distributed import LocalCluster
from dask.distributed import Client
from dask.diagnostics import ResourceProfiler, visualize

from haz.hp.dask import dask_profile_func, _wrap_rprof
from haz.rim2019.downscale._02_fines import run_chunk_to_dem_tile


 
 


if __name__=='__main__':

    #===========================================================================
    # #start a cluster and connect client
    # with LocalCluster( threads_per_worker=1, 
    #                    n_workers=4,
    #                    memory_limit='auto', 
    #                    processes=False, #use processes
    #                    ) as cluster, Client(cluster) as client:
    #     
    #     print(f' opening dask client {client.dashboard_link}')
    #     webbrowser.open(client.dashboard_link)
    #     
    #     
    #     #{'tdelta': '159.40 secs', 'outdir_GB': 0.015896964818239212}
    #     run_downscale_fines(
    #             'ems',
    #             fine_ds_dir = r'l:\\10_IO\\2307_roads\\outs\\rim_2019\\downscale\\02fine\\ems',
    #             #raster_index_l=[1, 32],
    #             use_cache=False        
    #             )
    #===========================================================================
    
    
    #start a cluster and connect client
    with LocalCluster( 
        #threads_per_worker=1, n_workers=12,processes=True, #total_time=43.17 secs, max_mem=5013.78 MB, max_cpu=231.1 %
        #threads_per_worker=12, n_workers=1,processes=False, #total_time=39.36 secs, max_mem=2619.95 MB, max_cpu=118.7 %
        threads_per_worker=1, n_workers=1,processes=False, #total_time=39.61 secs, max_mem=2298.59 MB, max_cpu=125.0 %
                       memory_limit='auto', 
                        #use processes {'tdelta': '171.80 secs', 'outdir_GB': 0.015896964818239212}
                       ) as cluster, Client(cluster) as client:
         
        print(f' opening dask client {client.dashboard_link}')
        webbrowser.open(client.dashboard_link)
         
         
        #{'tdelta': '128.63 secs', 'outdir_GB': 0.015899572521448135}
        with ResourceProfiler(dt=0.25) as rprof:
            run_chunk_to_dem_tile(
                'ems',
                wse_ds_dir = r'l:\10_IO\2307_roads\outs\rim_2019\downscale\01coarse\ems\wse',
                #raster_index_l=[4409],
                #dem_fn_l = ['dgm5_4380_2800_20'],
                use_cache=False,
                write_tiff=False
                )
            
        # profile results
        _wrap_rprof(rprof)

        """seems to be ignoring the filename kwarg"""
        """this also doesn't fix it
        os.chdir(os.path.expanduser('~'))"""

        rprof.visualize(
            # filename=os.path.join(os.path.expanduser('~'), f'dask_ReserouceProfile_{today_str}.html'),
            # filename=os.path.join(wrk_dir, f'dask_ReserouceProfile_{today_str}.html')
        )
