'''
Created on Jan. 20, 2024

@author: cef
'''

import webbrowser

from dask.distributed import LocalCluster
from dask.distributed import Client
from dask.diagnostics import ResourceProfiler, visualize

from haz.hp.dask import dask_profile_func, _wrap_rprof
from haz.rim2019.downscale._03_fdsc import run_downscale_fines


 
 


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
    with LocalCluster( threads_per_worker=1, 
                       n_workers=4,
                       memory_limit='auto', 
                       processes=True, #use processes {'tdelta': '171.80 secs', 'outdir_GB': 0.015896964818239212}
                       ) as cluster, Client(cluster) as client:
         
        print(f' opening dask client {client.dashboard_link}')
        webbrowser.open(client.dashboard_link)
         
         
        #{'tdelta': '128.63 secs', 'outdir_GB': 0.015899572521448135}
        with ResourceProfiler(dt=0.25) as rprof:
            run_downscale_fines(
                    'ems',
                    fine_ds_dir = r'l:\\10_IO\\2307_roads\\outs\\rim_2019\\downscale\\02fine\\ems',
                    #raster_index_l=[1, 32],
                    dev_stack_cnt=1,
                    use_cache=False        
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
