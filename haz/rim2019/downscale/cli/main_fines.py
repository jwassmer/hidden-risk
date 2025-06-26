'''
Created on Jan. 10, 2024

@author: cef

compile rasters for each hyd basin
'''

import argparse, winsound, os



 



if __name__ == "__main__":
    
    #===========================================================================
    # build parser
    #===========================================================================
    parser = argparse.ArgumentParser(description='haz.rim2019.downscale._02_fines run_chunk_to_dem_tile')
    
    parser.add_argument('--basinName2', type=str,)    
   
    parser.add_argument('--out_dir', type=str,default=None)
    parser.add_argument('--min_wet_cnt', type=int,default=5)
    parser.add_argument('--use_cache', type=bool,default=True)
    parser.add_argument('--log_level', type=str,default='INFO')
    
    #dask kwargs
    parser.add_argument('--threads_per_worker', type=int,default=1)
    parser.add_argument('--n_workers', type=int,default=1)
 
    
    args = parser.parse_args()
    
    #===========================================================================
    # set arguments
    #===========================================================================
    print(f"All arguments received: {args}")
    
 
    
    from haz.rim2019.parameters import out_base_dir
    basinName2 = args.basinName2
    out_dir=args.out_dir
    
    if out_dir is None: out_dir = os.path.join(out_base_dir, 'downscale', '02fine', basinName2)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    
         
    #===========================================================================
    # logger
    #===========================================================================
    from haz.hp.argp import setup_logger
    root_logger = setup_logger(args)
    
    from haz.hp.basic import today_str, get_new_file_logger
    log = get_new_file_logger(fp=os.path.join(out_dir, f'{basinName2}_{today_str}.log')).getChild(basinName2)
    #===========================================================================
    # set up dask
    #===========================================================================
        
    
    
    #from dask.distributed import LocalCluster
    from dask.distributed import Client
    
    #===========================================================================
    # with LocalCluster( 
    #     #threads_per_worker=1, n_workers=12,processes=True, #total_time=43.17 secs, max_mem=5013.78 MB, max_cpu=231.1 %
    #     #threads_per_worker=12, n_workers=1,processes=False, #total_time=39.36 secs, max_mem=2619.95 MB, max_cpu=118.7 %
    #     #threads_per_worker=1, n_workers=1,processes=False, #total_time=39.61 secs, max_mem=2298.59 MB, max_cpu=125.0 %
    #     threads_per_worker=args.threads_per_worker, n_workers=args.n_workers,processes=False,
    #     memory_limit='auto', 
    #                     #use processes {'tdelta': '171.80 secs', 'outdir_GB': 0.015896964818239212}
    #                    ) as cluster, Client(cluster) as client:
    #===========================================================================
    with Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker, processes=True) as client:
    
        print(f'opening dask client {client.dashboard_link}')
        #=======================================================================
        # execute
        #=======================================================================

        
        from haz.rim2019.downscale._02_fines import run_chunk_to_dem_tile as func
        
        func(args.basinName2,
            out_dir=out_dir,
            min_wet_cnt=args.min_wet_cnt,
            use_cache=args.use_cache, log=log,       
            )
    
    try:
        winsound.Beep(440, 500)
    except:
        pass
    
