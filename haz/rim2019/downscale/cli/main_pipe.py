'''
Created on Jan. 10, 2024

@author: cef

compile rasters for each hyd basin
'''

import argparse, winsound

def str2list(value):
    return value.split(',')
 



if __name__ == "__main__":
    print('start')
    #===========================================================================
    # build parser
    #===========================================================================
    parser = argparse.ArgumentParser(description='haz.rim2019.downscale.pipe_fdsc_nuts3 run_downscale_pipeline')
    
    parser.add_argument('--basinName2', type=str,default='ems')    
    parser.add_argument('--haz_index', type=str,default=None)    
    parser.add_argument('--out_dir', type=str,default=None)
    parser.add_argument('--processes', type=int,default=None)
    parser.add_argument('--log_level', type=str,default='INFO')
    
    parser.add_argument('--skip_l', type=str2list, default=None)
    
 
    
    args = parser.parse_args()
    
 

    
    #===========================================================================
    # set arguments
    #===========================================================================
    print(f"All arguments received: {args}")
     
    #set up sub kwargs
    skwargs=dict()
  
    if not args.haz_index is None:
        skwargs['run_coarse_wse_stack']=dict(haz_basin_index_fp=args.haz_index)
         
    if not args.processes is  None:
        skwargs['run_downscale_fines']=dict(processes=args.processes)
        skwargs['run_toWSH']=dict(processes=args.processes)
          
 
    #===========================================================================
    # setup loger
    #===========================================================================
    #retrieve logging level from logging module
    from haz.hp.argp import setup_logger
    root_logger = setup_logger(args)
 
    root_logger.debug(f'logger setup')
    #===========================================================================
    # execute
    #===========================================================================
          
    from haz.rim2019.downscale.pipe_fdsc_nuts3 import run_downscale_pipeline as func
       
    func(args.basinName2,
        out_dir=args.out_dir,
        skwargs= skwargs,
        #log_level=log_level,
        log=root_logger,
        skip_l = args.skip_l
        )
       
    try:
        winsound.Beep(440, 500)
    except:
        pass
    
