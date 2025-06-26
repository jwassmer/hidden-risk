'''
Created on Jan. 26, 2024

@author: cef
'''

import logging
import config

def setup_logger(args, **kwargs):
    """configure the root logger from argparser
    
    
    USEr
    ----------
    parser.add_argument('--log_level', type=str,default='INFO')
    
    from haz.hp.argp import setup_logger
    root_logger = setup_logger(args)
    
    
    """

    
    #===========================================================================
    # setup root logger
    #===========================================================================
    
    from haz.hp.basic import init_root_logger
    root_logger=init_root_logger(**kwargs)
    
    #===========================================================================
    # set level
    #===========================================================================
    # Retrieve logging level from logging module
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    
    #set this
    
    config.log_level = log_level
    
    log = logging.getLogger('r')
    log.setLevel(log_level)
    log.info(f'set log_level={config.log_level}')
    
    return log