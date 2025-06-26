"""
Created on Jul. 26, 2023

@author: cefect
"""
import os, logging, pprint, webbrowser, sys
import logging.config
from datetime import datetime
 
from definitions import wrk_dir
from parameters import log_format_str
import config
 



# ===============================================================================
# logging
# ===============================================================================




today_str = datetime.now().strftime("%Y%m%d")

#===============================================================================
# pandas settings
#===============================================================================
import pandas as pd

# Set the maximum number of rows to be displayed to 10
pd.set_option('display.max_rows', 10)


# ===============================================================================
# loggers-----------
# ===============================================================================



def init_root_logger(
    log_dir=wrk_dir,
    #log_level=None,
    logcfg_file=None,
    ):
    """initilze the root logger from teh config file
    
    typically, we use a 4 handler logger
        <StreamHandler <stdout> (INFO)>
        <FileHandler l:\10_IO\2307_roads\root.log (DEBUG)>
        <FileHandler l:\10_IO\2307_roads\rootwarn.log (WARNING)>
        <StreamHandler <stderr> (WARNING)>
        
    sometimes we also add a DEBUG FileHandler into the outputs directory
        
    this is nice for handling complex multi-part runs
    
    Params
    ---------
    log_level: int, optional
        over-ride the logging level on the StreamHandler
        
        
    
    """
    if logcfg_file is None:
        from parameters import logcfg_file
        
    assert os.path.exists(logcfg_file)
    

    logging.config.fileConfig(
        logcfg_file,
        defaults={"logdir": str(log_dir).replace("\\", "/")},
        disable_existing_loggers=True,
        )   
    
    logger = logging.getLogger()  # get the root logger
    
    #===========================================================================
    # #overwrite the log level
    #===========================================================================
    """this enables a lot of unwanted logging"""
 #==============================================================================
 #    if log_level is None:
 #        log_level = config.log_level
 # 
 #    if not log_level is None:
 #        for h in logger.handlers:
 #            #print(h)
 #            if h.name=='consoleHandler':
 #                h.setLevel(log_level)
 #                print(f'overide log_level={log_level} on {h.name}')
 #==============================================================================
 
 
    #===========================================================================
    # wrap
    #===========================================================================
    logger.info(
        f"root logger initiated and configured from file: {logcfg_file}\n    logdir={log_dir}"
    )
    logger.debug('\n    '+'\n    '.join([str(h) for h in logger.handlers]))

    return logger


def get_new_file_logger(
    name="r",
    level=logging.DEBUG,
    fp=None,  # file location to log to
    logger=None,
    ):
    # ===========================================================================
    # configure the logger
    # ===========================================================================
    if logger is None:
        logger = logging.getLogger(name)
        logger.setLevel(level)

    # ===========================================================================
    # configure the handler
    # ===========================================================================
    assert fp.endswith(".log")

    formatter = logging.Formatter(log_format_str)
    handler = logging.FileHandler(
        fp, mode="w"
    )  # Create a file handler at the passed filename
    handler.setFormatter(formatter)  # attach teh formater object
    handler.setLevel(level)  # set the level of the handler

    logger.addHandler(handler)  # attach teh handler to the logger

    logger.debug("built new file logger  here \n    %s" % (fp))

    return logger


#===============================================================================
# def init_log(log_dir=wrk_dir, log_level=None, **kwargs):
#     """wrapper to setup the root loger and create a file logger"""
# 
#     root_logger = init_root_logger(log_dir=log_dir, log_level=log_level)
#     
#     root_logger.debug('root_logger')
# 
#     # set up the file logger
#     return get_new_file_logger(logger=root_logger, **kwargs)
#===============================================================================


def get_log_stream(name=None, level=None):
    """get a logger with stream handler"""
    if name is None:
        name = str(os.getpid())
    if level is None:
        level = config.log_level
        #=======================================================================
        # if __debug__:
        #     level = logging.DEBUG
        # else:
        #=======================================================================
            

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # see if it has been configured
    if not logger.handlers:
        
        handler = logging.StreamHandler(
            stream=sys.stdout,  # send to stdout (supports colors)
        )  # Create a file handler at the passed filename
        formatter = logging.Formatter(log_format_str)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        
        logger.addHandler(handler)
    return logger


# ===============================================================================
# MISC-----------
# ===============================================================================
def dstr(
    d,
    width=100,
    indent=0.3,
    compact=True,
    sort_dicts=False,
):
    return pprint.pformat(
        d, width=width, indent=indent, compact=compact, sort_dicts=sort_dicts
    )


def view(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser

    # import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        # type(f)
        df.to_html(buf=f)

    webbrowser.open(f.name)

def get_tqdm_disable():
    if config.log_level==logging.INFO:
        return False
    else:
        return True

# ===============================================================================
# files/folders---------
# ===============================================================================


def get_temp_dir(temp_dir_sfx=r"py\temp"):
    from pathlib import Path

    homedir = str(Path.home())
    temp_dir = os.path.join(homedir, temp_dir_sfx)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    return temp_dir


def get_directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024 / 1024


def get_fp(search_dir, ext=".nc"):
    """search a directory for files with the provided extension. return the first result"""
    assert os.path.exists(search_dir), search_dir
    find_l = list()
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(ext):
                find_l.append(os.path.join(root, file))

    assert (
        len(find_l) == 1
    ), f"failed to get uinique match {len(find_l)} from \n    {search_dir}"
    return find_l[0]



def _filesearch(search_dir, ext='.asc'):
    fns = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(ext):
                fns.append(os.path.join(root, file))
                
    assert len(fns)==1, f'failed to get file from {search_dir}'
    
    return fns.pop(0)


def _filesearch_all(search_dir, ext='.asc'):
    fns = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(ext):
                fns.append(os.path.join(root, file))
                
 
    
    return fns










