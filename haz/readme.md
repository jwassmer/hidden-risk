# analysis/prep of flood hazard rasters
configured like a standalone project

see ./rim2019/readme.md for description


## installation
expects a ./definition.py file similar to the below

see ./environment.yml for conda build

the downscaling relies on the FloodDownscaler2 project (as a submodule). typically on the 2307_roads branch. 
`git submodule add -b 2307_roads --name FloodDownscaler2 https://github.com/cefect/FloodDownscaler2.git`

this submodule requires whitebox-tools v2.2.0 (as a sub-sub module). see ./FloodDownscaler2/README.md for instructions setting this up.

repo uses git LFS

### PYTHONPATH
replace PROJECT_DIR_NAME with the path to your repo. The last directory is created by building whitebox-tools.
```
PROJECT_DIR_NAME
PROJECT_DIR_NAME\FloodDownscaler2\
PROJECT_DIR_NAME\FloodDownscaler2\whitebox-tools
PROJECT_DIR_NAME\FloodDownscaler2\whitebox-tools\target\release 
```

### definition.py
adjust the below to match your build

```
"""machine specific definitions"""

import os, sys

from parameters import src_dir

# default working directory
wrk_dir = r'l:\10_IO\2307_roads' 

#location of test data
test_data_dir = os.path.join(wrk_dir, 'test')

#whitebox exe location
wbt_dir = r'l:\09_REPOS\05_FORKS\whitebox-tools\target\release'

#===============================================================================
# data and definitions---------
#===============================================================================


#location of RIM 2019 results librarires
asc_lib_d = {
    'donau':'donau',  # smallest
    'rhine':'Rhine',
    'weser':'weser',
    'elbe':'elbe', #differnet format...missing lookup table
    'ems':'ems'
    }

asc_lib_d = {k:os.path.join(wrk_dir, r'ins\2019_RIM', v) for k, v in asc_lib_d.items()}

#discretizations
zones_fp=os.path.join(src_dir, r'data_LFS\haz\rim2019\zones\nuts\NUTS_3_DE_20231009.gpkg')
basins_fp = os.path.join(src_dir,r'data_LFS\haz\rim2019\zones\analysis_basins\NUTS_hydro_divisions_1010.gpkg')



#RIM 2019 model DEMS
coarse_dem_fp_d = {k:os.path.join(r'l:\10_IO\2307_roads\ins\2019_RIM\burned_domains', k+'.tif') for k in asc_lib_d.keys()}
 
#===============================================================================
# downscaling---------
#===============================================================================
#===============================================================================
# fine DEM
#===============================================================================
 
#DGM index
dgm5_index_fp=r'l:\10_IO\2307_roads\ins\DEM\DGM5\index\dgm5_kacheln_20x20km_laea_4647.gpkg'

#faw file search directory
#dgm5_search_dir=r'\\rzv230c.gfz-potsdam.de\hydro_public\DATEN\GIS\DGM_DEM\DGM5_Germany\dgm5\laea\ascii'
dgm5_search_dir=r'e:\05_DATA\Germany\BKG\DGM5\20230717\raw'
```

## Tests
pytests are located in ./haz/tests
FloodDownscaler2 submodule has it's own pytests