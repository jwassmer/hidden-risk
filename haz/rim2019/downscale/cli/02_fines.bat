:: prepare fine resolution input layers by spatially clipping to the DEM tile.
call activate.bat


:: VARS
set min_wet_cnt=5
set use_cache=TRUE

@echo off
 


:: set basinName2=elbe_lower elbe_upper rhine_lower ems donau weser rhine_upper
set basinName2=ems

for %%b in (%basinName2%) do (
ECHO %%b
start cmd.exe /k python -O main_fines.py --basinName2 %%b --min_wet_cnt %min_wet_cnt% --use_cache %use_cache%

)

 
cmd.exe /k