:: compile rasters for each hyd basin
call activate.bat

set HAZ_INDEX=f:\Users\bryant\LS\10_IO\2307_roads\outs\rim_2019\nuts3\04_nuts3_collect\nuts3_collect_meta_7_6_20231023.gpkg

@echo off
 


set basinName2=elbe_lower elbe_upper rhine_lower ems donau weser rhine_upper

for %%b in (%basinName2%) do (
ECHO %%b
start cmd.exe /k python -O main_pipe.py --basinName2 %%b --haz_index %HAZ_INDEX%

)

 
cmd.exe /k