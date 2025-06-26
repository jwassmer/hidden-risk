:: extract rasters to netcdf library

call %~dp0..\..\..\..\env\conda_activate.bat

@echo off

set index_fp=l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_collect\nuts3_collect_meta_7_7_20240126.gpkg


set basinName2=elbe_lower elbe_upper rhine_lower ems donau weser rhine_upper
REM set basinName2=ems weser
REM set basinName2=elbe_lower elbe_upper rhine_lower donau rhine_upper

for %%b in (%basinName2%) do (
ECHO %%b
start cmd.exe /k python main_05_toRasters.py --basinName2 %%b --index_fp %index_fp%)

cmd.exe /k
