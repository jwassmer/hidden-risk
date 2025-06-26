:: compile rasters for each hyd basin
call activate.bat

set HAZ_INDEX=l:\10_IO\2307_roads\outs\rim_2019\nuts3\04_collect\nuts3_collect_meta_7_7_20240126.gpkg

set SKIP_L=chunk_to_dem_tile
@echo off
 

:: --skip_l %SKIP_L%
REM python -O main_pipe.py --basinName2 ems --haz_index %HAZ_INDEX% --processes 3
python main_pipe.py --basinName2 ems --processes 1 --haz_index %HAZ_INDEX% --log_level INFO 

 
cmd.exe /k