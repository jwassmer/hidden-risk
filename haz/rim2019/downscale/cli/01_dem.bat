:: compile rasters for each hyd basin
call activate.bat


@echo off
 
 
python -O main_01_dem.py --dem_index_fp l:\10_IO\2307_roads\test\test_downscale_dem\dgm5_kacheln_20x20km_laea_4647_test.gpkg
 
cmd.exe /k
