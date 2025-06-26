:: extract rasters to netcdf library

call %~dp0..\..\..\env\conda_activate.bat

@echo off
 
:: NRC_Desktop = 60-100%, 70sec/layer
set basinName=weser
set max_workers=7 

ECHO running w/ basinName=%basinName%
python -O main_extract_to_nc.py --basinName %basinName% --max_workers %max_workers%


cmd.exe /k
