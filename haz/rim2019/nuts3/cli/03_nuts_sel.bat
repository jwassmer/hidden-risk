:: extract rasters to netcdf library

call activate.bat

@echo off
 
 
set basinName=donau weser ems rhine elbe
 
 
for %%B in (%basinName%) do (
start cmd.exe /k python -O main_03_nuts3.py --basinName %%B
)
 
cmd.exe /k
