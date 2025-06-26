:: extract rasters to netcdf library

call %~dp0..\..\..\env\conda_activate.bat

@echo off
 
:: NRC_Desktop = 60-100%, 70sec/layer
set basinName=donau
set n_workers=14
 
 
python -O main_quantile.py --basinName %basinName%  --n_workers %n_workers%  

 

cmd.exe /k
