:: extract rasters to netcdf library

call %~dp0..\..\..\env\conda_activate.bat

@echo off
 
:: NRC_Desktop = 60-100%, 70sec/layer
set basinName=rhine
set max_workers=7 

ECHO running w/ basinName=%basinName%
:: EXTRACTION------------
REM python -O main_extract_to_nc.py --basinName %basinName% --max_workers %max_workers%

:: to SPARSE-------------
set threads_per_worker=7
set n_workers=2
REM python -O main_nctosparse.py --basinName %basinName% --threads_per_worker %threads_per_worker% --n_workers %n_workers%

:: reduce to Anual Max-----------
set n_workers=14
python -O main_annMax.py --basinName %basinName%  --n_workers %max_workers% 


:: QUANTILE---------------------
python -O main_quantile.py --basinName %basinName%  --n_workers %max_workers%  

ECHO finished on %basinName%
cmd.exe /k
