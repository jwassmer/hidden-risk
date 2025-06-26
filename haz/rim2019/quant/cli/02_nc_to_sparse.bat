:: extract rasters to netcdf library

call %~dp0..\..\..\env\conda_activate.bat

@echo off
 
 
set basinName=donau
set threads_per_worker=7
set n_workers=2

ECHO running w/ basinName=%basinName%
python -O main_nctosparse.py --basinName %basinName% --threads_per_worker %threads_per_worker% --n_workers %n_workers%


cmd.exe /k
