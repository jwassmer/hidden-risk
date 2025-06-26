
:: run pytest suite on all tests in the directory

:: DIRECTIONS
:: set the test_dir
:: add additional test paths if desired

:: VARIABLES
SET TEST_DIR=%~dp0
set maxfail=10

:: activate the environment
call %~dp0..\..\env\conda_activate.bat
 
:: call pytest

REM python -m pytest --maxfail=10 --verbosity=0 --capture=tee-sys test_downscale_fdsc.py test_downscale_fines.py test_downscale_wse.py

ECHO starting tests in separate windows
start cmd.exe /k python -m pytest --maxfail=%maxfail% --verbosity=0 --capture=tee-sys test_downscale_fdsc.py
start cmd.exe /k python -m pytest --maxfail=%maxfail% --verbosity=0 --capture=tee-sys test_downscale_fines.py
start cmd.exe /k python -m pytest --maxfail=%maxfail% --verbosity=0 --capture=tee-sys test_downscale_wse.py

:: submodule tests
start cmd.exe /k python -m pytest --maxfail=%maxfail% --verbosity=0 --capture=tee-sys %SRC_DIR%/FloodDownscaler2

:: cmd.exe /k