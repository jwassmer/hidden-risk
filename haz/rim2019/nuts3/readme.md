# rim2019 flood hazard extreme events selection (nuts3 realized)

Processing pipeline to identify, clip, and catalogue inundation events from the RFM2019 set.
This results in a set of *extreme* events per analysis basin where *extremeness* is determined from the ranked inundation within each nuts3 region.

## Updates
2023-10-23: removed some broken rasters. set a filter criteria to exclude events w/ <11 wet pixels. 


## RFM2019
see ../readme.md

## Pre-Processing Pipeline
see ../readme.md

## Processing Pipeline
Following pre-processing, the following scripts/steps are used:
- _03_nuts3_event_selec.py: The full 5000-year event stack is evaluated at each nuts3 region to rank the events with inundation within that nuts3 (by inundated cell count).
- _04_nuts3_collect.py: From the ranked events, the 5 most extreme events are identified for each nuts3. The union of these events for all nuts3 within an analysis basin is taken to establish the *analysis-basin event-set*. The event stack is then sliced and masked to the selected events and the basin boundaries. 
- _05_nuts3_rasters.py: The event stack is converted to GeoTIFFs and an index file is written.

## Main Inputs
### Nuts3 spatial discretization
EuroSTAT ADMINISTRATIVE UNITS / STATISTICAL UNITS
filtered to germany and nuts3

.\data_LFS\haz\rim2019\zones\nuts\NUTS_3_DE_20231009.gpkg

### Hydro Basins
Custom 7-basin split of major river basins in germany
./data_LFS\haz\rim2019\zones\analysis_basins\NUTS_hydro_divisions_1010.gpkg

## Outputs and Results
see /data_LFS/haz/rim2019/nuts3/readme.md

## known issues
some results rasters are empty/corrupted? 