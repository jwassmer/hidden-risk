# rim2019 flood hazard QUANTILE DEPTHS processing pipeline

(discontinued)

Processing pipeline to construct quantile depth rasters from RFM2019

This represents the probability of the shown depth being exceeded in a given year for each pixel according to RFM2019.

NOTE: the resulting rasters should **not** be confused with a flood map for the equivalent quantile (e.g., q=0.995 does not show a q=0.995 flood). Such a flood map would would use some larger aggregation (e.g., basin, gauge, or reach-based) to detrmine hazard probability relations, not the pixel-wise approach we take here. 

## RFM2019
see ../readme.md

## Pre-Processing Pipeline
see ../readme.md

## Quantile Processing pipeline
To construct the quantile maps for each catchment, the following steps were taken:
- reduce to annual maximum (03_annMax.bat): the simulation set is grouped by simulation year to determine the annual maximum
- calculate basin quantile map (04_quantile.bat): a set of quantiles (and the maximum) are calculated for each basin. 
- mosaic basins together: GeoTiffs are written for each individual quantile and a single netcdf file is written for the set. 

batch scripts for each step are contained in ./haz/rim2019/cli

## result files
The result of the pipeline is a 5-basin mosaic of the annual maximum flood depth for the given pixel-wise quantile (e.g., q=0.995). 
see /data_LFS/haz/rim2019/quant/readme.md

layers are loaded and stylized in the QGIS project file (/data_LFS/haz/rim2019/quant/rim2019_haz.qgz)

## known issues
see ../readme.md