# rim2019 flood hazard downscaler

pipeline to downscale rim2019 nuts-selected events (from 100m) to 5m DGM5 using a modified CostGrow downscaling algorithm.

For a description of the downscaling method, see .\FloodDownscaler2\README.md 
E
see ./cli/main_pipe.py for main cli

## UPDATES
- 2024 01 27: added **run_toWSH** to the pipeline to convert back to WSH. cleaned up some error handling to address missing events. re-ran on revised event selection.

## Pipeline Summary
1) run_coarse_wse_stack: build the coarse WSE from the DEM and WSH
2) run_chunk_to_dem_tile: prepare fine resolution input layers by spatially chunking to the DEM tile and CRS. outputs a DataSource w/ DEM, WSE, and WBDY
3) run_downscale_fines: multiprocess wrapper around CostGrow to downscale each chunk. outputs a WSE GeoTiFF per tile per raster_index
4) run_toWSH: convert the downscaled GeoTiff and join to the tiled DataSource for a complete stack (DEM, WSE, WSH, WBDY, WSEf, WSHf). Calc the WSH raster. 
5) run_organize: extract the downscaled WSH, reproject to the project CRS (4647), output a GeoTiff per tile, organized into sub-folders for each raster_index. Create a vrt for each raster_index.

see ./pipe_fdsc_nuts3 for more info


## Main Inputs
### haz_index
analysis basin index
links each basin to its (coarse) selected event stack DataSet
see haz.rim2019.nuts3._04_nuts3_collect() for production
see .\data_LFS\haz\rim2019\nuts3\readme.md for metadata


### dem_index
index of DEM tiles.
specified in definitions.py


### wbd_mask
rim2019 burned domain (which contains the burned channel areas)


## Outputs and Results
archive (created with 7zip) of outputs for each basin is uploaded here: [downscale_05org_20240127.zip](https://drive.google.com/drive/folders/1N1QWzD1BBZOxeX7inhYW9qjorecAgyze?usp=drive_link)

files are organized like:

- hydro_basin
  - raster_index: folder for this DGM5 tile
    - rim2019[...].tif: set of downscaled WSH GeoTiffs matching DGM5 tiling and resolution (reprojected to EPSG:4647)
  - meta_05org[...].csv: metadata for each GeoTiff (raster_index and tiles)
  - meta_05org[...].gpkg: similar to the above but spatial
  - rim2019[...].vrt: set of [Virtual Format](https://gdal.org/drivers/raster/vrt.html) rasters mosaicing together all tiles for a single raster_index (event). NOTE these have varying extents based on tile presence.


## Known Issues
- the meta_05org[...].gpkg square geometry matches the reprojected raster extents, slightly different than the data domain(s)
- due to the tiling, some floods are discontinous at tile boundaries
- inundation has been removed from the 'channel' domain of rim2019
- only tiles with >4 coarse/raw wet pixels are included
- many of the WSH GeoTiffs have a very low percentage of non-zero values, so the QGIS default styling (`Accuracy=Estimate`) may not behave as expected. 
- the .vrt mosaics may have some no-data artifacts at the boundaries of the tiles
- the hydro basin boundaries/divisions (e.g. elbe_lower vs. elbe_upper) are done based on political boundaries (nuts), not hydrological boundaries; leading to some discontinous flooding in some cases. 
- inundation in coastal areas should be ignored

### suspicous looking events found during spot checks
- rim2019_dgm5_wd_donau_01804_005m
- rim2019_dgm5_wd_elbe_upper_270059_005m
- rim2019_dgm5_wd_rhine_lower_11016_005m
- rim2019_dgm5_wd_rhine_lower_00482_005m (d/s boundary backwater)

 


