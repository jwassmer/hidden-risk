# rim2019 flood hazard processing

These scripts construct/post-process flood depth layers from Dung + Mostaffa's 2019/2020 RFM/RIM/SWIM 5000-year simulation runs (RFM2019) for use in our roads analysis.


 
## RFM2019
The RFM2019 simulations are best described in:
    Sairam, Nivedita, Fabio Brill, Tobias Sieg, Mostafa Farrag, Patric Kellermann, Viet Dung Nguyen, Stefan Lüdtke, et al. 2021. “Process-Based Flood Risk Assessment for Germany.” Earth’s Future 9 (10). https://doi.org/10.1029/2021EF002259.
    

Data from these simulations were retrieved from GFZ.hydrology server's  (hydro135\project_comprehensive_flood_risk_germany\RIM\Rim_data_pub\) following discussions w/ Nivedita and Dung.
Simulations (and resulting data) are divided into the five major catchments (donau, elbe, ems, rhine, weser).  
Each catchment was run continuously for 100-years with random weather in 50 batches (50x100 = 5000).
Results for 'inundation_depth' and 'duration' were written when overbank flooding was triggered with a rolling 10-day threshold to separate events (i.e., some years have many floods other years have no floods)
These results are stored as ~20,300 zipped asci files (~7GB).
Elbe simulations are structured slightly differently.

## Pre-Processing Pipeline
This includes two scripts for pre-processing and preparing the original RFM results .zip asc files for analysis:
- _01_extract_to_nc.py: RFM2019 'inundation_depth' results are unzipped per basin and extracted to a single netcdf per basin.
- _02_nc_to_sparse.py: extracted netcdf files are loaded using xarray/dask and converted to a 3D sparse array.


### Results
| River | RFM overtoping event count |
| Donau | 5670 |
| Elbe | 11413 |
| Ems | 121 |
| Rhine | 12971 |
| Weser | 1509 |

## Processing pipelines
see subfolder readmes for more detail

- **nuts3**: nuts3 realized extreme events selection
- **quant**: quantile depth rasters (discontinued)
- **downscale**: downscale rim2019 nuts-selected events (from 100m) to 5m DGM5


## known issues (with RFM data)
- some erroneus events were found in donau (12) and rhine (6) catchments. These are removed during pre-processing (see ./haz/rim2019/coms.py). There appears to still be 1 or more erroneous events for the rhine; however, we were not able to identify these so they remain in the stack (and are visible in the max quantile). Visual inspection suggests these remaining erroneous events have minimal impact on quantiles (other than the max). Some of these show an arc-like pattern of artifact flooding (maybe at some boundary?).
- many events only have 1 wet pixel (removed?)
- some events have no wet pixels  (removed?)
- basin (.\data_LFS\haz\meta\basin_polygons\rim2019_basins_merged_20230804.gpkg) and hydrauilc domains do not align (e.g., upper Rhine catchment)
- hydraulic domains do not align with German national boundary (e.g.,  Rhine at the border)


 