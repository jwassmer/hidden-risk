# %%
import sys

# sys.path.append("/home/jonas/Code/germanyFloods")

import os  ##xw

from germanyFloods import readFiles as rf
import subprocess


path = "germanyFloods/data/"
# %%
# run locally!!
sync_command = f"rsync --exclude '*.pkl' -av -e ssh jonaswa@hpc.pik-potsdam.de:road-networks/{path} {path}"
os.system(sync_command)
# %%
access_gdf = rf.read_hospital_catchement3(
    catchment="all", gamma=0.08, population_kwd="prob_service_population"
)
# access_gdf = access_gdf[access_gdf["population_pre_flood"] > 1_000]

# %%
# hostpital_df = rf.aggregate_hospitals(access_gdf)
access_gdf.sort_values(by=["population_percentual_diff", "min_dist"], inplace=True)
access_gdf.dropna(subset=["population_percentual_diff"], inplace=True)


# %%

N = 20
# drop double hospitals
access_no_duplicates = access_gdf.drop_duplicates(
    subset=["amenity_name"], inplace=False, keep="last"
)
# tail = access_no_duplicates.tail(10)
# tail
tail = access_gdf.tail(N)
tail
# %%
catchments = [
    "ems",
    "rhine_lower",
    "rhine_upper",
    "elbe_lower",
    "weser",
    "elbe_upper",
    "donau",
]

# %%

for catchment, event in tail[["catchment", "event"]].values:
    print(catchment, event)
    scp_path = f"jonaswa@hpc.pik-potsdam.de:road-networks/{path}/{catchment}/{event}/graph_r.pkl"
    local_path = f"{path}/{catchment}/{event}/"

    os.system(f"scp -r {scp_path} {local_path}")
    # subprocess.run(f"scp -r {scp_path} {local_path}", shell=True)
# %%
for catchment in catchments:
    print(catchment)
    scp_path = f"jonaswa@hpc.pik-potsdam.de:road-networks/{path}/{catchment}/graph.pkl"
    local_path = f"{path}/{catchment}/"

    os.system(f"scp -r {scp_path} {local_path}")
    # subprocess.run(f"scp -r {scp_path} {local_path}", shell=True)
# %%
