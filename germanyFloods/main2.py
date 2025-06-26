# %%
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
print(sys.version)
sys.path.append("/home/jonaswa/road-networks/")

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from src import FloodRaster as fr
from src import RoadNetwork2 as rn
from src import GeoModule as gm
from src import SupportFunctions as sf
from src import EmergencyModule as em
from src import TrafficCentrality2 as tc
from src import EffectiveVelocities2 as ev
import os

path = "data_LFS/haz/rim2019/0303_downscale_20240629"

threads = 12
buffer_km = 75


basin = "all"
event = "all"


try:
    k = int(sys.argv[1])
except:
    k = 0


# %%


def get_tifs_of_event(basin, event, concat_df, path=path):
    selected_event_df = concat_df.loc[basin, event]
    len_tifs = len(selected_event_df)
    print(f"Selected event has {len_tifs} tifs files.")
    tif_files = []
    for index, row in selected_event_df.iterrows():
        rel_dir = row["rel_dir"]
        file_name = row["wsh_fdsc_fn"]
        tif_files.append(f"{path}/{basin}/{rel_dir}/{file_name}")

    return tif_files


# %%

concat_df = pd.read_csv(
    f"{path}/concat_downscale_wsh_index_df.csv", index_col=[0, 5]
).sort_index()

try:
    event_df = concat_df.loc[basin, event]
    unique_events = event_df.index.unique()
    print("There are ", len(unique_events), " unique events.")
except:
    try:
        basin_df = concat_df.xs(basin, level=0)
        unique_events = basin_df.index.unique()
        print(f"There are ", len(unique_events), f"unique events in basin: {basin}.")
        event = unique_events[k]
    except:
        unique_events = concat_df.index.unique()
        print("There are ", len(unique_events), " unique events.")
        event = unique_events[k][1]
        basin = unique_events[k][0]


# %%


def _create_regions(
    buffer_km=buffer_km,
    threads=threads,
    basin=basin,
    event=event,
    concat_df=concat_df,
    path=path,
):

    print("Selected basin: ", basin)
    print("Selected event index: ", event)

    tif_files = get_tifs_of_event(basin, event, concat_df, path=path)
    path_to_basin = f"germanyFloods/data/{basin}"
    path_to_event = f"germanyFloods/data/{basin}/{event}"

    os.makedirs(path_to_basin, exist_ok=True)
    os.makedirs(path_to_event, exist_ok=True)

    basin_gdf = gpd.read_file(
        "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
    ).to_crs("EPSG:4326")

    current_basin_gdf = basin_gdf[basin_gdf["name"] == basin]
    buffered_basin = gm.buffer_gdf(current_basin_gdf, buffer_km)

    driving_tags = [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        # "residential",
    ]

    region = rn.RoadNetwork(
        osmpbf="ger-buffered-200km.osm.pbf",
        highway_filter=f"w/highway={','.join(driving_tags)}",
        gdf=buffered_basin,
    )
    key, tag = "amenity", "hospital"
    region.add_pois(key, tag, merge_km=0)

    region.loads("travel_time", threads=threads)
    ######################################################

    region_r = region.copy()

    hospital_nodes = region.nodes[region.nodes[key] == tag]
    hospital_nodes.loc[:, "min_dist"] = np.inf
    for tif_file in tif_files:
        print(tif_file)
        raster = fr.read_raster(tif_file)
        bounds = raster.rio.bounds()

        # Create a polygon from the bounds
        # The bounds are returned as (left, bottom, right, top)
        boundPoly = Polygon(
            [
                (bounds[0], bounds[1]),
                (bounds[0], bounds[3]),
                (bounds[2], bounds[3]),
                (bounds[2], bounds[1]),
            ]
        )

        region_r.water_depth(raster, polygon=boundPoly)
        wet_roads = region_r.edges[region_r.edges["water_depth"] > 0.3].index
        print(f"There are {len(wet_roads)} wet roads")

        min_dist_list = hospital_nodes.geometry.apply(
            lambda pt: em.min_dist_to_raster(pt, raster)
        )
        hospital_nodes.loc[:, "min_dist"] = np.minimum(
            hospital_nodes["min_dist"], min_dist_list
        )

    wet_roads = region_r.edges[region_r.edges["water_depth"] > 0.3].index
    print(f"There are {len(wet_roads)} wet roads")
    region_r.remove_edges(wet_roads)
    region_r.loads("travel_time", threads=threads)
    ##################################################################

    region_r.nodes["min_dist"] = hospital_nodes["min_dist"]

    print("Saving graphs to disk...")
    graph_path = os.path.join(path_to_basin, "graph.pkl")
    with open(graph_path, "wb") as graph_file:
        print("Saving graph to ", graph_path)
        pickle.dump(region.graph, graph_file)

    graph_r_path = os.path.join(path_to_event, "graph_r.pkl")
    with open(graph_r_path, "wb") as graph_r_file:
        print("Saving graph_r to ", graph_r_path)
        pickle.dump(region_r.graph, graph_r_file)

    return region, region_r


def main(gamma, region, region_r):
    region.effective_spillover_velocities(gamma)

    # g = region.graph

    access_gdf = em.get_access_gdf2(region)

    region_r.effective_spillover_velocities(gamma)

    access_gdf_r = em.get_access_gdf2(region_r)

    hospital_nodes = region_r.nodes[region_r.nodes["amenity"] == "hospital"]
    hospital_nodes["location_x"] = hospital_nodes.geometry.x
    hospital_nodes["location_y"] = hospital_nodes.geometry.y

    access_gdf_r.loc[:, "min_dist"] = hospital_nodes["min_dist"]
    access_gdf_r["hospital_location_x"] = hospital_nodes["location_x"]
    access_gdf_r["hospital_location_y"] = hospital_nodes["location_y"]

    access_gdf_r["population_percentual_diff"] = (
        access_gdf_r["prob_service_population"] - access_gdf["prob_service_population"]
    ) / access_gdf["prob_service_population"]

    return access_gdf, access_gdf_r


# %%

region, region_r = _create_regions()

gammas = [0, 0.02, 0.04, 0.06, 0.08, 0.1]

for gamma in gammas:
    print(f"Gamma: {gamma}")
    event_path = f"germanyFloods/data/{basin}/{event}/gamma_{gamma}"
    basin_path = f"germanyFloods/data/{basin}/gamma_{gamma}"

    os.makedirs(event_path, exist_ok=True)
    os.makedirs(basin_path, exist_ok=True)

    access_gdf, access_gdf_r = main(gamma, region, region_r)

    access_gdf.to_file(
        f"{basin_path}/hospital-access.geojson",
        driver="GeoJSON",
    )
    access_gdf_r.to_file(
        f"{event_path}/hospital-access.geojson",
        driver="GeoJSON",
    )

# %%
