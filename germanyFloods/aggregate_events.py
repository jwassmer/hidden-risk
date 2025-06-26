# %%
import pickle
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import geopandas as gpd
import os
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import pandas as pd


# from src import Plotting as pl
from src import FloodRaster as fr
from src import EffectiveVelocities2 as ev
from germanyFloods import readFiles as rf

# pl.mpl_params(fontsize=16)
path = "germanyFloods/data"
# path_100m = "data_LFS/haz/rim2019/03_rasters"

gamma = 0.08

# %%
access_gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=gamma
)


# %%

access_gdf.sort_values(by=["population_percentual_diff", "min_dist"], inplace=True)

access_no_duplicates = access_gdf.drop_duplicates(
    subset=["node"], inplace=False, keep="last"
)

tail = access_no_duplicates[access_no_duplicates["population_percentual_diff"] > 0.3]
tail.to_csv(f"{path}/hospitals_greater_0_3.csv")
# tail

# %%

catchments = tail["catchment"].values
events = tail["event"].values
hospital_names = tail["amenity_name"].values
osmids = tail["node"].values

for j, (catchment, event, hospital_name, osmid) in enumerate(
    zip(catchments, events, hospital_names, osmids)
):
    print(j, catchment, event, hospital_name)
    path_hospital_pre_flood = (
        f"{path}/{catchment}/gamma_{gamma}/hospital-access.geojson"
    )
    path_hospital_post_flood = (
        f"{path}/{catchment}/{event}/gamma_{gamma}/hospital-access.geojson"
    )
    path_pre_flood_graph = f"{path}/{catchment}/graph.pkl"
    path_graph = f"{path}/{catchment}/{event}/graph_r.pkl"

    with open(path_pre_flood_graph, "rb") as f:
        G_pre = pickle.load(f)

    G_pre = ev.effective_spillover_velocities(G_pre, gamma=gamma)

    with open(path_graph, "rb") as f:
        G = pickle.load(f)

    G = ev.effective_spillover_velocities(G, gamma=gamma)

    nodes, edges = ox.graph_to_gdfs(G)
    nodes_pre, edges_pre = ox.graph_to_gdfs(G_pre)

    edges["spillover_load_r"] = edges_pre["spillover_load"]
    edges["spillover_travel_time_r"] = edges_pre["spillover_travel_time"]

    access_gdf = gpd.read_file(path_hospital_pre_flood)
    access_gdf["node"] = access_gdf["node"].astype(int).astype(object)
    access_gdf = access_gdf.set_index("node")

    access_gdf_r = gpd.read_file(path_hospital_post_flood)
    access_gdf_r["node"] = access_gdf_r["node"].astype(int).astype(object)
    access_gdf_r = access_gdf_r.set_index("node")

    access_gdf_r["population_pre_flood"] = access_gdf["population"]
    access_gdf_r.rename(columns={"population": "population_post_flood"}, inplace=True)
    access_gdf_r["population_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    )
    access_gdf_r["population_percentual_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    ) / access_gdf_r["population_pre_flood"]

    hospital = access_gdf.loc[osmid:osmid]
    hospital_r = access_gdf_r.loc[osmid:osmid]

    x, y = (
        hospital_r.geometry.centroid.x.values[0],
        hospital_r.geometry.centroid.y.values[0],
    )

    eps_x_metre = 0.5
    # golden_ratio = 1.61803398875
    eps_y_metre = eps_x_metre / 2
    xmin, ymin, xmax, ymax = (
        x - eps_x_metre,
        y - eps_y_metre,
        x + eps_x_metre,
        y + eps_y_metre,
    )

    nodes_cx = nodes.cx[xmin:xmax, ymin:ymax]
    nodes_pre_cx = nodes_pre.cx[xmin:xmax, ymin:ymax]
    edges_cx = edges.cx[xmin:xmax, ymin:ymax]

    # Create directories if they don't exist
    os.makedirs(
        f"{path}/{catchment}/{event}/gamma_{gamma}/{hospital_name}", exist_ok=True
    )
    nodes_cx.drop("voronoi", axis=1).to_file(
        f"{path}/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes.geojson"
    )
    nodes_pre_cx.drop("voronoi", axis=1).to_file(
        f"{path}/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes_pre.geojson"
    )
    edges_cx[
        [
            "spillover_load",
            "spillover_load_r",
            "spillover_travel_time",
            "spillover_travel_time_r",
            "removed",
            "geometry",
        ]
    ].to_file(f"{path}/{catchment}/{event}/gamma_{gamma}/{hospital_name}/edges.geojson")


# %%
