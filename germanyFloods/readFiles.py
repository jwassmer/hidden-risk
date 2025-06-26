# %%
import pandas as pd
import os

os.environ["GEOPANDAS_IO_ENGINE"] = "fiona"
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import geopandas as gpd
import osmnx as ox

from src import FloodRaster as fr
from src import Plotting as pl

# pl.mpl_params(fontsize=16)


# %%
def quantile(x, q=0.9):
    return x.quantile(q)


def pos_sum(x):
    return np.sum(x[x >= 0])


def pos_mean(x):
    return np.mean(x[x >= 0])


def neg_sum(x):
    return np.sum(x[x < 0])


def neg_mean(x):
    return np.mean(x[x < 0])


def hospital_access_pre_flood(
    path="data_LFS/haz/rim2019/0303_downscale_20240629", basin="ems"
):
    if basin == "all":
        basin_list = [
            "ems",
            "rhine_upper",
            "elbe_lower",
            "rhine_lower",
            "weser",
            "elbe_upper",
            "donau",
        ]
    else:
        if type(basin) == list:
            basin_list = basin
        else:
            basin_list = [basin]

    gdf_list = []
    for basin in basin_list:
        catchment_path = os.path.join(path, basin)
        print(basin)
        access_gdf = gpd.read_file(
            os.path.join(catchment_path, "hospital-access.geojson")
        )

        access_gdf["basin"] = basin

        gdf_list.append(access_gdf)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # change type of "node" col to object
    combined_gdf["node"] = combined_gdf["node"].astype(int).astype(object)

    new_noname = combined_gdf[combined_gdf["amenity_name"].isna()]["node_map"]
    new_noname = new_noname.apply(lambda x: f"{basin}_NoName_{x}")

    # rename all hospitals with no name to their respective node_map
    combined_gdf["amenity_name"] = combined_gdf["amenity_name"].fillna(new_noname)
    return combined_gdf


def read_hospital_gdf(
    path="data_LFS/haz/rim2019/0303_downscale_20240629", basin="ems", event="all"
):
    if basin == "all":
        basin_list = [
            "ems",
            "rhine_upper",
            "elbe_lower",
            "rhine_lower",
            "weser",
            "elbe_upper",
            "donau",
        ]
    else:
        if type(basin) == list:
            basin_list = basin
        else:
            basin_list = [basin]

    gdf_list = []
    for basin in basin_list:
        catchment_path = os.path.join(path, basin)
        if event == "all":
            events = [
                event
                for event in os.listdir(catchment_path)
                if os.path.isdir(os.path.join(catchment_path, event))
            ]
        else:
            if type(event) != list:
                events = [event]
            else:
                events = event

        print(basin)
        access_gdf = gpd.read_file(
            os.path.join(catchment_path, "hospital-access.geojson")
        )
        for l in tqdm(range(len(events))):
            event_path = os.path.join(catchment_path, events[l])

            access_gdf_r = gpd.read_file(
                os.path.join(event_path, "hospital-access.geojson")
            )

            access_gdf_r["population_pre_flood"] = access_gdf["population"]
            access_gdf_r.rename(
                columns={"population": "population_post_flood"}, inplace=True
            )

            access_gdf_r = access_gdf_r.dropna(axis=1, how="all")
            # with open(os.path.join(event_path, "meta.json")) as f:
            #    meta = json.load(f)

            access_gdf_r["event"] = events[l]
            access_gdf_r["basin"] = basin

            gdf_list.append(access_gdf_r)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # change type of "node" col to object
    combined_gdf["node"] = combined_gdf["node"].astype(int).astype(object)

    new_noname = combined_gdf[combined_gdf["amenity_name"].isna()]["node_map"]
    new_noname = new_noname.apply(lambda x: f"{basin}_NoName_{x}")

    # rename all hospitals with no name to their respective node_map
    combined_gdf["amenity_name"] = combined_gdf["amenity_name"].fillna(new_noname)

    combined_gdf["population_percentual_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    ) / combined_gdf["population_pre_flood"]
    combined_gdf["population_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    )
    combined_gdf[combined_gdf["min_dist"].isna()]["min_dist"] = 0
    combined_gdf.dropna(inplace=True, subset=["population_percentual_diff"])
    combined_gdf.sort_values("population_percentual_diff", inplace=True)
    return combined_gdf


def read_hospital_catchement2(
    path="germanyFloods/data",
    catchment="ems",
    events="all",
    population_kwd="prob_service_population",
):
    if catchment == "all":
        catchment_list = [
            "ems",
            "rhine_upper",
            "elbe_lower",
            "rhine_lower",
            "weser",
            "elbe_upper",
            "donau",
        ]
    else:
        if type(catchment) == list:
            catchment_list = catchment
        else:
            catchment_list = [catchment]
    gdf_list = []

    concat_df = pd.read_csv(
        os.path.join(path, "concat_downscale_wsh_index_df.csv"), index_col=[0, 5]
    ).sort_index()

    for catchment in catchment_list:
        print(catchment)

        basin_concat_df = concat_df.xs(catchment, level=0)

        catchment_path = os.path.join(path, catchment)
        try:
            access_gdf = gpd.read_file(
                os.path.join(catchment_path, "hospital-access.geojson")
            )
            access_gdf["node"] = access_gdf["node"].astype(int).astype(object)

            access_gdf = access_gdf.set_index("node")
        except:
            print(f"Could not open {catchment_path}. Skipping catchment {catchment}.")
            continue

        if events == "all":
            unique_events = basin_concat_df.index.unique()
        else:
            unique_events = events

        for event in tqdm(unique_events):

            event_path = os.path.join(catchment_path, str(event))
            try:
                access_gdf_r = gpd.read_file(
                    os.path.join(event_path, "hospital-access.geojson")
                )
                access_gdf_r["node"] = access_gdf_r["node"].astype(int).astype(object)
                access_gdf_r = access_gdf_r.set_index("node")

                access_gdf_r["population_pre_flood"] = access_gdf[population_kwd]
                access_gdf_r.rename(
                    columns={population_kwd: "population_post_flood"}, inplace=True
                )

                access_gdf_r = access_gdf_r.dropna(axis=1, how="all")

                access_gdf_r["event"] = event
                access_gdf_r["catchment"] = catchment

                gdf_list.append(access_gdf_r)
            except:
                print(f"Could not open {event_path}.")

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=False))

    # change type of "node" col to object
    combined_gdf["node"] = combined_gdf.index.astype(int).astype(object)

    new_noname = combined_gdf[combined_gdf["amenity_name"].isna()]["node_map"]
    new_noname = new_noname.apply(lambda x: f"{catchment}_NoName_{x}")

    # rename all hospitals with no name to their respective node_map
    combined_gdf["amenity_name"] = combined_gdf["amenity_name"].fillna(new_noname)

    combined_gdf["population_percentual_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    ) / combined_gdf["population_pre_flood"]
    combined_gdf["population_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    )
    combined_gdf["min_dist"] = combined_gdf["min_dist"].fillna(0)
    combined_gdf.dropna(inplace=True, subset=["population_percentual_diff"])
    return combined_gdf


def read_hospital_catchement3(
    path="germanyFloods/data",
    gamma=0.08,
    catchment="ems",
    events="all",
    population_kwd="prob_service_population",
):
    if catchment == "all":
        catchment_list = [
            "ems",
            "rhine_upper",
            "elbe_lower",
            "rhine_lower",
            "weser",
            "elbe_upper",
            "donau",
        ]
    else:
        if type(catchment) == list:
            catchment_list = catchment
        else:
            catchment_list = [catchment]
    gdf_list = []

    concat_df = pd.read_csv(
        os.path.join(path, "concat_downscale_wsh_index_df.csv"), index_col=[0, 5]
    ).sort_index()

    for catchment in catchment_list:
        print(catchment)

        basin_concat_df = concat_df.xs(catchment, level=0)

        catchment_path = os.path.join(path, catchment)
        gamma_path = os.path.join(catchment_path, f"gamma_{gamma}")

        try:
            access_gdf = gpd.read_file(
                os.path.join(gamma_path, "hospital-access.geojson")
            )
            access_gdf["node"] = access_gdf["node"].astype(int).astype(object)

            access_gdf = access_gdf.set_index("node")
        except:
            print(f"Could not open {gamma_path}. Skipping catchment {catchment}.")
            continue

        if events == "all":
            unique_events = basin_concat_df.index.unique()
        else:
            unique_events = events

        for event in tqdm(unique_events):

            event_path = os.path.join(catchment_path, str(event))
            gamma_event_path = os.path.join(event_path, f"gamma_{gamma}")
            try:
                access_gdf_r = gpd.read_file(
                    os.path.join(gamma_event_path, "hospital-access.geojson")
                )
                access_gdf_r["node"] = access_gdf_r["node"].astype(int).astype(object)
                access_gdf_r = access_gdf_r.set_index("node")

                access_gdf_r["population_pre_flood"] = access_gdf[population_kwd]
                access_gdf_r.rename(
                    columns={population_kwd: "population_post_flood"}, inplace=True
                )

                access_gdf_r = access_gdf_r.dropna(axis=1, how="all")

                access_gdf_r["event"] = event
                access_gdf_r["catchment"] = catchment

                gdf_list.append(access_gdf_r)
            except:
                print(f"Could not open {gamma_event_path}.")

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=False))

    # change type of "node" col to object
    combined_gdf["node"] = combined_gdf.index.astype(int).astype(object)

    new_noname = combined_gdf[combined_gdf["amenity_name"].isna()]["node_map"]
    new_noname = new_noname.apply(lambda x: f"{catchment}_NoName_{x}")

    # rename all hospitals with no name to their respective node_map
    combined_gdf["amenity_name"] = combined_gdf["amenity_name"].fillna(new_noname)

    combined_gdf["population_percentual_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    ) / combined_gdf["population_pre_flood"]
    combined_gdf["population_diff"] = (
        combined_gdf["population_post_flood"] - combined_gdf["population_pre_flood"]
    )
    combined_gdf["min_dist"] = combined_gdf["min_dist"].fillna(0)
    combined_gdf["population_percentual_diff"] = combined_gdf[
        "population_percentual_diff"
    ].fillna(0)
    # combined_gdf.dropna(inplace=True, subset=["population_percentual_diff"])
    combined_gdf.sort_values("population_percentual_diff", inplace=True)
    return combined_gdf


def aggregate_hospitals(gdf, col="population_percentual_diff", new_index="idxmax"):
    # gdf.reset_index(inplace=True)
    grouped = (
        gdf.groupby(gdf.index)[col]
        .agg(
            [
                "sum",
                "mean",
                "std",
                "median",
                "var",
                "min",
                "max",
                new_index,
                ("q50", lambda x: quantile(x, q=0.5)),
                ("q75", lambda x: quantile(x, q=0.75)),
                ("q90", lambda x: quantile(x, q=0.9)),
                ("q99", lambda x: quantile(x, q=0.99)),
            ]
        )
        .set_index(new_index)
    )
    hospital_stats_gdf = grouped.join(gdf).set_geometry("geometry")

    hospital_stats_gdf.sort_values(by=new_index, inplace=True)

    return hospital_stats_gdf


def group_hospitals(gdf, col="population_percentual_diff"):
    grouped = gdf.groupby("node")[col].agg(
        ["sum", "mean", "std", "median", "var", "min", "max"]
    )
    return grouped


# %%
