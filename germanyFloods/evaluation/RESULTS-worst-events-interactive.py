# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray as rxr
import folium
from shapely.geometry import Polygon
import os

from src import FloodRaster as fr
from src import Plotting as pl
from germanyFloods import readFiles as rf

# pl.mpl_params(fontsize=22)


def create_cross(point, size=1e-3):

    x, y = point.x, point.y
    cross_coords = [
        (x - size, y + 3 * size),
        (x + size, y + 3 * size),
        (x + size, y + size),
        (x + 3 * size, y + size),
        (x + 3 * size, y - size),
        (x + size, y - size),
        (x + size, y - 3 * size),
        (x - size, y - 3 * size),
        (x - size, y - size),
        (x - 3 * size, y - size),
        (x - 3 * size, y + size),
        (x - size, y + size),
        (x - size, y + 3 * size),
    ]
    return Polygon(cross_coords)


def custom_cmap(cm, scale=9999):
    ncm = np.zeros((scale, 4))

    pcm = mpl.colors.ListedColormap(ncm)

    ncm = np.vstack((pcm(np.linspace(0, 1, 100)), cm(np.linspace(0, 1, 2000))))

    return mpl.colors.ListedColormap(ncm)


def hospital_event(path, tail, j, population_kwd="prob_service_population", gamma=0.08):

    raster_path = "data_LFS/haz/rim2019/0303_downscale_20240629"

    catchment = tail["catchment"].iloc[j]
    event = tail["event"].iloc[j]
    hospital_name = tail["amenity_name"].iloc[j]
    osmid = tail["node"].iloc[j]

    print(j, catchment, event, hospital_name)

    nodes = gpd.read_file(
        f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes.geojson"
    ).set_index("osmid")

    edges = gpd.read_file(
        f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/edges.geojson"
    )
    raster = fr.read_total_event(
        catchment,
        event,
        rescale=1 / 2,
        path=raster_path,  # bounds=nodes.total_bounds
    )
    access_gdf = gpd.read_file(
        f"{path}/{catchment}/gamma_{gamma}/hospital-access.geojson"
    )
    access_gdf["node"] = access_gdf["node"].astype(int).astype(object)
    access_gdf = access_gdf.set_index("node")

    access_gdf_r = gpd.read_file(
        f"{path}/{catchment}/{event}/gamma_{gamma}/hospital-access.geojson"
    )
    access_gdf_r["node"] = access_gdf_r["node"].astype(int).astype(object)
    access_gdf_r = access_gdf_r.set_index("node")

    all_indices = access_gdf_r.index.union(access_gdf.index)

    # Reindex GeoDataFrames, filling missing values with zeros
    access_gdf_r = access_gdf_r.reindex(all_indices)
    # Fill missing geometries in gdf1 with the centroid of the corresponding geometry in gdf2
    for index, row in access_gdf_r.iterrows():
        if pd.isna(row["geometry"]):
            access_gdf_r.loc[index, "geometry"] = access_gdf.loc[index, "geometry"]
    access_gdf_r["amenity_name"] = access_gdf["amenity_name"]
    access_gdf_r[population_kwd] = access_gdf_r[population_kwd].fillna(0)

    access_gdf_r["population_pre_flood"] = access_gdf[population_kwd]
    access_gdf_r.rename(columns={population_kwd: "population_post_flood"}, inplace=True)

    access_gdf_r["population_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    )
    access_gdf_r["population_percentual_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    ) / access_gdf_r["population_pre_flood"]

    # hospital = access_gdf[access_gdf["amenity_name"] == hospital_name]
    hospital_r = access_gdf_r.loc[osmid:osmid]
    hospital = access_gdf.loc[osmid:osmid]
    return nodes, edges, raster, access_gdf, access_gdf_r, hospital, hospital_r


# %%


def generate_folium_map(
    k=0, tail=None, population_kwd="prob_service_population", gamma=0.08
):
    # raster_path = "data_LFS/haz/rim2019/0303_downscale_20240629"
    path = "germanyFloods/data"
    if tail is None:
        tail = pd.read_csv(f"{path}/worst_9_events.csv")

    tail = tail.sort_values(by="population_percentual_diff", ascending=False)

    nodes, edges, raster, access_gdf, access_gdf_r, hospital, hospital_r = (
        hospital_event(path, tail, k, population_kwd=population_kwd, gamma=gamma)
    )

    event = tail["event"].iloc[k]

    bounds = nodes.total_bounds
    minlon, minlat, maxlon, maxlat = bounds

    bounds_polygon = Polygon(
        [
            (minlon, minlat),
            (minlon, maxlat),
            (maxlon, maxlat),
            (maxlon, minlat),
            (minlon, minlat),
        ]
    )

    # Create a GeoDataFrame for the polygon
    gdf_boundary = gpd.GeoDataFrame({"geometry": [bounds_polygon]})

    center_lat = minlat - ((minlat - maxlat) / 2)
    center_lon = minlon - ((minlon - maxlon) / 2)

    cmap = custom_cmap(mpl.cm.get_cmap("Blues"))
    norm = mpl.colors.Normalize(vmin=0.3, vmax=3)

    access_gdf_r = access_gdf_r.cx[minlon:maxlon, minlat:maxlat]

    hospital_nodes = nodes[nodes["amenity"] == "hospital"]

    merged = pd.merge(
        hospital_nodes.reset_index(), access_gdf_r, left_on="osmid", right_on="node"
    )[
        [
            "amenity_name_x",
            "population_pre_flood",
            "population_post_flood",
            "population_diff",
            "population_percentual_diff",
            "min_dist",
            "geometry_x",
        ]
    ].set_geometry(
        "geometry_x"
    )
    crs = merged.crs
    utm_crs = merged.estimate_utm_crs()
    merged = merged.to_crs(utm_crs)
    merged["geometry_x"] = merged["geometry_x"].apply(
        lambda x: create_cross(x, size=100)
    )
    merged = merged.to_crs(crs)

    removed_edges = edges[edges["removed"] == "1"]
    conncected_edges = edges[edges["removed"] != "1"]
    conncected_edges["diff_travel_time"] = (
        conncected_edges["spillover_travel_time_r"]
        - conncected_edges["spillover_travel_time"]
    )

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    # hospitals.explore(m=m, color="red")

    vmin, vmax = min(conncected_edges["diff_travel_time"]), max(
        conncected_edges["diff_travel_time"]
    )

    hospital_r.explore(m=m, color="black", opacity=0.2)
    hospital.explore(m=m, color="lightgrey", size=0.8)
    gdf_boundary.boundary.explore(m=m, color="black", opacity=0.5)

    raster_bounds = raster.rio.bounds()
    rminlon, rminlat, rmaxlon, rmaxlat = raster_bounds

    image_overlay = folium.raster_layers.ImageOverlay(
        image=cmap(raster.data[0]),
        bounds=[[rminlat, rminlon], [rmaxlat, rmaxlon]],
        mercator_project=True,
        interactive=True,
        opacity=0.85,
    )
    image_overlay.add_to(m)

    if len(removed_edges) > 0:
        removed_edges.explore(m=m, color="crimson", opacity=1)

    # f_merged = merged[(merged["min_dist"] < 50) | (merged["min_dist"].isna())]
    # n_merged = merged[merged["min_dist"] >= 50]

    # n_merged.explore(m=m, color="orangered")
    # if len(f_merged) > 0:
    #    f_merged.explore(m=m, color="blue")
    merged.explore(m=m, color="orangered")

    # nodes.explore(m=m, color="black", size=0.5)

    # current_basin_gdf.boundary.explore(m=m)

    # folium.map.LayerControl("topleft", collapsed=False).add_to(m)
    # edges.explore(m=m, color="black", opacity=0.5)
    hospital_name = hospital_r["amenity_name"].values[0]
    os.makedirs(f"germanyFloods/figs/folium/{hospital_name}", exist_ok=True)
    m.save(
        f"germanyFloods/figs/folium/{hospital_name}/map-{event}-{hospital_name}.html"
    )
    return m
    # m


# %%
gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=0.08
)

gdf.sort_values(by=["population_percentual_diff", "min_dist"], inplace=True)

# %%

# no_dupes = gdf.drop_duplicates(subset=["node"], keep="last")
tail = gdf[gdf["population_percentual_diff"] >= 0.3]
len(tail)


# %%

for k in range(len(tail)):
    print(k)
    generate_folium_map(k, tail=tail)


# %%
