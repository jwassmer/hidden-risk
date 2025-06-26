# %%
# ------------------ PACKAGES ---------------------------#
import geopandas as gpd
import os
import subprocess

from pathlib import Path
import hashlib
import osmnx as ox
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon

import requests
from io import StringIO

from src import PopulationFromRaster as pfr


# ------------------ FUNCTIONS ---------------------------#
def get_location_shape(location_name):
    """
    Returns the shape of a location in a geopandas dataframe with coordinate reference system epsg:4326.

    Parameters
    ----------
        location_name (str): The name of the location for which to retrieve the shape.

    Returns
    -------
        A geopandas dataframe representing the shape of the location in epsg:4326 coordinate reference system.
    """
    # Use the geopandas library to retrieve the shape of the location
    # shape = gpd.read_file(
    #    f"https://nominatim.openstreetmap.org/search.php?q={location_name}&polygon_geojson=1&format=geojson"
    # )

    url = f"https://nominatim.openstreetmap.org/search.php?q={location_name}&polygon_geojson=1&format=geojson"
    headers = {
        "User-Agent": "MyGeocodingApp/1.0 (contact@mygeocodingapp.com)",
        "Referer": "https://mygeocodingapp.com",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        shape = gpd.read_file(StringIO(response.text))
    else:
        print(f"Failed to fetch data. HTTP response code: {response.status_code}")

    admin_boundary = shape[shape["type"] == "administrative"]

    # Set the coordinate reference system of the shape to epsg:4326
    admin_boundary = admin_boundary.to_crs("epsg:4326")

    return admin_boundary


def extract_pbf_from_poly(path, osm_pbf, gdf, return_uuid=False):
    """
    Extracts a PBF file from a polygon bounded by a GeoDataFrame.

    Parameters
    ----------
        path (str): The path to the directory where the extracted PBF file and related files will be saved.
        osm_pbf (str): The path to the original OSM PBF file.
        gdf (GeoDataFrame): The GeoDataFrame containing the polygon geometry.
        return_uuid (bool, optional): Whether to return the UUID. Defaults to False.

    Returns
    --------
        str or None: The UUID if `return_uuid` is True, otherwise None.
    """
    uuid = hashlib.sha256(gdf.geometry.unary_union.wkt.encode("utf-8")).hexdigest()

    if not Path(f"{path}/{uuid}").is_dir():
        os.system(f"mkdir {path}/{uuid}")

    gdf.to_file(f"{path}/{uuid}/polygon.geojson", driver="GeoJSON")

    # if not Path(f"{path}/{uuid}/nofilter.osm.pbf").is_file():

    subprocess.run(
        [
            "osmium",
            "extract",
            "-p",
            f"{path}/{uuid}/polygon.geojson",
            f"{path}/{osm_pbf}",
            "-o",
            f"{path}/{uuid}/nofilter.osm.pbf",
            "--overwrite",
        ]
    )

    if return_uuid:
        return uuid


def filter_osmpbf(path, osm_pbf, gdf, filter_str, return_uuid=False):
    """
    Filters an OSM PBF file based on a given filter string and the polygon bounded by a GeoDataFrame.

    Paramters
    ---------
        path (str): The path to the directory where the filtered OSM PBF file and related files will be saved.
        osm_pbf (str): The path to the original OSM PBF file.
        gdf (GeoDataFrame): The GeoDataFrame containing the polygon geometry.
        filter_str (str): The filter string to apply to the OSM PBF file.
        return_uuid (bool, optional): Whether to return a unique identifier (uuid) of the Polygon. Defaults to False.

    Returns
    -------
        str or None: The UUID if `return_uuid` is True, otherwise None.
    """
    # Extract the *.osm.pbf bounded by the polygon and create unique identifier str
    uuid = extract_pbf_from_poly(path, osm_pbf, gdf, return_uuid=True)

    filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()
    if not Path(f"{path}/{uuid}/{filter_hash}").is_dir():
        os.system(f"mkdir {path}/{uuid}/{filter_hash}")

    if not Path(f"{path}/{uuid}/{filter_hash}/filtered.osm.pbf").is_file():
        os.system(
            f"osmium tags-filter {path}/{uuid}/nofilter.osm.pbf {filter_str}\
            -o {path}/{uuid}/{filter_hash}/filtered.osm.pbf --overwrite"
        )

    # Convert the filtered OSM PBF file to XML format
    osmpbf_to_xml(path, uuid, filter_hash)
    if return_uuid:
        return uuid


def filter_osmpbf_by_id(path, osm_pbf, gdf, id_str, return_uuid=False):
    """
    Filters an OSM PBF file based on a given filter string and the polygon bounded by a GeoDataFrame.

    Paramters
    ---------
        path (str): The path to the directory where the filtered OSM PBF file and related files will be saved.
        osm_pbf (str): The path to the original OSM PBF file.
        gdf (GeoDataFrame): The GeoDataFrame containing the polygon geometry.
        id_str: The way osmids in an array to filter the OSM PBF file.
        return_uuid (bool, optional): Whether to return a unique identifier (uuid) of the Polygon. Defaults to False.

    Returns
    -------
        str or None: The UUID if `return_uuid` is True, otherwise None.
    """
    # Extract the *.osm.pbf bounded by the polygon and create unique identifier str
    uuid = extract_pbf_from_poly(path, osm_pbf, gdf, return_uuid=True)

    filter_hash = hashlib.sha256(id_str.encode("utf-8")).hexdigest()
    if not Path(f"{path}/{uuid}/{filter_hash}").is_dir():
        os.system(f"mkdir {path}/{uuid}/{filter_hash}")

    # Split the string into a list of IDs
    id_list = id_str.split()
    # Define the file name where the IDs will be saved
    filename = f"{path}/{uuid}/id_file.txt"
    # Write each ID to a separate line in the file
    with open(filename, "w") as file:
        for id in id_list:
            file.write(id + "\n")

    subprocess.run(
        f"osmium getid -r -f xml {path}/{uuid}/nofilter.osm.pbf --id-file={filename} -o {path}/{uuid}/{filter_hash}/filtered.osm --overwrite",
        shell=True,
    )

    # Convert the filtered OSM PBF file to XML format
    # osmpbf_to_xml(path, uuid, filter_hash)
    if return_uuid:
        return uuid


def osmpbf_to_xml(path, uuid, filter_hash):
    """
    Converts an OSM PBF file to XML format.

    Parameters
    ----------
        path (str): The path to the directory containing the files.
        uuid (str): The UUID of the polygon.
        filter_hash (str): The hash of the filter string.

    Returns
    -------
        None
    """
    if not Path(f"{path}/{uuid}/{filter_hash}/filtered.osm").is_file():
        os.system(
            f"osmium cat {path}/{uuid}/{filter_hash}/filtered.osm.pbf\
                -o {path}/{uuid}/{filter_hash}/filtered.osm.bz2 --overwrite"
        )
        os.system(f"bzip2 -d {path}/{uuid}/{filter_hash}/filtered.osm.bz2 -f")


def graph_from_gdf(path, osmpbf, gdf, filter_str, retain_all=False):
    """
    Creates a networkx graph from a filtered OSM PBF file.

    Parameters
    ----------
        path (str): The path to the directory containing the files.
        osmpbf (str): The path to the original OSM PBF file.
        gdf (GeoDataFrame): The GeoDataFrame containing the polygon geometry.
        filter_str (str): The filter string to apply to the OSM PBF file.

    Returns
    -------
        networkx.MultiDiGraph: The created graph.
    """
    filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()
    uuid = filter_osmpbf(path, osmpbf, gdf, filter_str, return_uuid=True)
    G = ox.graph_from_xml(
        f"{path}/{uuid}/{filter_hash}/filtered.osm",
        retain_all=retain_all,
        simplify=True,
    )

    # Remove the temporary files
    # os.system(f"rm {path}/{uuid}/{filter_hash}/filtered.osm")
    os.system(f"rm {path}/{uuid}/{filter_hash}/filtered.osm.pbf")
    return G


def features_from_osmids(path, osmpbf, gdf, id_arr, return_path=False):
    """
    Creates a networkx graph from a filtered OSM PBF file.

    Parameters
    ----------
        path (str): The path to the directory containing the files.
        osmpbf (str): The path to the original OSM PBF file.
        gdf (GeoDataFrame): The GeoDataFrame containing the polygon geometry.
        id_arr (str): The edge osmids to filter the OSM PBF file.

    Returns
    -------
        networkx.MultiDiGraph: The created graph.
    """

    id_str = " ".join(f"w{num}" for num in id_arr)

    filter_hash = hashlib.sha256(id_str.encode("utf-8")).hexdigest()
    uuid = filter_osmpbf_by_id(path, osmpbf, gdf, id_str, return_uuid=True)
    feats = ox.features.features_from_xml(
        f"{path}/{uuid}/{filter_hash}/filtered.osm",
    )

    # Remove the temporary files
    # os.system(f"rm {path}/{uuid}/{filter_hash}/filtered.osm")
    # os.system(f"rm {path}/{uuid}/{filter_hash}/filtered.osm.pbf")
    if return_path:
        return feats, f"{path}/{uuid}/{filter_hash}"

    return feats


def road_features_from_tags(path, tags, polygon=None):
    filtered_path = f"{path}/filtered.osm"
    feats = ox.features.features_from_xml(filtered_path, polygon=polygon, tags=tags)
    return feats


def gdf_from_bbox(north, south, west, east):
    """
    Creates a GeoDataFrame representing a bounding box defined by coordinates.

    Parameters
    ----------
        north (float): The northern latitude coordinate.
        south (float): The southern latitude coordinate.
        west (float): The western longitude coordinate.
        east (float): The eastern longitude coordinate.

    Returns
    -------
        GeoDataFrame: The GeoDataFrame representing the bounding box.
    """
    # Create a Shapely polygon object using the coordinates
    polygon = Polygon([(west, north), (east, north), (east, south), (west, south)])

    # Create a GeoDataFrame with the polygon as the geometry and CRS set to EPSG:4326
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    return gdf


def add_edge_speed(graph):
    """
    Adds edge speeds and travel times to a networkx MultiDiGraph.

    Parameters
    ----------
        graph (MultiDiGraph): The graph to update with edge speeds.

    Returns
    -------
        MultiDiGraph: The updated graph with edge speeds and travel times.
    """
    graph = ox.speed.add_edge_speeds(graph, fallback=13)
    speed_ms_arr = np.array(list(nx.get_edge_attributes(graph, "speed_kph").values()))
    nx.set_edge_attributes(
        graph,
        dict(zip(graph.edges(keys=True), speed_ms_arr * 1000 / 60 / 60)),
        "speed_ms",
    )
    graph = ox.speed.add_edge_travel_times(graph)
    return graph


def generate_uuid_path(gdf=None, path_to_osmpbf=None, highway_filter=None):
    uuid = hashlib.sha256(gdf.geometry.unary_union.wkt.encode("utf-8")).hexdigest()
    filter_hash = hashlib.sha256(highway_filter.encode("utf-8")).hexdigest()

    file_path = f"{path_to_osmpbf}/{uuid}/{filter_hash}"
    return file_path


def generate_region_gdf(**kwargs):
    if "place" in kwargs:
        gdf = get_location_shape(kwargs.get("place"))
        # if type(gdf.geometry.iloc[0]) == shapely.geometry.multipolygon.MultiPolygon:
        #    gdf.geometry.iloc[0] = Polygon(gdf.geometry.iloc[0].geoms[0].exterior)
    elif "bbox" in kwargs:
        gdf = gdf_from_bbox(*kwargs.get("bbox"))
    elif "polygon" in kwargs:
        gdf = gpd.GeoDataFrame(geometry=[kwargs.get("polygon")], crs="EPSG:4326")
    elif "gdf" in kwargs:
        gdf = kwargs.get("gdf")
    return gdf


def road_graph_with_populations(
    gdf=None, path_to_osmpbf=None, osmpbf=None, highway_filter=None, retain_all=False
):
    """
    Creates a road graph with population data.

    Parameters
    ----------
        return_bound_gdf (bool, optional): Whether to return the boundary GeoDataFrame. Defaults to False.
        path (str, optional): The path to the directory containing the files. Defaults to False.
        osmpbf (str, optional): The path to the original OSM PBF file. Defaults to False.
        highway_filter (str, optional): The filter string to apply to the OSM PBF file. Defaults to False.

    Returns
    -------
        tuple or MultiDiGraph: If return_bound_gdf is True, returns a tuple containing the road graph and the boundary GeoDataFrame.
        Otherwise, returns the road graph.
    """
    path_to_file = generate_uuid_path(
        gdf=gdf, path_to_osmpbf=path_to_osmpbf, highway_filter=highway_filter
    )

    file_name = f"{path_to_file}/graph.gpickle"

    # file_path = f"{path}/{uuid}/{filter_hash}/graph.gpickle"
    try:
        with open(file_name, "rb") as pickle_file:
            graph = pickle.load(pickle_file)
        print(f"Loaded existing graph from '{file_name}'.")

    except:
        print("Generating new graph...")
        graph = graph_from_gdf(
            path_to_osmpbf,
            osmpbf,
            gdf,
            highway_filter,
            retain_all=retain_all,
        )
        graph = add_edge_speed(graph)
        print("Adding population to nodes...")
        graph = pfr.population_to_graph_nodes(graph)

        with open(file_name, "wb") as pickle_file:
            pickle.dump(graph, pickle_file)
        print(f"Saved to '{file_name}'.")

    return graph, path_to_file


# %%
"""


path = "data/osmfiles/latest"
osmpbf = "germany.osm.pbf"
locgdf = get_location_shape("Cologne,Germany")

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
    "road",
    "unclassified",
    "residential",
    "living_street",
]

g = graph_from_pbf(path, osmpbf, locgdf, f"w/highway={','.join(driving_tags)}")

nodes, edges = ox.graph_to_gdfs(g)
# %%

edges.plot()
#edges.plot(column=edges["highway"].astype(str), legend=True)

# %%
"""
