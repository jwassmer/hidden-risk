# %%
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd

# import geopandas as gpd
import numpy as np

# import matplotlib.pyplot as plt
import hashlib
import pyproj

from shapely.geometry import Point
from shapely.strtree import STRtree
import heapq


from src import OSMparser as osmp
from src import SupportFunctions as sf
from src import Isochrones as ic

# %%


def dists_to_raster(region, raster, key="amenity", tag="hospital"):
    hospital_nodes = region.nodes[region.nodes[key] == tag].copy()

    hospital_nodes["min_dist"] = hospital_nodes.geometry.apply(
        lambda pt: min_dist_to_raster(pt, raster)
    )
    return hospital_nodes["min_dist"]


def dists_to_tifs(region, tif_files, key="amenity", tag="hospital"):
    if type(tif_files) == str:
        tif_files = [tif_files]

    hospital_nodes = region.nodes[region.nodes[key] == tag].copy()

    hospital_nodes.loc[:, "min_dist"] = np.inf
    for tif_file in tif_files:
        raster = fr.read_raster(tif_file)
        min_dist_list = hospital_nodes.geometry.apply(
            lambda pt: min_dist_to_raster(pt, raster)
        )
        hospital_nodes.loc[:, "min_dist"] = np.minimum(
            hospital_nodes["min_dist"], min_dist_list
        )

    return hospital_nodes["min_dist"]


def access_gdf(
    region,
    key="amenity",
    tag="hospital",
    weight="spillover_travel_time",
    alpha=25,
    buffer_metre=100,
):
    nodes = region.nodes

    sources = list(nodes[nodes[key] == tag].index)
    # targets = list(nodes[nodes["amenity"] == "hospital"].index)
    lengths, paths = nx.multi_source_dijkstra(region.graph, sources, weight=weight)
    for node, length in lengths.items():
        if length < np.inf:
            nodes.loc[node, f"closest_{tag}"] = int(paths[node][0])
        else:
            nodes.loc[node, f"closest_{tag}"] = np.nan

    nx.set_node_attributes(region.graph, nodes[f"closest_{tag}"], f"closest_{tag}")
    access_arr = nodes[f"closest_{tag}"].unique()
    access_arr = access_arr[~np.isnan(access_arr)]

    mapping = dict(zip(access_arr, range(0, len(access_arr))))
    nodes[f"closest_{tag}_map"] = nodes[f"closest_{tag}"].map(mapping)
    access_gdf = access_polys(
        region.graph, f"closest_{tag}", alpha=alpha, buffer_metre=buffer_metre
    )
    access_gdf[f"{key}_name"] = nodes.loc[access_gdf.index][f"{key}_name"]
    access_gdf = access_gdf.set_crs("EPSG:4326")

    access_gdf.to_crs("EPSG:3857", inplace=True)
    access_gdf["area_size"] = access_gdf["isochrone_poly"].area

    access_gdf.to_crs("EPSG:4326", inplace=True)
    access_gdf.sort_values(by="area_size", ascending=False)
    return access_gdf


def access_gdfs(
    region,
    region_r,
    key="amenity",
    tag="hospital",
    weight="spillover_travel_time",
    buffer_metre=100,
):
    access_gdfs = []
    for r in [region, region_r]:
        nodes = r.nodes
        sources = list(nodes[nodes[key] == tag].index)
        # targets = list(nodes[nodes["amenity"] == "hospital"].index)
        lengths, paths = nx.multi_source_dijkstra(r.graph, sources, weight=weight)
        for node, length in lengths.items():
            if length < np.inf:
                nodes.loc[node, f"closest_{tag}"] = int(paths[node][0])
            else:
                nodes.loc[node, f"closest_{tag}"] = np.nan

        nx.set_node_attributes(r.graph, nodes[f"closest_{tag}"], f"closest_{tag}")
        access_arr = nodes[f"closest_{tag}"].unique()
        access_arr = access_arr[~np.isnan(access_arr)]

        mapping = dict(zip(access_arr, range(0, len(access_arr))))
        nodes[f"closest_{tag}_map"] = nodes[f"closest_{tag}"].map(mapping)
        access_gdf = access_polys(
            r.graph, f"closest_{tag}", alpha=25, buffer_metre=buffer_metre
        )
        access_gdf[f"{key}_name"] = nodes.loc[access_gdf.index][f"{key}_name"]
        access_gdf = access_gdf.set_crs("EPSG:4326")
        access_gdfs.append(access_gdf)
    return access_gdfs


def find_n_closest_nodes(G, H, N, weight="spillover_travel_time"):
    # Dictionary to store the closest nodes and their distances for each node
    closest_nodes = {node: [] for node in G.nodes()}

    # Calculate shortest paths from each node in H to every other node
    for h_node in H:
        lengths = nx.single_source_dijkstra_path_length(G, h_node, weight=weight)
        for target_node, distance in lengths.items():
            if target_node != h_node:
                # Use a max heap to keep only the N closest nodes
                if len(closest_nodes[target_node]) < N:
                    heapq.heappush(closest_nodes[target_node], (-distance, h_node))
                else:
                    heapq.heappushpop(closest_nodes[target_node], (-distance, h_node))

    # Format the result to include distances and nodes
    result = {
        node: [
            (-dist, h_node)
            for dist, h_node in sorted(closest_nodes[node], reverse=True)
        ]
        for node in closest_nodes
    }

    for h_node in H:
        result[h_node].insert(
            0, (1.0, h_node)
        )  # set closest distance to 1m to avoid division by zero
        result[h_node] = result[h_node][:N]

    return result


def service_population_from_n_closest(G, H, N, weight="spillover_travel_time"):
    n_closest_nodes = find_n_closest_nodes(G, H, N, weight=weight)

    service_population = {h: 0 for h in H}
    for node in n_closest_nodes:
        distances = np.array([dist for dist, h_node in n_closest_nodes[node]])

        # Compute the likelihood using the inverse function
        likelihoods = np.where(np.isinf(distances), 0, 1 / distances)

        # Normalize the likelihoods so they sum to 1, only consider finite distances
        likelihoods_sum = np.sum(likelihoods[~np.isinf(distances)])
        if likelihoods_sum == 0:
            likelihoods = np.zeros_like(likelihoods)
        else:
            likelihoods /= likelihoods_sum

        for prob, (dist, h_node) in zip(likelihoods, n_closest_nodes[node]):
            service_population[h_node] += prob * G.nodes[node]["population"]

    closest_nodes = {
        node: (
            np.nan if len(n_closest_nodes[node]) == 0 else n_closest_nodes[node][0][1]
        )
        for node in n_closest_nodes
    }
    return service_population, closest_nodes


def get_access_gdf2(
    region,
    N=5,
    key="amenity",
    tag="hospital",
    weight="spillover_travel_time",
    buffer_metre=100,
):

    r = region
    nodes = r.nodes
    G = r.graph
    sources = list(nodes[nodes[key] == tag].index)
    service_population, closest_nodes = service_population_from_n_closest(
        G, sources, N, weight=weight
    )

    for node, H in closest_nodes.items():
        nodes.loc[node, f"closest_{tag}"] = H

    hospital_nodes = nodes[nodes[key] == tag].index

    for h_node in hospital_nodes:
        nodes.loc[h_node, f"closest_{tag}"] = h_node

    nodes[f"closest_{tag}"] = nodes[f"closest_{tag}"]

    nx.set_node_attributes(r.graph, nodes[f"closest_{tag}"], f"closest_{tag}")
    access_arr = nodes[f"closest_{tag}"].unique()
    # access_arr = access_arr[~np.isnan(access_arr)]

    mapping = dict(zip(access_arr, range(0, len(access_arr))))
    nodes[f"closest_{tag}_map"] = nodes[f"closest_{tag}"].map(mapping)
    region.nodes = nodes
    access_gdf = access_polys(
        r.graph, f"closest_{tag}", alpha=25, buffer_metre=buffer_metre
    )
    access_gdf[f"{key}_name"] = nodes.loc[access_gdf.index][f"{key}_name"]
    access_gdf = access_gdf.set_crs("EPSG:4326")
    access_gdf["prob_service_population"] = access_gdf.index.map(service_population)

    return access_gdf


def access_polys(G, nodeweight, alpha=100, buffer_metre=5):
    nodes = ox.graph_to_gdfs(G, edges=False)
    crs = nodes.crs

    access_arr = nodes[nodeweight].unique()
    access_arr = sorted(access_arr[~np.isnan(access_arr)])

    mapping = dict(zip(access_arr, range(0, len(access_arr))))
    # nodes[f"{nodeweight}_map"] = nodes[nodeweight].map(mapping)

    pop_arr = []
    iso_shapes = []
    for h in access_arr:
        sub_nodes = nodes[(nodes[nodeweight] == h)]  # (nodes[nodeweight] >= t0)]# &
        shape = ic.alpha_shape(sub_nodes.geometry, alpha)
        pop_arr.append(sf.finite_sum(sub_nodes["population"]))

        iso_shapes.append(shape)
    gdf = (
        gpd.GeoDataFrame(
            {
                "node": access_arr,
                f"isochrone_poly": gpd.GeoSeries(iso_shapes),
            }
        )
        .set_geometry("isochrone_poly")
        .set_crs(crs)
    )
    gdf["node_map"] = gdf["node"].map(mapping)
    gdf["population"] = pop_arr
    gdf.set_index("node", inplace=True)

    gdf.to_crs("EPSG:3857", inplace=True)
    gdf["isochrone_poly"] = gdf["isochrone_poly"].buffer(buffer_metre)
    gdf.to_crs(crs, inplace=True)

    return gdf


def clean_hospital_names(gdf):
    good_names = [
        "Krankenhaus",
        "Klinik",
        "Hospital",
        "Klinikum",
        "Clinic",
        "Medizin",
        "Medic",
        "Charité",
        "Charite",
        "Uni",
        "St",
        "Sankt",
        "Arzt",
        "Doctor",
        "Ärztlich",
        "Ambulanz",
        "Ambulance",
        "zgt",
        "kliniek",
        "Hospice",
        "Hospiz",
        "Johanniter",
        "Ziekenhuis",
    ]
    bad_names = ["covid"]
    # remove bad names
    gdf = gdf[gdf["name"].str.contains("|".join(bad_names), case=False) == False]
    gdf = gdf[gdf["name"].notna()]
    # good_entries = gdf[gdf["name"].str.contains("|".join(good_names), case=False)]
    # bad_entries = gdf[~gdf["name"].str.contains("|".join(good_names), case=False)]
    # print("Removed the following hospitals:", bad_entries["name"].tolist())
    return gdf


def keep_longest_unique_sublists_with_distances(input_list_with_distances):
    # Create a dictionary to hold the longest sublist for each unique number, along with its distance
    longest_sublists_with_distances = {}

    # Iterate through the original list to populate the dictionary
    for sublist, distance in input_list_with_distances:
        for item in sublist:
            # Update the dictionary only if the item is not already a key,
            # or if the current sublist is longer than the stored one
            if item not in longest_sublists_with_distances or len(sublist) > len(
                longest_sublists_with_distances[item][0]
            ):
                longest_sublists_with_distances[item] = (sublist, distance)

    # Extract the unique longest sublists and their distances while preserving the original list's order
    unique_longest_sublists_with_distances = []
    seen_sublists = set()
    for sublist, distance in input_list_with_distances:
        # Convert the sublist to a tuple for hashability
        sublist_tuple = tuple(sublist)
        if sublist_tuple in seen_sublists:
            continue  # Skip if we've already added this sublist
        if all(
            item in longest_sublists_with_distances
            and longest_sublists_with_distances[item][0] == sublist
            for item in sublist
        ):
            unique_longest_sublists_with_distances.append((sublist, distance))
            seen_sublists.add(sublist_tuple)

    return unique_longest_sublists_with_distances


def merge_similar(gdf, key, distance_metre=1000):
    original_crs = gdf.crs
    gdf = gdf.to_crs(epsg=3857)
    similar_list = []

    tree = STRtree([geom.centroid for geom in gdf.geometry])

    for geom in gdf.geometry:
        similar = tree.query(geom.centroid.buffer(distance_metre))
        similar_idx = [gdf.iloc[idx].name for idx in similar]
        if len(similar_idx) > 1:
            similar_list.append(similar_idx)
            distance = np.max(
                [geom.distance(gdf.loc[idx].geometry) for idx in similar_idx]
            )
            similar_list[-1] = (similar_list[-1], distance)

    for name in gdf["name"].unique():
        similar_idx = gdf[gdf["name"] == name].index.to_list()
        if len(similar_idx) > 1:
            similar_list.append(similar_idx)
            distance = np.max(
                [
                    gdf.loc[similar_idx[0]].geometry.distance(gdf.loc[idx].geometry)
                    for idx in similar_idx
                ]
            )

            # merge same names if distance is less than 100m
            if distance < 100:
                similar_list[-1] = (similar_list[-1], distance)
                print("Identical names:", name, similar_list[-1])
            else:
                similar_list.pop()

    similar_list = keep_longest_unique_sublists_with_distances(similar_list)

    print(
        "Merging:",
        [
            (gdf.loc[similar]["name"].to_list(), round(dist, 2))
            for (similar, dist) in similar_list
        ],
    )
    similar_list = [similar for (similar, dist) in similar_list]
    for similar in similar_list:
        name = ""
        for k, index in enumerate(similar):
            amenity_name = gdf.loc[index, f"name"]
            if amenity_name:
                if name:
                    name += f" + {amenity_name}"
                else:
                    name = amenity_name
            if k != 0:
                gdf.loc[index, key] = None
        gdf.loc[similar[0], f"name"] = name
    gdf = gdf.to_crs(original_crs)
    return gdf


def poi_from_gdf(path, osmpbf, gdf, key, tag, overwrite=False):
    """
    Retrieves points of interest (POIs) from a GeoDataFrame based on a specified key and tag.

    Args:
        path (str): The path to the directory to the OSM folder.
        osmpbf (str): The path to the OSM PBF file.
        gdf (geopandas.GeoDataFrame): The GeoDataFrame representing the area of interest.
        key (str): The OSM key for filtering POIs.
        tag (str): The OSM tag for filtering POIs.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the retrieved POIs.
    """
    filter_str = f"nw/{key}={tag}"
    uuid = hashlib.sha256(gdf.geometry.iloc[0].wkt.encode("utf-8")).hexdigest()
    filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()

    file_path = f"{path}/{uuid}/{filter_hash}/pois.geojson"

    try:
        if overwrite:
            raise Exception
        poi_df = gpd.read_file(file_path, index_col=0)
        poi_df = poi_df.reset_index().set_index("osmid")
    except:
        osmp.filter_osmpbf(path, osmpbf, gdf, filter_str)

        poi_df = ox.features_from_xml(f"{path}/{uuid}/{filter_hash}/filtered.osm")

        if len(poi_df) == 0:
            print(f"No {key}_{tag} in specified area.")
            return

        poi_df = poi_df[~poi_df[key].isna()]
        poi_df = poi_df.reset_index().set_index("osmid")
        poi_df = poi_df[[key, "name", "geometry"]]

        poi_df.to_file(file_path, driver="GeoJSON")

    return poi_df


def features_from_gdf(path, osmpbf, gdf, key, tag, extra_col=None):
    """
    Retrieves points of interest (POIs) from a GeoDataFrame based on a specified key and tag.

    Args:
        path (str): The path to the directory to the OSM folder.
        osmpbf (str): The path to the OSM PBF file.
        gdf (geopandas.GeoDataFrame): The GeoDataFrame representing the area of interest.
        filter_str (str): filter str to filter osm

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the retrieved POIs.
    """
    filter_str = f"w/{key}={tag}"

    uuid = hashlib.sha256(gdf.geometry.iloc[0].wkt.encode("utf-8")).hexdigest()

    filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()

    # if not Path(f"{path}/{uuid}/{filter_hash}/pois.geojson").is_file():
    osmp.filter_osmpbf(path, osmpbf, gdf, filter_str)
    poi_df = ox.features_from_xml(f"{path}/{uuid}/{filter_hash}/filtered.osm")

    poi_df = poi_df[~poi_df[key].isna()]

    poi_df = poi_df.reset_index().set_index("osmid")
    # col_list = [key, "geometry"]

    # if extra_col != None:
    #    poi_df = poi_df.append(extra_col)
    # poi_df = poi_df[col_list]

    # poi_df.to_file(f"{path}/{uuid}/{filter_hash}/pois.geojson", driver="GeoJSON")
    # else:
    #    poi_df = gpd.read_file(f"{path}/{uuid}/{filter_hash}/pois.geojson", index_col=0)
    #    poi_df = poi_df.reset_index().set_index("osmid")

    return poi_df


def set_shortest_path_to_emergency(G, weight="travel_time"):
    """
    Extract points of interest (POIs) from a GeoDataFrame based on specified key-value tags using OpenStreetMap data.

    Parameters:
        path (str): The path to the directory to the OSM folder.
        osmpbf (str): Path to the OSM PBF file.
        gdf (geopandas.GeoDataFrame): Input GeoDataFrame representing the area of interest.
        key (str): Key for filtering the POIs.
        tag (str): Tag value for filtering the POIs.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the extracted POIs.
    """
    nodes = ox.graph_to_gdfs(G, edges=False)
    path_length_emerg = nx.multi_source_dijkstra_path_length(
        G, list(nodes[nodes["emergency"]].index), weight=weight
    )
    path_length_emerg = {k: v / 60 for k, v in path_length_emerg.items()}
    return path_length_emerg


def min_dist_to_raster(pt, raster, projected_crs="EPSG:32633"):
    """
    Calculates the minimum distance between a point and a raster.

    Parameters:
        pt (shapely.geometry.Point): The point.
        raster (rioxarray.RasterArray): The raster.
        projected_crs (str): The projected CRS to use for the calculation.
    Returns:
        float: The minimum distance between the point and the raster.
    """
    # project pt to projected_crs
    transformer = pyproj.Transformer.from_crs(
        raster.rio.crs, projected_crs, always_xy=True
    )
    pt_projected = Point(transformer.transform(pt.x, pt.y))

    point_x, point_y = pt_projected.x, pt_projected.y

    # Reproject both the point and the raster to a projected CRS (e.g., UTM)
    raster_reproj = raster.rio.reproject(projected_crs)

    # raster_array = raster_reproj.squeeze().values
    # Get the raster's geographical coordinates
    x_coords, y_coords = np.meshgrid(raster_reproj.x, raster_reproj.y)
    greater_zero_pixels = (
        ~np.isnan(raster_reproj.values) & (raster_reproj.values > 0.3)
    )[0]

    distances = np.full(greater_zero_pixels.shape, np.nan)

    # Calculate distances only for non-NaN pixels
    distances[greater_zero_pixels] = np.sqrt(
        (x_coords[greater_zero_pixels] - point_x) ** 2
        + (y_coords[greater_zero_pixels] - point_y) ** 2
    )

    return np.nanmin(distances)


# %%

if "__main__" == __name__:
    from src import FloodRaster as fr
    from src import RoadNetwork2 as rn
    from src import GeoModule as gm

    flood_path = "germanyFloods/data/wse2_clip_fdsc-r05_wsh.tif"
    path = "data/osmfiles/latest"
    osmpbf = "ger-buffered-200km.osm.pbf"
    # raster = fr.read_raster(flood_path)
    xmin, ymin, xmax, ymax = 7.9, 52.2, 8.1, 52.4
    gamma = 0.1
    threads = 16
    # raster = raster.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)

    bbox = gm.bbox(north=ymax, south=ymin, west=xmin, east=xmax)

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
        osmpbf=osmpbf,
        highway_filter=f"w/highway={','.join(driving_tags)}",
        gdf=bbox.gdf,
    )

    key = "amenity"
    tag = "hospital"
    region.add_pois(key, tag)

    nodes = region.nodes
    # %%
    region.loads(threads=threads, weight="travel_time")
    region.effective_spillover_velocities(0.02)

    # region_r = region.copy()
    # region_r.water_depth(raster)

    # wet_roads = region_r.edges[region_r.edges["water_depth"] > 0.3].index
    # print(f"There are {len(wet_roads)} wet roads")
    # region_r.remove_edges(wet_roads)
    # region_r.loads("travel_time", threads=threads)
    # region_r.effective_spillover_velocities(gamma)

    access_gdf = get_access_gdf2(
        region,
        key=key,
        tag=tag,
        weight="spillover_travel_time",
        buffer_metre=100,
        N=5,
    )
    # access_gdf.plot(column="prob_service_population", legend=True)

    # %%

    h = nodes[nodes["amenity"] == "hospital"]

    cnodes = find_n_closest_nodes(
        region.graph, h.index, 3, weight="spillover_travel_time"
    )
    cnodes[h.index[1]]

    # %%
