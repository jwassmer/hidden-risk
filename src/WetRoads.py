# %%
import rioxarray as riox
import numpy as np
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import pandas as pd
import networkx as nx
import osmnx as ox
import pickle
import geopandas as gpd
import hashlib
from shapely.geometry import mapping
import os


from src import GeoModule as gm
from src import OSMparser as osmp


# %%
def matching_value(list1, list2):
    for value in list1:
        if value in list2:
            return value
    return False


def water_depth_bridges(region, flood_raster, polygon=None):
    edges = region.edges
    if polygon != None:
        edges = edges[edges.intersects(polygon)]
    bridge_edges = edges[edges["bridge"] == "yes"]  # .reset_index(drop=True)

    osmid_arr = []
    for osmid_list in bridge_edges["osmid"]:
        if type(osmid_list) == int:
            osmid_arr.append(osmid_list)
            continue
        for osmid in osmid_list:
            osmid_arr.append(osmid)

    feats = region.geometries_from_osmid(osmid_arr)
    bridge_gdf = feats[feats["bridge"] == "yes"].reset_index(level=0)
    bridge_gdf = bridge_gdf[
        ~(bridge_gdf["geometry"].is_empty | bridge_gdf["geometry"].isna())
    ]

    stats = zonal_stats(
        bridge_gdf["geometry"].boundary,
        flood_raster.data[0],
        affine=flood_raster.rio.transform(),
        stats=["max", "min"],
        nodata=np.nan,
        all_touched=False,
    )

    bridge_gdf["water_depth"] = np.array(
        [s["max"] if s["max"] != None else 0.0 for s in stats]
    )

    wet_bridges = bridge_gdf[bridge_gdf["water_depth"] > 0]

    water_depth_arr = np.zeros(len(bridge_edges))

    for j, osmid in enumerate(bridge_edges["osmid"]):
        if type(osmid) == int:
            osmid = [osmid]
        match_osmid = matching_value(wet_bridges.index, osmid)
        if match_osmid != False:
            water_depth_arr[j] = wet_bridges.loc[match_osmid]["water_depth"]

    bridge_edges = bridge_edges.assign(water_depth=water_depth_arr)
    return bridge_edges


def water_depth_roads(region, flood_raster, polygon=None):
    edges = region.edges
    if polygon != None:
        edges = edges[edges.intersects(polygon)]

    roads = edges[edges["bridge"] != "yes"]  # .reset_index(drop=True)

    stats = zonal_stats(
        roads["geometry"],
        flood_raster.data[0],
        affine=flood_raster.rio.transform(),
        stats="max",
        nodata=np.nan,
        all_touched=False,
    )

    roads = roads.assign(
        water_depth=np.array([s["max"] if s["max"] != None else 0.0 for s in stats])
    )
    return roads


def get_bridge_polys_from_id(region):
    edges = region.edges
    bridge_edges = edges[edges["bridge"] == "yes"]  # .reset_index(drop=True)

    osmid_arr = []
    for osmid_list in bridge_edges["osmid"]:
        if type(osmid_list) == int:
            osmid_arr.append(osmid_list)
            continue
        for osmid in osmid_list:
            osmid_arr.append(osmid)

    feats = region.geometries_from_osmid(osmid_arr)
    bridge_gdf = feats[feats["bridge"] == "yes"].reset_index(level=0)
    return bridge_gdf


def get_tunnel_polys(region):
    try:
        tunnel_gdf = gpd.read_file(
            f"{region.path}/tunnel_gdf.geojson", driver="GeoJSON"
        )
        print("Tunnel geometries loaded from cache.")
    except:
        os.system(
            f"osmium tags-filter {region.path}/filtered.osm w/tunnel\
            -o {region.path}/tunnels.osm --overwrite"
        )

        tunnel_gdf = ox.features.features_from_xml(f"{region.path}/tunnels.osm")
        tunnel_gdf = tunnel_gdf[~tunnel_gdf["tunnel"].isna()].reset_index(level=0)
        tunnel_gdf = tunnel_gdf[["tunnel", "geometry"]]

        # bridge_gdf = osmp.road_features_from_tags(region.path, tags={"bridge": True})
        # bridge_gdf = bridge_gdf[~bridge_gdf["bridge"].isna()].reset_index(level=0)
        # bridge_gdf = bridge_gdf[["bridge", "geometry"]]
        tunnel_gdf.to_file(f"{region.path}/tunnel_gdf.geojson", driver="GeoJSON")

    return tunnel_gdf


def get_bridge_polys(region):
    try:
        bridge_gdf = gpd.read_file(
            f"{region.path}/bridge_gdf.geojson", driver="GeoJSON"
        )
        print("Bridge geometries loaded from cache.")
    except:
        os.system(
            f"osmium tags-filter {region.path}/filtered.osm w/bridge\
            -o {region.path}/bridges.osm --overwrite"
        )

        bridge_gdf = ox.features.features_from_xml(f"{region.path}/bridges.osm")
        bridge_gdf = bridge_gdf[~bridge_gdf["bridge"].isna()].reset_index(level=0)
        bridge_gdf = bridge_gdf[["bridge", "geometry"]]

        # bridge_gdf = osmp.road_features_from_tags(region.path, tags={"bridge": True})
        # bridge_gdf = bridge_gdf[~bridge_gdf["bridge"].isna()].reset_index(level=0)
        # bridge_gdf = bridge_gdf[["bridge", "geometry"]]
        bridge_gdf.to_file(f"{region.path}/bridge_gdf.geojson", driver="GeoJSON")

    return bridge_gdf


def mask_poly_raster(raster, gdf):
    clipped = raster.rio.clip(gdf["geometry"].apply(mapping), gdf.crs, invert=True)
    clipped = clipped.where(clipped != 0, other=0)
    return clipped


def water_depth_edges(edges, flood_raster):
    stats = zonal_stats(
        edges["geometry"],
        flood_raster.data[0],
        affine=flood_raster.rio.transform(),
        stats="max",
        nodata=np.nan,
        all_touched=False,
    )
    water_depth_arr = np.array([s["max"] if s["max"] != None else 0.0 for s in stats])

    if "water_depth" in edges.columns:
        max_water_vals = np.maximum(edges["water_depth"], water_depth_arr)
        # edges["water_depth"] = max_water_vals
        edges = edges.assign(water_depth=max_water_vals)
    else:
        edges = edges.assign(water_depth=water_depth_arr)
    return edges


def water_depth(region, flood_raster, polygon=None):
    graph_hash = _generate_graph_hash(
        region.graph,
    )  # Refactor hash generation to a function
    raster_hash = hashlib.sha256(flood_raster.data).hexdigest()
    path = f"cache/wet-roads/{graph_hash}_{raster_hash}.pkl"

    edges = region.edges
    if polygon != None:
        edges = region.edges[region.edges.intersects(polygon)]

    print(len(edges))

    try:
        wet_roads = pd.read_pickle(path)

        print(f"Water depth computed. Providing values from '{path}'.")
    except:
        # print("Computing water depth...", end=" ")

        ## region.edges["water_depth"] = np.zeros(len(region.edges))

        # roads = water_depth_roads(region, flood_raster, polygon=polygon)
        # bridges = water_depth_bridges(region, flood_raster, polygon=polygon)
        print("Get bridges...")
        bridges = get_bridge_polys(region)
        print("Get tunnels...")
        tunnels = get_tunnel_polys(region)
        print("Mask raster...")
        clipped = mask_poly_raster(flood_raster, bridges)
        clipped = mask_poly_raster(clipped, tunnels)
        print("Compute water depth...")
        edges = water_depth_edges(edges, clipped)

        wet_roads = edges[edges["water_depth"] > 0]["water_depth"]

        ## region.edges["water_depth"] = tot_water_depth
        wet_roads.to_pickle(path)
        print("\u2713")

    if "water_depth" not in region.edges.columns:
        region.edges["water_depth"] = 0

    if not wet_roads.empty:
        merged_df = region.edges.merge(
            wet_roads,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_df1", "_df2"),
        )
        merged_df["max_water_depth"] = (
            merged_df[["water_depth_df1", "water_depth_df2"]].fillna(0).max(axis=1)
        )
        region.edges["water_depth"] = merged_df["max_water_depth"]

    nx.set_edge_attributes(region.graph, region.edges["water_depth"], "water_depth")


def _generate_graph_hash(graph):
    if graph.is_multigraph():
        hash_graph = ox.convert.to_digraph(graph)
        hash = f"multigraph_"
    else:
        hash_graph = graph
        hash = f"graph_"

    hash += nx.weisfeiler_lehman_graph_hash(
        hash_graph,
        edge_attr="travel_time",
        node_attr="population",
        iterations=10,
        digest_size=32,
    )
    return hash


# %%

if __name__ == "__main__":
    from src import RoadNetwork2 as rn
    from src import FloodRaster as fr

    path = "data_LFS/haz/rim2019/downscale/rim2019_wd_ems_day-20141_realisation-13_raster_index-33_wse_781f853c/wse2_clip_fdsc-r05_wsh.tif"
    path2 = "data_LFS/haz/rim2019/nuts3/tifs/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif"

    raster = fr.read_raster(path)
    # %%

    bbox = gm.bbox(north=53, south=52, west=6.5, east=8)

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
        gdf=bbox.gdf,
    )
    water_depth(region, raster, polygon=bbox.shape)
    edges = region.edges
    # %%

    bridge_gdf = get_bridge_polys(region)
    clipped = mask_poly_raster(raster, bridge_gdf)

    # %%
    # clipped.rio.to_raster("clipped-normal.tif")
    # %%
    fig, ax = plt.subplots()
    cmap_b = plt.get_cmap("Blues")
    cmap_b.set_under("none")
    cmap = plt.get_cmap("cividis")
    cmap.set_under("lightgrey")
    norm = plt.Normalize(vmin=1e-3, vmax=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # edges.plot(ax=ax, column="water_depth", cmap=cmap, norm=norm)
    bridge_gdf.plot(color="red", ax=ax, linewidth=3, zorder=5)

    edges.plot(
        column="water_depth",
        cmap=cmap,
        norm=norm,
        ax=ax,
        linewidth=3,
    )
    cbar = fig.colorbar(sm, ax=ax, extend="both", orientation="horizontal", shrink=0.5)
    cbar.set_label("Water depth (m)")
    edges.boundary.plot(ax=ax, color="black", markersize=3, zorder=5, label="nodes")
    raster.plot(ax=ax, cmap=cmap_b, norm=norm, zorder=3)
    ax.legend(loc="upper right")

    xmin, ymin, xmax, ymax = bbox.shape.bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # %%
    # fig.savefig("bridge_water_depth.png", dpi=1000, bbox_inches="tight")
    # %%
