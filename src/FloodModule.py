import shapely
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd

from scipy.spatial import Voronoi
from shapely.strtree import STRtree
from shapely.ops import unary_union, nearest_points

from src import PopulationFromRaster as pfr

PATH_TO_FLOOD_SHAPE = "data/EMSR517_AOI15_GRA_MONIT01_r1_RTP03_v3_vector/EMSR517_AOI15_GRA_MONIT01_observedEventA_r1_v3.shp"


def flood_footprint(PATH_TO_FLOOD_SHAPE=PATH_TO_FLOOD_SHAPE):
    """
    Read the flood shapefile and return the union of its geometries.

    Parameters:
        PATH_TO_FLOOD_SHAPE (str): Path to the flood shapefile.

    Returns:
        shapely.geometry.Polygon: Union of the flood geometries.
    """
    return unary_union(gpd.read_file(PATH_TO_FLOOD_SHAPE).geometry)


def poly_great_circle_vec(poly1, poly2):
    """
    Calculate the great circle distance between two polygons.

    Parameters:
        poly1 (shapely.geometry.Polygon): First polygon.
        poly2 (shapely.geometry.Polygon): Second polygon.

    Returns:
        float: Great circle distance between the polygons.
    """
    p1, p2 = nearest_points(poly1, poly2)
    return ox.distance.great_circle(p1.x, p1.y, p2.x, p2.y)


def edge_dist_to_flood(G, flood_poly):
    """
    Calculate the distance from each edge in the graph to the flood footprint.

    Parameters:
        G (networkx.MultiDiGraph): Input graph.
        flood_poly (shapely.geometry.Polygon): Flood footprint.

    Returns:
        networkx.MultiDiGraph: Graph with flood distances assigned to edges.
    """
    edg_polys = nx.get_edge_attributes(G, "geometry")

    edg_dist = {
        e: poly_great_circle_vec(flood_poly, epoly) for e, epoly in edg_polys.items()
    }

    nx.set_edge_attributes(G, edg_dist, "flood_dist")
    return G


def flooded_bridges(graph, flood_poly):
    """
    Identify bridges that are flooded within the flood footprint.

    Parameters:
        graph (networkx.MultiDiGraph): Input graph.
        flood_poly (shapely.geometry.Polygon): Flood footprint.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing flooded bridges.
    """
    graph = edge_dist_to_flood(graph, flood_poly)
    edges = ox.graph_to_gdfs(graph, nodes=False)

    flooded_bridges = edges[(edges["bridge"] == "yes") & (edges["flood_dist"] == 0)]
    return flooded_bridges


def flooded_roads(graph, flood_poly):
    """
    Identify roads that are flooded within the flood footprint.

    Parameters:
        graph (networkx.MultiDiGraph): Input graph.
        flood_poly (shapely.geometry.Polygon): Flood footprint.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing flooded roads.
    """
    graph = edge_dist_to_flood(graph, flood_poly)
    edges = ox.graph_to_gdfs(graph, nodes=False)

    flooded_roads = edges[(edges["flood_dist"] == 0)]
    return flooded_roads


"""
def node_distance_to_flood(G, flood_poly):
    nodes = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
    # flood_poly = unary_union(flood.geometry)

    min_flood_poly_dist = {}
    for idx, n in nodes.iterrows():
        p1, p2 = nearest_points(flood_poly, n.geometry)
        min_flood_poly_dist[idx] = ox.distance.great_circle_vec(p1.x, p1.y, p2.x, p2.y)

    nx.set_node_attributes(G, min_flood_poly_dist, "flood_dist")
    return G


def vor_distance_to_flood(G, gdf_flood):
    gs_nodes = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
    points = list(gs_nodes.geometry)

    xs = [point.x for point in points]
    ys = [point.y for point in points]

    boundary_shape = ox.utils_geo.bbox_to_poly(max(ys), min(ys), max(xs), min(xs))

    coords = np.array(
        [p.coords[0] for p in gs_nodes.geometry]
    )  # points_to_coords(gs_nodes.geometry)

    vor = Voronoi(coords)
    regions, vertices = pfr.voronoi_finite_polygons_2d(vor)
    geoms = [shapely.geometry.Polygon(vertices[region]) for region in regions]

    # geoms = list(region_polys.values())
    min_flood_poly_dist = {}

    for idx, n in enumerate(G.nodes()):
        poly = geoms[idx]
        # poly_size[idx] = geom.area

        min_flood_poly_dist[n] = np.inf
        for flood in gdf_flood.geometry:
            flood_poly_dist = flood.distance(poly)

            if flood_poly_dist < min_flood_poly_dist[n]:
                min_flood_poly_dist[n] = flood_poly_dist

    nx.set_node_attributes(G, min_flood_poly_dist, "flood_dist")
    return G








def flood_area(G, gdf_flood, travel_time_factor=np.inf):
    G.travel_time_factor = travel_time_factor

    G = edge_dist_to_flood(G, gdf_flood)
    G = node_distance_to_flood(G, gdf_flood)

    nodes, edges = ox.graph_to_gdfs(G)

    nodes["flood"] = nodes.flood_dist == 0
    edges["flood"] = edges.flood_dist == 0

    nx.set_node_attributes(G, values=nodes["flood"], name="flood")
    nx.set_edge_attributes(G, values=edges["flood"], name="flood")
    # length = {k : travel_time_factor*G.edges[k]['travel_time'] if v == 0 else G.edges[k]['travel_time'] for k, v in edg_dist_to_flood.items()}
    # nx.set_edge_attributes(G, length, 'travel_time_flood')

    # nx.set_edge_attributes(G, {}, 'load')

    return G
"""
