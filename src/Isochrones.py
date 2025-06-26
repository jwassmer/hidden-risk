import networkx as nx
import osmnx as ox
import numpy as np
from shapely import geometry
import geopandas as gpd
from shapely.ops import unary_union, polygonize, polygonize_full
from scipy.spatial import Delaunay


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Parameters
    ----------
        points (iterable): Iterable container of points.
        alpha (float): Alpha value to influence the gooeyness of the border. Smaller numbers don't fall inward as much as larger numbers. Too large, and you lose everything!

    Returns
    -------
        geometry.Polygon: The alpha shape polygon.
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = (
        (triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2
        + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2
    ) ** 0.5
    b = (
        (triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2
        + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2
    ) ** 0.5
    c = (
        (triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2
        + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2
    ) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize_full(m))

    return unary_union(triangles)  # , edge_points


def isochrones_from_shortest_path(
    G, centernodes, time_arr, weight="travel_time", alpha=100, buffer=1e-3
):
    """
    Compute isochrones (areas of equal travel time) from a set of center nodes using shortest path distances.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        centernodes (list): List of center nodes to compute isochrones from.
        time_arr (list): List of time values representing the desired isochrone boundaries.
        weight (str): The edge weight attribute to use for computing shortest paths. Default is 'travel_time'.
        alpha (float): Alpha value for alpha shape computation. Default is 100.
        buffer (float): Buffer distance for the isochrone polygons. Default is 1e-3.

    Returns
    -------
        geopandas.GeoDataFrame: A GeoDataFrame containing the isochrone polygons.
    """
    nx.set_node_attributes(
        G,
        values=nx.multi_source_dijkstra_path_length(
            G, centernodes, weight=weight, cutoff=time_arr[-1]
        ),
        name="iso_time",
    )
    # nodes = ox.graph_to_gdfs(G, edges=False)
    return isochrone_polys(G, "iso_time", time_arr, alpha=alpha, buffer=buffer)
    # return gpd.GeoDataFrame({'time':time_arr[1:], f'{weight}_isochrone':gpd.GeoSeries(iso_shapes).buffer(1e-3)})


def isochrone_polys(G, nodeweight, time_arr, alpha=100, buffer=1e-3):
    """
    Compute isochrone polygons from nodal weights and time values.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        nodeweight (str): The nodal weight attribute to use for filtering nodes.
        time_arr (list): List of time values representing the desired isochrone boundaries.
        alpha (float): Alpha value for alpha shape computation. Default is 100.
        buffer (float): Buffer distance for the isochrone polygons. Default is 1e-3.

    Returns
    -------
        geopandas.GeoDataFrame: A GeoDataFrame containing the isochrone polygons.
    """
    nodes = ox.graph_to_gdfs(G, edges=False)
    iso_shapes = []
    time_arr = sorted(time_arr, reverse=True)
    # time_attr = f'{weight}_from_{emergency}'
    for i in range(len(time_arr) - 1):
        t1, t0 = time_arr[i], time_arr[i + 1]
        sub_nodes = nodes[(nodes[nodeweight] >= t0) & (nodes[nodeweight] < t1)]
        shape = alpha_shape(sub_nodes.geometry, alpha)

        if len(sub_nodes) >= 1:
            # sub_graph = G.subgraph(sub_nodes.index)
            # sub_edges = ox.graph_to_gdfs(sub_graph, nodes=False)
            sub_edges = [
                geom for _, _, geom in G.edges(sub_nodes.index, data="geometry")
            ]
            edgshape = unary_union(sub_edges)

            shape = unary_union([shape, edgshape])
        iso_shapes.append(shape)
    return gpd.GeoDataFrame(
        {
            "time": time_arr[1:],
            f"isochrone_poly": gpd.GeoSeries(iso_shapes).buffer(buffer),
        }
    )
