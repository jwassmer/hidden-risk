import numpy as np
import networkx as nx
import osmnx as ox
import shapely

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from scipy.spatial import Voronoi

import matplotlib as mpl
from shapely.geometry import Polygon

from src import PopulationFromRaster as pfr


def scientific(x):
    # Calculate the order of magnitude
    n = int(np.floor(np.log10(abs(x))))
    # Calculate the leading coefficient
    a = x / (10**n)
    # Return the formatted string in LaTeX format
    return f"${a:.2f} \\times 10^{{{n}}}$"


def generate_bbox(pt, width):
    """
    Generates a bounding box within a given area and a reference geometry.

    Args:
        area (float): The desired area of the bounding box.
        location_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the reference geometry.

    Returns:
        tuple: A tuple containing the north, south, west, and east coordinates of the generated bounding box.
    """

    # Calculate the size of the bounding box based on the area
    box_width = width
    box_height = width / 2

    # Calculate random coordinates within the bounding box
    north = pt[1] + box_height / 2
    south = pt[1] - box_height / 2
    east = pt[0] + box_width / 2
    west = pt[0] - box_width / 2

    # Create a polygon from the generated coordinates
    bbox_polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

    return bbox_polygon


# Function to generate random points within a circle
def generate_points_in_circle(center, radius, num_points, seed=None):
    np.random.seed(seed)  # Set the seed
    points = []
    while len(points) < num_points:
        # Generate random point within the bounding box of the circle
        x = np.random.uniform(center[0] - radius, center[0] + radius)
        y = np.random.uniform(center[1] - radius, center[1] + radius)

        # Check if the point is within the circle
        distance_to_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        if distance_to_center <= radius:
            points.append((x, y))
    return points


def shortest_path_from_key_tag(graph, weight, key, tag, cutoff=None):
    """
    Compute shortest paths from nodes with a specific key and tag to all other nodes in the graph.

    Parameters:
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths.
        key (str): The key attribute to filter nodes.
        tag: The tag value to filter nodes.
        cutoff: Optional parameter specifying the maximum path length. Default is None.

    Returns:
        networkx.Graph: The modified graph with added node attributes representing the shortest path lengths.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)
    emergency_nodes = nodes[nodes[key] == tag]
    if len(nx.get_edge_attributes(graph, weight)) > 0:
        path_length_emerg = nx.multi_source_dijkstra_path_length(
            graph, list(emergency_nodes.index), weight=weight, cutoff=cutoff
        )
    else:
        print(f"graph has no weight {weight}.")
        return

    nodes[f"{weight}_from_{key}_{tag}"] = path_length_emerg
    nx.set_node_attributes(graph, path_length_emerg, f"{weight}_from_{key}_{tag}")
    return graph


def shortest_path_to_key_tag(graph, weight, key, tag, cutoff=None):
    """
    Compute shortest paths from all other nodes in the graph to nodes with a specific key and tag.

    Parameters:
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths.
        key (str): The key attribute to filter nodes.
        tag: The tag value to filter nodes.
        cutoff: Optional parameter specifying the maximum path length. Default is None.

    Returns:
        networkx.Graph: The modified graph with added node attributes representing the shortest path lengths.
    """
    G_reversed = graph.reverse()
    nodes = ox.graph_to_gdfs(G_reversed, edges=False)
    emergency_nodes = nodes[nodes[key] == tag]
    path_length_emerg = nx.multi_source_dijkstra_path_length(
        G_reversed, list(emergency_nodes.index), weight=weight, cutoff=cutoff
    )

    nodes[f"{weight}_to_{key}_{tag}"] = path_length_emerg
    nx.set_node_attributes(graph, path_length_emerg, f"{weight}_to_{key}_{tag}")
    return graph


def shortest_path_key_tag(
    graph,
    direction="from",
    weight="spillover_travel_time",
    key="amenity",
    tag="fire_station",
    inplace=True,
):
    """
    Compute shortest paths from or to nodes with a specific key and tag in the graph.

    Parameters:
        graph (networkx.Graph): The input graph.
        direction (str): Direction of the shortest paths. Can be 'from' or 'to'. Default is 'from'.
        weight (str): The edge weight attribute to use for computing shortest paths. Default is 'spillover_travel_time'.
        key (str): The key attribute to filter nodes. Default is 'amenity'.
        tag: The tag value to filter nodes. Default is 'fire_station'.
        inplace (bool): If True, modify the input graph in-place. If False, return a modified copy of the graph. Default is True.

    Returns:
        networkx.Graph or None: The modified graph if inplace=True, otherwise returns None.
    """
    if direction == "from":
        if inplace:
            shortest_path_from_key_tag(graph, weight, key, tag)
        else:
            return shortest_path_from_key_tag(graph, weight, key, tag)
    elif direction == "to":
        if inplace:
            shortest_path_to_key_tag(graph, weight, key, tag)
        else:
            return shortest_path_to_key_tag(graph, weight, key, tag)
    else:
        print("Invalid direction kwd . Use -from- or -to-.")


def dict_substract(d1, d2):
    """
    Subtract the values of two dictionaries with numeric values element-wise.

    Parameters:
        d1 (dict): First dictionary.
        d2 (dict): Second dictionary.

    Returns:
        dict: A new dictionary with element-wise subtraction of the values.
    """
    return {k: v - d2[k] for k, v in d1.items()}


def commuter_hours(G):
    """
    Compute the commuting hours for each edge in the graph based on spillover travel time and load attributes.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        list: A list of commuting hours for each edge in the graph.
    """
    seconds = nx.get_edge_attributes(G, "spillover_travel_time")
    comuters = nx.get_edge_attributes(G, "spillover_load")

    return [s * comuters[k] / 60 / 60 for k, s in seconds.items()]


def LODFs(graph, graph_r, kwd="load"):
    """
    Compute Line Outage Distribution Factors (LODFs) between two graphs based on a given attribute.

    Parameters:
        graph (networkx.Graph): The original graph.
        graph_r (networkx.Graph): The modified graph.
        kwd (str): The attribute to use for computing LODFs. Default is 'load'.

    Returns:
        numpy.ndarray: An array of LODF values for each edge in the modified graph.
    """
    loads = np.array(list(nx.get_edge_attributes(graph, kwd).values()))
    loads_r = np.array(list(nx.get_edge_attributes(graph_r, kwd).values()))

    loads_diff = loads_r - loads

    lodfs = loads_diff / loads
    nx.set_edge_attributes(graph_r, dict(zip(graph_r.edges, lodfs)), "LODF")
    return lodfs


def finite_sum(lst):
    """
    Compute the sum of finite values in a list.

    Parameters:
        lst (list): The input list.

    Returns:
        float: The sum of finite values in the list.
    """
    lst = np.array(lst)
    return sum(lst[np.isfinite(lst)])


def finite_mean(lst):
    """
    Compute the mean of finite values in a list.

    Parameters:
        lst (list): The input list.

    Returns:
        float: The mean of finite values in the list.
    """
    lst = np.array(lst)
    return np.mean(lst[np.isfinite(lst)])


def finite_max(lst):
    """
    Compute the maximum value among finite values in a list.

    Parameters:
        lst (list): The input list.

    Returns:
        float: The maximum value among finite values in the list.
    """
    lst = np.array(lst)
    return max(lst[np.isfinite(lst)])


def finite_min(lst):
    """
    Compute the minimum value among finite values in a list.

    Parameters:
        lst (list): The input list.

    Returns:
        float: The minimum value among finite values in the list.
    """
    lst = np.array(lst)
    return min(lst[np.isfinite(lst)])


def voronoi_region_of_nodes(nodes):
    """
    Generates Voronoi regions for a set of nodes in a road network.

    Args:
        nodes (geopandas.GeoDataFrame): A GeoDataFrame representing the nodes.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with added Voronoi regions for each node.
    """
    coords = np.array([p.coords[0] for p in nodes.geometry])
    vor = Voronoi(coords)
    regions, vertices = pfr.voronoi_finite_polygons_2d(vor)

    polys = [shapely.geometry.Polygon(vertices[region]) for region in regions]
    nodes["Voronoi"] = polys
    return nodes


def marker_from_svg(path):
    """
    Creates a marker from an SVG file.

    Args:
        path (str): The path to the SVG file.

    Returns:
        matplotlib.path.Path: A path representing the marker.
    """
    path, attributes = svg2paths(path)
    marker = parse_path(attributes[0]["d"])
    marker.vertices -= marker.vertices.mean(axis=0)
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    marker = marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))
    return marker


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    Adds a scale bar to a plot.

    Args:
        ax (matplotlib.axes.Axes): The axes to draw the scale bar on.
        length (float, optional): The length of the scale bar in km.
        location (tuple, optional): The center of the scale bar in axis coordinates.
        linewidth (int, optional): The thickness of the scale bar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ["1", "2", "5"]:
                return int(x)
            else:
                return scale_number(x - 10**ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color="k", linewidth=linewidth)
    # Plot the scalebar label
    ax.text(
        sbx,
        sby,
        str(length) + " km",
        transform=tmc,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
