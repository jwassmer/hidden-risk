# %%

import networkx as nx
import warnings
from rasterstats import zonal_stats
import osmnx as ox
import numpy as np
import rioxarray as riox
from rasterio.enums import Resampling

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd
import hashlib
from pathlib import Path

PATH_TO_GHSL_TIF = "data/GHS/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif"


def box_to_poly(long0, lat0, lat1, long1):
    """
    Create a shapely Polygon object representing a rectangle from the given coordinates.

    Parameters:
        long0 (float): The longitude of the lower-left corner of the rectangle.
        lat0 (float): The latitude of the lower-left corner of the rectangle.
        lat1 (float): The latitude of the upper-right corner of the rectangle.
        long1 (float): The longitude of the upper-right corner of the rectangle.

    Returns:
        shapely.geometry.Polygon: The Polygon object representing the rectangle.
    """
    return Polygon([[long0, lat0], [long1, lat0], [long1, lat1], [long0, lat1]])


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Generate finite Voronoi polygons based on a given Voronoi diagram.

    Parameters:
        vor: Voronoi object representing a Voronoi diagram.
        radius: Maximum distance from the Voronoi center to the finite polygons (default: None).

    Returns:
        tuple: A tuple containing two elements:
            - List of regions, where each region is represented by a list of vertices.
            - Numpy array of vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def compute_voronoi_polys_of_nodes(nodes):
    """
    Compute Voronoi polygons for a set of nodes.

    Parameters:
        nodes (geopandas.GeoDataFrame): The input nodes as a GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: A modified GeoDataFrame with added Voronoi polygon geometries.
    """
    vor_nodes = nodes.copy()
    points = np.array(list(zip(np.array(vor_nodes.x), np.array(vor_nodes.y))))
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    voronoi_polys = gpd.GeoSeries([Polygon(vertices[region]) for region in regions])
    voronoi_polys = voronoi_polys.intersection(mask)

    vor_nodes["voronoi"] = list(voronoi_polys)
    vor_nodes = vor_nodes.set_geometry("voronoi").set_crs(nodes.crs)
    return vor_nodes


def clip_to_gdf(voronoi_nodes, raster_path):
    """
    Clip a raster to the extent of a GeoDataFrame representing Voronoi nodes.

    Parameters:
        voronoi_nodes: GeoDataFrame representing Voronoi nodes.
        raster_path: Path to the raster file.

    Returns:
        xarray.DataArray: Clipped raster data.
    """
    # Read raster
    raster = riox.open_rasterio(raster_path)

    # update crs
    crs = raster.spatial_ref.crs_wkt
    voronoi_nodes = voronoi_nodes.to_crs(crs)

    # clip to convex hull of gdf
    clipped = raster.rio.clip_box(*voronoi_nodes.unary_union.bounds)
    clipped = clipped.rio.clip([voronoi_nodes.unary_union.convex_hull])
    return clipped


def upsample_raster(voronoi_nodes, raster):
    """
    upsample the clipped raster.

    Parameters:
        voronoi_nodes: GeoDataFrame representing Voronoi nodes.
        raster_path: Path to the raster file.

    Returns:
        xarray.DataArray: Upsampled raster data.
    """

    # update crs
    crs = raster.spatial_ref.crs_wkt
    voronoi_nodes = voronoi_nodes.to_crs(crs)

    # compute updscale factors
    y_res = int(
        np.floor(
            min(
                voronoi_nodes.voronoi.bounds["maxy"]
                - voronoi_nodes.voronoi.bounds["miny"]
            )
            / 2
        )
    )
    x_res = int(
        np.floor(
            min(
                voronoi_nodes.voronoi.bounds["maxx"]
                - voronoi_nodes.voronoi.bounds["minx"]
            )
            / 2
        )
    )

    if x_res == 0:
        x_res = 1
    if y_res == 0:
        y_res = 1

    # upsample raster
    up_sampled = raster.rio.reproject(
        raster.rio.crs,
        resolution=(y_res, x_res),
        resampling=Resampling.sum,
        nodata=np.nan,
    )

    return up_sampled


def pop_in_polygon(raster, geom):
    """
    Calculate the total population within a polygon geometry from a raster dataset.

    Parameters:
        raster: Raster dataset.
        geom: Polygon geometry.

    Returns:
        float: Total population within the polygon.
    """
    clipped_poly = raster.rio.clip([geom], drop=True)
    clipped_poly = clipped_poly.where(clipped_poly != clipped_poly.rio.nodata)
    return float(clipped_poly.sum(skipna=True).data)


def population_from_raster_to_gdf(
    gdf,
    raster_path=PATH_TO_GHSL_TIF,
):
    """
    Calculate population-related attributes for a GeoDataFrame based on a raster dataset.

    Parameters:
        gdf: GeoDataFrame containing Voronoi polygons.
        raster_path: Path to the raster dataset.

    Returns:
        gdf: GeoDataFrame with population-related attributes added.
    """
    org_crs = gdf.crs
    clipped_raster = clip_to_gdf(gdf, raster_path)

    # update crs
    crs = clipped_raster.spatial_ref.crs_wkt
    gdf = gdf.to_crs(crs)

    # upsampling raster would come here.
    # Memory intensive, thus leaving out atm. Maybe better though

    # warnings.filterwarnings("ignore")
    stats = zonal_stats(
        gdf["voronoi"],
        clipped_raster.data[0],
        affine=clipped_raster.rio.transform(),
        stats="sum",
        nodata=np.nan,
        all_touched=False,
    )

    gdf["population"] = [s["sum"] for s in stats]

    # warn if population smaller zero
    if gdf["population"].min() < 0:
        warnings.warn("At least one value for population smaller zero")
        gdf.loc[gdf["population"] < 0, "population"] = 0

    if gdf["population"].isna().sum() > 0:
        print("At least one nan value for population. Setting na to zero")
        gdf.loc[gdf["population"].isna(), "population"] = 0

    pop_dens = gdf["population"] / (gdf["voronoi"].area * 1e-6)
    vor_area = gdf["voronoi"].area * 1e-6

    gdf["population_density"] = pop_dens
    gdf["voronoi_area"] = vor_area
    gdf = gdf.to_crs(org_crs)
    return gdf


def raster_population_to_voronois(
    nodes,
    raster_path=PATH_TO_GHSL_TIF,
):
    voronoi_nodes = compute_voronoi_polys_of_nodes(nodes)
    voronoi_nodes = population_from_raster_to_gdf(voronoi_nodes, raster_path)

    return voronoi_nodes


def population_to_graph_nodes(graph):
    nodes = ox.graph_to_gdfs(graph, edges=False)
    voronoi_nodes = raster_population_to_voronois(nodes)

    nx.set_node_attributes(graph, voronoi_nodes["population"], "population")
    nx.set_node_attributes(
        graph, voronoi_nodes["population_density"], "population_density"
    )
    nx.set_node_attributes(graph, voronoi_nodes["voronoi"], "voronoi")

    return graph


# %%
# G = ox.graph_from_place('Cologne Germany', network_type='drive')

# G, vor_nodes = population_from_raster_to_graph(G, return_voronoi=True)
