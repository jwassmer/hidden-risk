import numpy as np
import geopandas as gpd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.geometry import box


def voronoi_polygons(voronoi, radius=None):
    """
    Construct polygons for each Voronoi region.
    If 'radius' is specified, it limits the size of infinite Voronoi regions.
    """
    if voronoi.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi.vertices.tolist()

    center = voronoi.points.mean(axis=0)
    if radius is None:
        radius = voronoi.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = voronoi.points[p2] - voronoi.points[p1]  # Tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal

            midpoint = voronoi.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish the polygon
        new_regions.append(new_region.tolist())

    return [
        Polygon(np.array([new_vertices[v] for v in region])) for region in new_regions
    ]


def voronoi_gdf(points, minx, miny, maxx, maxy):
    """
    Create a GeoDataFrame of Voronoi polygons from a set of points.
    """
    vor = Voronoi(points)
    vor_polygons = voronoi_polygons(vor)
    gdf = gpd.GeoDataFrame(geometry=vor_polygons)

    bounding_box = box(minx, miny, maxx, maxy)

    gdf["geometry"] = gdf.geometry.apply(lambda geom: geom.intersection(bounding_box))

    return gdf


def voronoi_to_gdf(gdf, bounding_box=None):
    geoms = gdf.geometry
    if bounding_box is None:
        minx, miny, maxx, maxy = gpd.GeoSeries(geoms).total_bounds
        bounding_box = box(minx, miny, maxx, maxy)

    points = list(zip(geoms.x, geoms.y))

    vor = Voronoi(points)
    vor_polygons = voronoi_polygons(vor)
    gdf.loc[:, "voronoi"] = vor_polygons

    gdf["voronoi"] = gdf["voronoi"].apply(lambda geom: geom.intersection(bounding_box))

    return gdf
