# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import requests
from io import StringIO

# %%


class location:
    def __init__(self, name=None, gdf=None):
        if name is not None:
            self.name = name
        if gdf is None:
            self.gdf = get_location_shape(name)
        else:
            self.gdf = gdf
        self.shape = self.gdf.iloc[0]["geometry"]
        self.bbox = bbox(*self.gdf.total_bounds).shape

    def buffer(self, buffer_size_km):
        return buffer_gdf(self.gdf, buffer_size_km)


class bbox:
    def __init__(self, west, east, south, north, crs="EPSG:4326"):
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.crs = crs

        self.gdf = gdf_from_bbox(
            [self.west, self.south, self.east, self.north], crs=self.crs
        )

        self.shape = self.gdf.iloc[0]["geometry"]

    def buffer(self, buffer_size_km):
        return buffer_gdf(self.gdf, buffer_size_km)


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

    response = requests.get(url)

    if response.status_code == 200:
        # data = response.json()
        shape = gpd.read_file(StringIO(response.text))
    else:
        print(f"Failed to fetch data. HTTP response code: {response.status_code}")

    admin_boundary = shape[shape["type"] == "administrative"]

    # Set the coordinate reference system of the shape to epsg:4326
    admin_boundary = admin_boundary.to_crs("epsg:4326")

    return admin_boundary


def create_square_polygon(lon, lat, width=100):
    height = width
    # Half the width and height in meters
    half_width = width / 2
    half_height = height / 2

    # Calculate the change in latitude for half the height (in degrees)
    delta_lat = (half_height / 6371000) * (180 / np.pi)

    # Calculate the change in longitude for half the width, adjusted for latitude (in degrees)
    delta_lon = (half_width / (6371000 * np.cos(np.radians(lat)))) * (180 / np.pi)

    # Determine the four corners of the rectangle
    minx = lon - delta_lon
    maxx = lon + delta_lon
    miny = lat - delta_lat
    maxy = lat + delta_lat

    # Define the polygon using the four corners
    polygon = [
        (minx, miny),  # Bottom-left corner
        (minx, maxy),  # Top-left corner
        (maxx, maxy),  # Top-right corner
        (maxx, miny),  # Bottom-right corner
        (minx, miny),  # Close the polygon
    ]

    return gpd.GeoSeries(Polygon(polygon), crs="EPSG:4326")


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


def buffer_geometry(geom, buffer_size_km, crs="EPSG:4326"):
    # Convert input geometry to a GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)

    # Transform to a projected coordinate system suitable for distance measurements
    # For simplicity, using a global equal-area projection (Mollweide)
    gdf = gdf.to_crs("ESRI:54009")

    # Buffer by 40km
    gdf["geometry"] = gdf["geometry"].buffer(buffer_size_km * 1e3)

    # Transform back to EPSG:4326
    gdf = gdf.to_crs(crs)

    return gdf  # gdf.iloc[0]["geometry"]


def buffer_gdf(gdf, buffer_size_km):
    # Transform to a projected coordinate system suitable for distance measurements
    # For simplicity, using a global equal-area projection (Mollweide)
    crs = gdf.crs
    gdf = gdf.to_crs("ESRI:54009")

    # Buffer by 40km
    gdf["geometry"] = gdf["geometry"].buffer(buffer_size_km * 1e3)

    # Transform back to EPSG:4326
    gdf = gdf.to_crs(crs)

    return gdf  # gdf.iloc[0]["geometry"]


import geopandas as gpd


def geodataframe_dimensions(geodf):
    """
    Calculate the height and width of a GeoDataFrame in kilometers.

    Parameters:
    geodf (GeoDataFrame): A GeoPandas GeoDataFrame.

    Returns:
    height (float): Height of the GeoDataFrame in kilometers.
    width (float): Width of the GeoDataFrame in kilometers.
    """
    # Calculate the bounds of the GeoDataFrame
    minx, miny, maxx, maxy = geodf.geometry.total_bounds

    # Calculate the height and width in kilometers
    height = haversine(minx, miny, minx, maxy)
    width = haversine(minx, miny, maxx, miny)

    return height, width


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Parameters:
    lon1, lat1, lon2, lat2 (float): Longitude and latitude of the two points.

    Returns:
    distance (float): Distance between the two points in kilometers.
    """
    from math import radians, sin, cos, sqrt, atan2

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers
    distance = r * c

    return distance


def gdf_from_bbox(bbox, crs="EPSG:4326"):
    """
    Create a GeoDataFrame from a bounding box.

    Parameters:
    bbox (list): A bounding box in the form [minx, miny, maxx, maxy].

    Returns:
    gdf (GeoDataFrame): A GeoDataFrame with a rectangular Polygon geometry and CRS EPSG:4326.
    """
    # Create a DataFrame with a single row of data
    data = {"geometry": [Polygon.from_bounds(*bbox)]}
    df = pd.DataFrame(data, index=[0])

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, crs=crs)

    return gdf


# %%
"""from src import OSMparser as op

berlin = op.get_location_shape("Berlin, Germany")


geodataframe_dimensions(berlin)"""

# %%
