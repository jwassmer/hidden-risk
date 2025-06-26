# %%
# ------------------ PACKAGES ---------------------------#
import osmnx as ox
import networkx as nx
import warnings
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import pandas as pd
from copy import copy, deepcopy


from src import TrafficCentrality2 as tc
from src import OSMparser as osmp
from src import EmergencyModule as em
from src import EffectiveVelocities2 as ev
from src import WetRoads as wr
import pickle

# %%
# ------------------ SET GLOBALS ---------------------------#

PATH_TO_OSMFILES = "data/osmfiles/latest"


# %%
# ------------------ FUNCTIONS ---------------------------#
class RoadNetwork:
    """A RoadNetwork that stores a MuliDiGraph, the popuation size per node, as well as the geometries of the nodes and edges.

    Parameters
    ----------
    graph (networkx.Graph): The road network graph.
    bound_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the boundaries of the road network.
    nodes (geopandas.GeoDataFrame): A GeoDataFrame representing the nodes of the road network.
    edges (geopandas.GeoDataFrame): A GeoDataFrame representing the edges of the road network.
    population (float): The total population of the road network.
    area (float): The area covered by the road network.
    graph_r (None or networkx.Graph): An optional reversed graph of the road network.
    path (str): The path to the OSM files used for constructing the road network.
    osmpbf (str): The path to the OSM PBF file used for constructing the road network.
    """

    def __init__(self, osmpbf=None, highway_filter=None, **kwargs):
        """Initialise a RoadNetwork.

        Parameters
        ----------

        osmpbf (str): The path to the OSM PBF file used for constructing the road network.
        highway_filter (str): OSM filter that is of the form f"w/highway={tag0,tag1,...,tag_N}", where the tags correspond to OSM highway tags.
        **kwargs : keyword arguments. Set to either place="some_place" or bbox="[north,south,west,east]".

        Examples
        --------
        >>> R = rn.RoadNetwork(
                osmpbf="germany.osm.pbf",
                highway_filter=f"w/highway={','.join(["primary","secondary","tertiary"])}",
                bbox=[50.0, 51.2,  6.4, 7.9],
            )
        >>> R = rn.RoadNetwork(
                osmpbf="germany.osm.pbf",
                highway_filter=f"w/highway={','.join(["motorway","trunk","unclassified"])}",
                place="Cologne,Germany",
            )
        """
        self.bound_gdf = osmp.generate_region_gdf(**kwargs)
        self.graph, self.path = osmp.road_graph_with_populations(
            gdf=self.bound_gdf,
            path_to_osmpbf=PATH_TO_OSMFILES,
            osmpbf=osmpbf,
            highway_filter=highway_filter,
            retain_all=False,
        )

        self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
        self.population = self.nodes["population"].sum()
        self.area = (
            self.nodes.set_geometry("voronoi", crs=self.nodes.crs)
            .to_crs("ESRI:54009")
            .area.sum()
            * 1e-6
        )
        # self.graph_r = None
        self.osmpbf = osmpbf

        print("num(nodes)=", len(self.graph.nodes()))
        print("num(edges)=", len(self.graph.edges()))

    def loads(
        self,
        weight,
        cache=True,
        normalized=False,
        cutoff="default",
        fit="lognorm",
        **kwargs,
    ):
        """
        Computes the load (a traffic based centrality measure) for each edge in the graph and sets the values as an edge attribute.

        Parameters
        ----------
        weight  (str) : The edge weight which the shortest path search is based on. Set to 'travel_time' or 'length'.
        cache (bool, optional): Whether to cache results. Defaults to True.
        cutoff (str or int, optional) : The cutoff value when to stop the shortest path search. Defaults to 'default'.
                When set to 'deafult' will only traverse the graph until a maximum shortest path of 3600, if weight='travel_time'
                or to 6000 if weight='length'.
        **kwargs : keyword arguments. Possible options are cpu_cores=N, to run the code in paralell with N cores and
                jobs_per_cpu=M to set the amount of node searches per CPU start. Defaults to 5.
        """
        load_dict = tc.interaction_betweenness_centrality(
            self.graph,
            cache=cache,
            cutoff=cutoff,
            weight=weight,
            return_graph=False,
            normalized=normalized,
            fit=fit,
            **kwargs,
        )
        nx.set_edge_attributes(
            self.graph,
            load_dict,
            "load",
        )
        self.edges["load"] = load_dict  # list(load_dict.values())

    def effective_velocities(self, gamma):
        """Computes the effective velocity of every edge, that is the average traffic affected velocity a vehicle can travel through a road.
        Sets the values as edge attribute.

        Parameters
        ----------
        gamma (str) : The fraction of individuals in a region that are on the road. 0.15 is considered high traffic.

        """
        self.graph = ev.effective_velocities(self.graph, gamma)
        self.nodes, self.edges = ox.graph_to_gdfs(self.graph)

    def effective_spillover_velocities(self, gamma, cache=True):
        """Reroutes the excess load of every edge to incoming edges and computes the effective velocity of every edge.
        Sets the values as edge attribute.

        Parameters
        ----------
        gamma (str) : The fraction of individuals in a region that are on the road. 0.15 is considered high traffic.

        """
        self.graph = ev.effective_spillover_velocities(self.graph, gamma, cache=cache)
        self.edges = ox.graph_to_gdfs(self.graph, nodes=False)

    def add_pois(self, key, tag, merge_km=1_000, overwrite=False):
        """Adds points of interest such as hospitals to graph nodes and sets as node attributes.

        Parameters
        ----------
        key (str) : osm key. Example 'amenity' or 'emergency'
        tag (str) : osm tag. Example 'restaurant' or 'hospital'

        """
        nodes = self.nodes
        poi_gdf = em.poi_from_gdf(
            PATH_TO_OSMFILES, self.osmpbf, self.bound_gdf, key, tag, overwrite=overwrite
        )

        if key == "amenity" and tag == "hospital":
            poi_gdf = em.clean_hospital_names(poi_gdf)

        poi_gdf = em.merge_similar(poi_gdf, key, merge_km)

        # set true/false for poi nodes
        if key not in nodes.columns:
            nodes[key] = np.full(len(nodes), None)
            nodes[f"{key}_name"] = np.full(len(nodes), None)

        geoms = poi_gdf.geometry
        node_tree = STRtree(nodes.geometry)

        for j, geo in enumerate(geoms):
            # find nearest emegreny to road i
            centroid_pt = geo.centroid
            idx = node_tree.nearest(centroid_pt)

            # set poi to node
            nodes.iloc[idx, nodes.columns.get_loc(key)] = poi_gdf[key].iloc[j]
            nodes.iloc[idx, nodes.columns.get_loc(f"{key}_name")] = poi_gdf[
                "name"
            ].iloc[j]

        nx.set_node_attributes(self.graph, nodes[key], name=key)
        nx.set_node_attributes(self.graph, nodes[f"{key}_name"], name=f"{key}_name")

    def geometries_from_osmid(self, id_arr, polygon=None):
        feat_gdf = osmp.features_from_osmids(
            PATH_TO_OSMFILES, self.osmpbf, self.bound_gdf, id_arr, polygon=polygon
        )
        return feat_gdf

    # def geometries_from_tags(self, tags):
    #    feat_gdf = osmp.features_from_tags(
    #        PATH_TO_OSMFILES, self.osmpbf, self.bound_gdf, tags
    #    )
    #    return feat_gdf
    def get_bridges(self):
        wr.get_bridge_polys(self)

    def water_depth(self, raster, polygon=None):
        wr.water_depth(self, raster, polygon=polygon)

    def remove_edges(self, edges):
        """Effectively removes edge from graph, by setting the speed limit to zero, the travel time to inf and the length to inf,
        ensuring that the shortest path algorithm will never visit this edge.

        Parameters
        ----------
        edges (iterator) : Iterator for edges, e.g. list of edge indices.
        """
        graph = self.graph
        self.edges["removed"] = pd.Series(dtype="bool")

        for e in edges:
            u, v, k = e
            graph[u][v][k]["speed_kph"] = 0
            graph[u][v][k]["speed_ms"] = 0
            graph[u][v][k]["maxspeed"] = 0
            graph[u][v][k]["travel_time"] = np.inf
            graph[u][v][k]["length"] = np.inf
            graph[u][v][k]["removed"] = True

            self.edges.loc[(u, v, k), "removed"] = True
            self.edges.loc[(u, v, k), "speed_kph"] = 0
            self.edges.loc[(u, v, k), "speed_ms"] = 0
            self.edges.loc[(u, v, k), "maxspeed"] = 0
            self.edges.loc[(u, v, k), "travel_time"] = np.inf
            self.edges.loc[(u, v, k), "length"] = np.inf

        self.graph = graph
        # self.nodes, self.edges = ox.graph_to_gdfs(graph)

    def copy(self):
        return deepcopy(self)


# %%


def remove_edges(graph, edg):
    graph_r = graph.copy()
    for e in edg:
        u, v, k = e
        graph_r[u][v][k]["speed_kph"] = 0
        graph_r[u][v][k]["speed_ms"] = 0
        graph_r[u][v][k]["maxspeed"] = 0
        graph_r[u][v][k]["travel_time"] = np.inf
        graph_r[u][v][k]["length"] = np.inf
        graph_r[u][v][k]["removed"] = True
    return graph_r


def convert_road_to_bicycle_lane(graph, edg):
    graph_r = graph.copy()
    for e in edg:
        u, v, k = e
        graph_r[u][v][k]["speed_kph"] = 5
        graph_r[u][v][k]["maxspeed"] = 5
        graph_r[u][v][k]["speed_ms"] = 5 * 1000 / 60 / 60
        graph_r[u][v][k]["travel_time"] = (
            graph_r[u][v][k]["length"] / graph_r[u][v][k]["speed_ms"]
        )
        graph_r[u][v][k]["bike_lane"] = True

    return graph_r


def digraph_to_gdfs(
    G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True
):
    """
    Convert a MultiDiGraph to node and/or edge GeoDataFrames.

    This function is the inverse of `graph_from_gdfs`.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y attributes
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
        is indexed by osmid and gdf_edges is multi-indexed by u, v, key
        following normal MultiDiGraph structure.
    """
    crs = G.graph["crs"]

    if nodes:
        if not G.nodes:  # pragma: no cover
            msg = "graph contains no nodes"
            raise ValueError(msg)

        nodes, data = zip(*G.nodes(data=True))

        if node_geometry:
            # convert node x/y attributes to Points for geometry column
            geom = (Point(d["x"], d["y"]) for d in data)
            gdf_nodes = gpd.GeoDataFrame(
                data, index=nodes, crs=crs, geometry=list(geom)
            )
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes)

        gdf_nodes.index.rename("osmid", inplace=True)

    if edges:
        if not G.edges:  # pragma: no cover
            msg = "graph contains no edges"
            raise ValueError(msg)

        u, v, data = zip(*G.edges(data=True))

        if fill_edge_geometry:
            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def _make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]

                # otherwise
                return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

            geom = map(_make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if "geometry" not in gdf_edges.columns:
                # if no edges have a geometry attribute, create null column
                gdf_edges = gdf_edges.set_geometry([None] * len(gdf_edges))
            gdf_edges = gdf_edges.set_crs(crs)

        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        # gdf_edges["key"] = k
        gdf_edges.set_index(["u", "v"], inplace=True)

    if nodes and edges:
        return gdf_nodes, gdf_edges

    if nodes:
        return gdf_nodes

    if edges:
        return gdf_edges


# %%

if __name__ == "__main__":

    # ------------------ EXAMPLES ---------------------------#
    # west, east = 6.4, 7.9
    # south, north = 50.0, 51.2

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
        "residential",
    ]
    highway_filter = f"w/highway={','.join(driving_tags)}"

    R = RoadNetwork(
        osmpbf="ger-buffered-200km.osm.pbf",
        highway_filter=highway_filter,
        place="Cologne,Germany",
    )

    # R.geometries_from_tags({"building": True})
    G = R.graph
    # Save the graph using pickle
    with open("cologne_full_graph.pickle", "wb") as f:
        pickle.dump(G, f)


# %%
