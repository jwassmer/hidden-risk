# %%
import networkx as nx
import numpy as np
import osmnx as ox
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import logging
import pickle


def map_highway_to_number_of_lanes(highway_type):
    """
    Map a highway type to the corresponding number of lanes.

    Parameters:
        highway_type (str): The highway according to OSM tags.

    Returns:
        int: The number of lanes.
    """
    if highway_type == "motorway" or highway_type == "trunk":
        return 4
    elif highway_type == "primary":
        return 3
    elif (
        highway_type == "secondary"
        or highway_type == "motorway_link"
        or highway_type == "trunk_link"
        or highway_type == "primary_link"
    ):
        return 2
    else:
        return 1


def set_number_of_lanes(G):
    """
    Set the number of lanes attribute for each edge in the graph.

    Parameters:
        G (networkx.MultiDiGraph): Input road network graph.

    Returns:
        networkx.MultiDiGraph: Updated road network graph with the 'lanes' attribute set for each edge.
    """
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges["lanes"] = [
        (
            np.mean(list(map(float, v.iloc[0])))
            if isinstance(v.iloc[0], list)
            else (
                isinstance(v.iloc[0], float)
                if isinstance(v.iloc[0], str)
                else (
                    map_highway_to_number_of_lanes(v.iloc[1])
                    if np.isnan(v.iloc[0])
                    else v.iloc[0]
                )
            )
        )
        for k, v in edges[["lanes", "highway"]].iterrows()
    ]
    nx.set_edge_attributes(G, edges["lanes"], "lanes")
    return G


def peak_loads(G, gamma):
    """
    Compute the peak load under occupancy gamma for each edge in the road network graph.

    Parameters:
        G (networkx.MultiDiGraph): Input road network graph.
        gamma (float): Occupancy rate of road network.

    Returns:
        networkx.MultiDiGraph: Updated road network graph with the 'peak_load' attribute set for each edge.
    """
    Gc = G.copy()
    # Gc = set_number_of_lanes(Gc)
    pop = sum(nx.get_node_attributes(Gc, "population").values())
    loads = nx.get_edge_attributes(Gc, "load")

    tot_load = sum(loads.values())
    peak_load = {k: gamma * pop * (v / tot_load) for k, v in loads.items()}

    nx.set_edge_attributes(Gc, peak_load, "peak_load")
    return Gc


def load_capacities(G):
    """
    Compute the capacity for each edge in the road network graph.

    Parameters:
        G (networkx.MultiDiGraph): Input road network graph.

    Returns:
        networkx.MultiDiGraph: Updated road network graph with the 'peak_load' and 'capacity' attributes set for each edge.
    """
    Gc = G.copy()
    # Gc = set_number_of_lanes(Gc)

    lanes = nx.get_edge_attributes(Gc, "lanes")
    lanes_mod = dict((k, max(v - 1, 1)) for k, v in lanes.items())

    lengths = nx.get_edge_attributes(Gc, "length")

    v_min_kph = 5  # kmh
    v_min_mps = v_min_kph * 1000 / 60 / 60  # mps
    d = 5  # meter

    capacities = {
        e: lengths[e] * lanes_mod[e] / (v_min_mps + d) for e in Gc.edges(keys=True)
    }

    nx.set_edge_attributes(Gc, capacities, "capacity")
    return Gc


import os


def reroute_overloaded_roads(G, gamma, cache=True):
    if G.is_multigraph():
        hash_graph = ox.get_digraph(G)
        hash_val = f"multigraph_"
    elif nx.is_directed(G):
        hash_graph = G
        hash_val = f"digraph_"

    hash_val += nx.weisfeiler_lehman_graph_hash(
        hash_graph,
        edge_attr="load",
        node_attr="population",
        iterations=10,
        digest_size=32,
    )
    file_path = f"cache/spillover-load-files/{gamma}/{hash_val}.pkl"

    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    G = set_number_of_lanes(G)
    G = peak_loads(G, gamma)
    G = load_capacities(G)
    if cache:
        try:
            with open(file_path, "rb") as f:
                print(f"Providing spillover loads from '{file_path}'.")
                spillover_load = pickle.load(f)
        except:
            spillover_load = _reroute_overloaded_roads(G)
            with open(file_path, "wb") as f:
                pickle.dump(spillover_load, f)
    else:
        spillover_load = _reroute_overloaded_roads(G)
        print(type(spillover_load))
    nx.set_edge_attributes(G, spillover_load, "spillover_load")
    return G


def _reroute_overloaded_roads(G):
    """
    Reroute overloaded roads in the road network graph to balance the load.

    Parameters:
        Gc (networkx.MultiDiGraph): Input road network graph.
        #gamma (float): Occupancy rate of road network.

    Returns:
        spillover_load_dict (dict): Dictionary of spillover loads for each edge in the road network graph.
    """

    # G = Gc.copy()
    # G = set_number_of_lanes(Gc)
    # G = peak_loads(G, gamma)
    # G = load_capacities(G)

    peak_load_dict = nx.get_edge_attributes(G, "peak_load")
    capacities_dict = nx.get_edge_attributes(G, "capacity")

    if sum(peak_load_dict.values()) > sum(capacities_dict.values()):
        logging.warning(
            "Total loads greater than capacity. Will not converge; returning without rerouting."
        )
        return G

    nx.set_edge_attributes(G, peak_load_dict, "spillover_load")
    over_idxs = [
        e for e in G.edges(keys=True) if (peak_load_dict[e] - capacities_dict[e]) > 0
    ]

    last_len_over_idx = len(over_idxs)
    counter = 0

    while over_idxs:
        if len(over_idxs) == last_len_over_idx:
            counter += 1
        else:
            counter = 0
            last_len_over_idx = len(over_idxs)

        if counter >= 10:
            excess_load = sum(
                [
                    G[e[0]][e[1]][e[2]]["spillover_load"]
                    - G[e[0]][e[1]][e[2]]["capacity"]
                    for e in over_idxs
                ]
            )
            logging.warning(
                f"Rerouting is stuck (in a roundabout); excess loads on {len(over_idxs)} roads will be ignored. Ignored excess load is: {excess_load:.1f} on {over_idxs}."
            )
            return peak_load_dict

        for e in over_idxs:
            u, v, k = e
            capacity = G[u][v][k]["capacity"]
            diff_load = G[u][v][k]["spillover_load"] - capacity
            G[u][v][k]["spillover_load"] = capacity  # Reset to max capacity

            pred_edges = G.in_edges(u, keys=True)

            total_spillover = sum(
                [G[n][m][key]["spillover_load"] for n, m, key in pred_edges]
            )
            if total_spillover > 0:
                for n, m, key in pred_edges:
                    proportion = G[n][m][key]["spillover_load"] / total_spillover
                    G[n][m][key]["spillover_load"] += diff_load * proportion

        peak_load_dict = nx.get_edge_attributes(G, "spillover_load")
        over_idxs = [
            e
            for e in G.edges(keys=True)
            if (peak_load_dict[e] - capacities_dict[e]) > 1e-5
        ]

    return peak_load_dict


def effective_spillover_velocities(G, gamma, inplace=False, cache=True):
    """
    Compute the effective spillover velocities for each edge in the road network graph.

    Parameters:
        G (networkx.MultiDiGraph): Input road network graph.
        gamma (float): Occupancy rate of road network.
        inplace (bool): Whether to update the graph in place.
    Returns:
        networkx.MultiDiGraph: Updated road network graph with the 'spillover_velocity' and 'spillover_travel_time' attributes set for each edge.
    """

    print(f"Computing effective spillover velocities with gamma={gamma:.3f}...")
    x_veh = 5
    t_react = 2

    if not inplace:
        G = G.copy()

    G = reroute_overloaded_roads(G, gamma, cache=cache)
    # G = set_number_of_lanes(G)

    spillover_loads = nx.get_edge_attributes(G, "spillover_load")
    lanes = nx.get_edge_attributes(G, "lanes")
    lengths = nx.get_edge_attributes(G, "length")
    speed_lims = nx.get_edge_attributes(G, "speed_kph")

    lanes_mod = dict((k, max(v - 1, 1)) for k, v in lanes.items())

    eff_velos = {
        e: (
            lengths[e] * lanes_mod[e] / (spillover_loads[e] * t_react) - x_veh / t_react
        )
        * 60
        * 60
        / 1000
        for e in G.edges(keys=True)
    }
    eff_velos = dict(
        (e, speed_lims[e]) if v > speed_lims[e] else (e, np.round(v, 1))
        for e, v in eff_velos.items()
    )

    rr_travel_time = {
        e: np.round(np.divide(lengths[e] / 1000 * 60 * 60, eff_velos[e]), 1)
        for e in G.edges(keys=True)
    }

    nx.set_edge_attributes(G, rr_travel_time, "spillover_travel_time")
    nx.set_edge_attributes(G, eff_velos, "spillover_velocity")

    if not inplace:
        return G


def effective_velocities(G, gamma, inplace=False):
    x_veh = 5
    t_react = 2

    if not inplace:
        G = G.copy()

    G = peak_loads(G, gamma)

    peak_load_dict = nx.get_edge_attributes(G, "peak_load")
    lanes = nx.get_edge_attributes(G, "lanes")
    lengths = nx.get_edge_attributes(G, "length")
    speed_lims = nx.get_edge_attributes(G, "speed_kph")

    lanes_mod = dict((k, max(v - 1, 1)) for k, v in lanes.items())

    eff_velos = {
        e: (lengths[e] * lanes_mod[e] / (peak_load_dict[e] * t_react) - x_veh / t_react)
        * 60
        * 60
        / 1000
        for e in G.edges(keys=True)
    }
    eff_velos = dict(
        (
            (e, speed_lims[e])
            if v > speed_lims[e]
            else (e, 5) if v < 5 else (e, np.round(v, 1))
        )
        for e, v in eff_velos.items()
    )

    travel_time = {
        e: np.round(np.divide(lengths[e] / 1000 * 60 * 60, eff_velos[e]), 1)
        for e in G.edges(keys=True)
    }

    nx.set_edge_attributes(G, travel_time, "effective_travel_time")
    nx.set_edge_attributes(G, eff_velos, "effective_velocity")
    if not inplace:
        return G


# %%

if __name__ == "__main__":
    from src import RoadNetwork2 as rn

    highway_tags = [
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
    ]

    r = rn.RoadNetwork(
        osmpbf="ger-buffered-200km.osm.pbf",
        highway_filter=f"w/highway={','.join(highway_tags)}",
        place="Cologne,Germany",
    )
    r.loads("travel_time", normalized=False, threads=5, tasks_per_cpu=1)
    G = r.graph
    # %%
    gamma = 0.1
    G = effective_spillover_velocities(G, gamma, cache=False)

    edges = ox.graph_to_gdfs(G, nodes=False)

    # %%
