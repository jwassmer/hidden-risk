# %%
# ------------------ PACKAGES ---------------------------#
import os
import time
from multiprocess import Pool
import multiprocess
import numpy as np
import networkx as nx
import pickle
import osmnx as ox


from src import GermanMobiltyPanel as gmp
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function

from src import GermanMobiltyPanel as gmp

# from src import RoadNetwork as rn

# ------------------ FUNCTIONS ---------------------------#


def _single_source_dijkstra_path_basic(G, s, weight, cutoff=None):
    """
    Compute the shortest paths and related information from a single source using Dijkstra's algorithm.

    Parameters
    ----------
        G (NetworkX graph): The graph in which to find the shortest paths.
        s (node): The source node from which to start the computation.
        weight (str or callable): The edge weight attribute or function used for the shortest path calculations.
        cutoff (float or None, optional): If specified, the algorithm will terminate if the shortest path
            to a node exceeds this cutoff value. Defaults to None.

    Returns
    -------
        tuple: A tuple containing the following elements:
            - S (list): List of nodes in the order they were visited during the search.
            - P (dict): Dictionary of lists of predecessors for each node.
            - sigma (dict): Dictionary of node counts representing the number of shortest paths.
            - D (dict): Dictionary of shortest path distances from the source node.
    """
    weight = _weight_function(G, weight)
    S = []
    P = {}
    for v in G:
        P[v] = []

    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        if cutoff is not None:
            if dist > cutoff:
                continue
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma, D


def _accumulate_edges(SIBC, seen, pred, sigma, dists, populations, travel_function):
    """
    Accumulate the edge weights based on the populations, shortest path distances, and travel function.

    Parameters
    ----------
        SIBC (dict): Dictionary to store the spatial interaction edge betweennness centrality (SIBC).
        seen (list): List of nodes in the order they were visited.
        pred (dict): Dictionary of lists of predecessors for each node.
        sigma (dict): Dictionary of node counts representing the number of shortest paths.
        dists (dict): Dictionary of shortest path distances from the source node.
        populations (dict): Dictionary of node populations.
        travel_function (callable): Function to compute the travel weight for an edge.

    Returns
    -------
        dict: The updated SIBC dictionary with the accumulated edge weights.
    """
    delta = dict.fromkeys(seen, 0)
    source_node = seen[0]
    denominator = sum([populations[k] * travel_function(v) for k, v in dists.items()])

    if denominator != 0:
        while seen:
            w = seen.pop()
            flow = (
                populations[source_node]
                * populations[w]
                * travel_function(dists[w])
                / denominator
            )
            coeff = (flow + delta[w]) / sigma[w]
            for v in pred[w]:
                c = sigma[v] * coeff
                if (v, w) not in SIBC:
                    SIBC[(w, v)] += c
                else:
                    SIBC[(v, w)] += c
                delta[v] += c
    return SIBC


def _rescale_e(betweenness, population, normalized):
    """
    Rescale the edge SIBC values based on population and normalization options.

    Parameters
    ----------
        betweenness (dict): Dictionary of betweenness values.
        population (dict): Dictionary of node populations.
        normalized (bool): Flag indicating whether to normalize the betweenness values.

    Returns
    -------
        dict: The rescaled betweenness centrality values.
    """
    if normalized:
        total_pop = sum(population.values())
        total_betweenness = sum(betweenness.values())
        betweenness = {
            k: total_pop * v / total_betweenness for k, v in betweenness.items()
        }

    return betweenness


def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc


# %%
def interaction_betweenness_centrality(
    graph,
    weight="travel_time",
    normalized=True,
    cutoff="default",
    cache=True,
    return_graph=True,
    **kwargs,
):
    """
    Computes the spatial interaction betweenness centrality for each edge in the graph.

    Parameters
    ----------
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths. Either 'length or' 'travel_time.
                    'Default is 'travel_time'.
        normalized (bool): Flag indicating whether to normalize the betweenness centrality values. Default is True.
        cutoff (float or str): Cutoff value for the maximum shortest path length. Default is 'default'.
        cache (bool): Flag indicating whether to cache and reuse previously computed results. Default is True.
        return_graph (bool): Flag indicating whether to return the graph with updated edge attributes. Default is True.
        **kwargs: Additional keyword arguments for parallel computation.

    Returns
    -------
        networkx.Graph or dict: The graph with updated edge attributes if return_graph=True,
                                otherwise, a dictionary of edge betweenness centrality values.

    """

    if cutoff == "default":
        if "length" in weight:
            cutoff = 60 * 1000  # 60 km
        elif "travel_time" in weight:
            cutoff = 60 * 60  # 60 mins

    if cache:
        hash = _generate_graph_hash(
            graph, weight, cutoff, normalized
        )  # Refactor hash generation to a function
        path = f"cache/load-files/{hash}.pkl"
        try:
            with open(path, "rb") as f:
                print(
                    f"The SIBC has already been computed. Providing values from '{path}'."
                )
                load_dict = pickle.load(f)
        except:
            load_dict = _compute_betweenness_centrality(
                graph, weight, normalized, cutoff, **kwargs
            )
            with open(path, "wb") as f:
                pickle.dump(load_dict, f)
    else:
        load_dict = _compute_betweenness_centrality(
            graph, weight, normalized, cutoff, **kwargs
        )

    if return_graph:
        nx.set_edge_attributes(graph, load_dict, "load")
        return graph
    else:
        return load_dict


def _generate_graph_hash(graph, weight, cutoff, normalized):
    if graph.is_multigraph():
        hash_graph = ox.get_digraph(graph)
        hash = f"multigraph_{weight}_{cutoff}_{normalized}_"
    else:
        hash_graph = graph
        hash = f"graph_{weight}_{cutoff}_{normalized}_"

    hash += nx.weisfeiler_lehman_graph_hash(
        hash_graph,
        edge_attr=weight,
        node_attr="population",
        iterations=10,
        digest_size=32,
    )
    return hash


def _compute_betweenness_centrality(graph, weight, normalized, cutoff, **kwargs):
    print("Computing SIBC...")
    if "cpu_cores" in kwargs or "threads" in kwargs:
        num_threads = max(kwargs.get("cpu_cores", 1), kwargs.get("threads", 1))
        jobs_per_cpu = kwargs.get("jobs_per_cpu", 5)
        return _parallel_interaction_betweenness_centrality(
            graph,
            weight=weight,
            cpu_cores=num_threads,
            jobs_per_cpu=jobs_per_cpu,
            normalized=normalized,
            cutoff=cutoff,
        )
    else:
        return _interaction_betweenness_centrality(
            graph, weight=weight, normalized=normalized, cutoff=cutoff
        )


def _interaction_betweenness_centrality(
    graph, weight="travel_time", normalized=False, cutoff=None
):
    """
    Computes the spatial interaction betweenness centrality for each edge in the graph.

    Parameters:
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths. Default is 'travel_time'.
        normalized (bool): Flag indicating whether to normalize the betweenness centrality values. Default is False.
        cutoff (float or None): Cutoff value for the maximum shortest path length. Default is None.

    Returns:
        dict: A dictionary of edge betweenness centrality values.
    """
    max_bin, popt_exp, popt_lin = gmp.mobility_fit_params(
        "data/GMP/mobility/", weight, bincount=250
    )
    mobility_fit = (
        lambda x: gmp.exp_func(x, *popt_exp)
        if x > max_bin
        else gmp.lin_func(x, *popt_lin)
    )
    populations = nx.get_node_attributes(graph, "population")

    start = time.time()

    SIBC = dict(zip(graph.edges(), np.zeros(len(graph.edges()))))
    for o in graph.nodes:
        seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
            graph, o, weight, cutoff=cutoff
        )
        SIBC = _accumulate_edges(
            SIBC, seen, pred, sigma, dists, populations, mobility_fit
        )

    SIBC = _rescale_e(SIBC, populations, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)

    end = time.time()
    print("Time:", round(end - start, 1), "seconds")

    return SIBC


def _parallel_interaction_betweenness_centrality(
    graph,
    cpu_cores=5,
    jobs_per_cpu=5,
    weight="travel_time",
    normalized=False,
    cutoff=None,
):
    """
    Computes the spatial interaction betweenness centrality for each edge in the graph in parallel using multiple CPU cores.

    Parameters:
        graph (networkx.Graph): The input graph.
        cpu_cores (int): The number of CPU cores to use for parallel computation. Default is 5.
        jobs_per_cpu (int): The number of jobs per CPU core. Default is 5.
        weight (str): The edge weight attribute to use for computing shortest paths. Default is 'travel_time'.
        normalized (bool): Flag indicating whether to normalize the betweenness centrality values. Default is False.
        cutoff (float or None): Cutoff value for the maximum shortest path length. Default is None.

    Returns:
        dict: A dictionary of edge betweenness centrality values.
    """
    global SIBC_array_of_nodelist

    max_bin, popt_exp, popt_lin = gmp.mobility_fit_params(weight=weight, bincount=250)
    mobility_fit = (
        lambda x: gmp.exp_func(x, *popt_exp)
        if x > max_bin
        else gmp.lin_func(x, *popt_lin)
    )
    populations = nx.get_node_attributes(graph, "population")

    def SIBC_array_of_nodelist(origin_list):
        SIBC = dict(zip(graph.edges(), np.zeros(len(graph.edges()))))
        for o in origin_list:
            seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
                graph, o, weight, cutoff=cutoff
            )
            SIBC = _accumulate_edges(
                SIBC, seen, pred, sigma, dists, populations, mobility_fit
            )

        SIBC_arr = np.array(list(SIBC.values()))
        return SIBC_arr

    node_arr = np.array(list(graph.nodes()))

    m = int(np.ceil(len(node_arr) / (cpu_cores * jobs_per_cpu)))
    node_list = [node_arr[i : i + m] for i in range(0, len(node_arr), m)]

    start = time.time()
    with Pool(processes=cpu_cores) as pool:
        future = pool.map_async(SIBC_array_of_nodelist, node_list)
        load_arr = np.sum(future.get(), axis=0)

    SIBC = dict(zip(dict.fromkeys(graph.edges()).keys(), load_arr))

    SIBC = _rescale_e(SIBC, populations, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)

    # if normalized != False:
    #    sum_SIBC = sum(list(SIBC.values()))
    #    SIBC = {k: normalized * v / sum_SIBC for k, v in SIBC.items()}
    end = time.time()
    print("Time:", round(end - start, 1), "seconds")

    return SIBC


# %%
# from src import RoadNetwork as rn

# %%
# from src import RoadNetwork as rn

"""driving_tags = [
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

city = rn.RoadNetwork(
    osmpbf="germany.osm.pbf",
    highway_filter=f"w/highway={','.join(driving_tags)}",
    place="Heidelberg,Germany",
)

graph = city.graph"""

"""# %%
SIBC = interaction_betweenness_centrality(
    graph, weight="travel_time", return_graph=False, cache=False
)
# %%
SIBC"""
# %%
