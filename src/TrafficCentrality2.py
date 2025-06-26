# %%
# ------------------ PACKAGES ---------------------------#
import time
from multiprocess import Pool
import numpy as np
import networkx as nx
import pickle
import osmnx as ox
import gc
from scipy.stats import weibull_min
import scipy.stats as stats

from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function

from src import GermanMobiltyPanel as gmp


# ------------------ FUNCTIONS ---------------------------#
def interaction_betweenness_centrality(
    graph,
    weight="travel_time",
    normalized=False,
    cutoff="default",
    cache=True,
    return_graph=True,
    fit="lognorm",
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
            graph, weight, cutoff, normalized, fit
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
                graph, weight, normalized, cutoff, fit=fit, **kwargs
            )
            with open(path, "wb") as f:
                pickle.dump(load_dict, f)
    else:
        load_dict = _compute_betweenness_centrality(
            graph, weight, normalized, cutoff, fit=fit, **kwargs
        )

    if return_graph:
        nx.set_edge_attributes(graph, load_dict, "load")
        return graph
    else:
        return load_dict


def _generate_graph_hash(graph, weight, cutoff, normalized, fit):
    if graph.is_multigraph():
        hash_graph = ox.get_digraph(graph)
        hash = f"multigraph_{weight}_{cutoff}_{normalized}_{fit}_"
    elif nx.is_directed(graph):
        hash_graph = graph
        hash = f"digraph_{weight}_{cutoff}_{normalized}_{fit}_"
    else:
        hash_graph = graph
        hash = f"graph_{weight}_{cutoff}_{normalized}_{fit}_"
    if fit == "exp_lin":
        hash = hash[: -(len(fit) + 1)]
        # bc i used  a different hash for exp-lin, before. can be removed later

    hash += nx.weisfeiler_lehman_graph_hash(
        hash_graph,
        edge_attr=weight,
        node_attr="population",
        iterations=10,
        digest_size=32,
    )
    return hash


def _compute_betweenness_centrality(
    graph, weight, normalized, cutoff, fit="weibull", **kwargs
):
    print("Computing SIBC...")
    if "num_cpus" in kwargs or "threads" in kwargs or "cpu_cores" in kwargs:
        num_threads = max(
            kwargs.get("cpu_cores", 1),
            kwargs.get("threads", 1),
            kwargs.get("num_cpus", 1),
        )
        tasks_per_cpu = max(
            kwargs.get("tasks_per_cpu", 2), kwargs.get("jobs_per_cpu", 2)
        )
        return _interaction_betweenness_centrality_multiprocessing(
            graph,
            weight=weight,
            num_cpus=num_threads,
            tasks_per_cpu=tasks_per_cpu,
            normalized=normalized,
            cutoff=cutoff,
            fit=fit,
        )
    else:
        return _interaction_betweenness_centrality(
            graph, weight=weight, normalized=normalized, cutoff=cutoff, fit=fit
        )


def mobility_fit(weight, fit="weibull"):
    if fit == "weibull":
        params = gmp.weibull_fit(weight=weight)
        mobility_fit = lambda x: gmp.weibull_pdf(x, *params)
        return mobility_fit
    elif fit == "lognorm":
        params = gmp.lognorm_fit(weight=weight)
        mobility_fit = lambda x: gmp.lognorm_pdf(x, *params)
        return mobility_fit
    elif fit == "exp_lin":
        max_bin, popt_exp, popt_lin = gmp.mobility_fit_params(
            "data/GMP/mobility/", weight, bincount=250
        )
        mobility_fit = lambda x: (
            gmp.exp_func(x, *popt_exp) if x > max_bin else gmp.lin_func(x, *popt_lin)
        )
        return mobility_fit


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


def _interaction_betweenness_centrality(
    graph, weight="travel_time", normalized=False, cutoff=None, fit="weibull"
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
    dict
        A dictionary where keys are edges in the graph and values are dictionaries containing the accumulated
        spatial interaction betweenness centrality values for the respective edges.
    """
    populations = nx.get_node_attributes(graph, "population")

    start = time.time()

    SIBC = {e: 0 for e in graph.edges()}

    for o in graph.nodes:
        seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
            graph, o, weight, cutoff=cutoff
        )

        SIBC = _accumulate_edges(
            SIBC,
            seen,
            pred,
            sigma,
            dists,
            populations,
            mobility_fit(weight, fit=fit),
        )

    SIBC = _rescale_e(SIBC, populations, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)
    end = time.time()

    print("Single-threaded Time:", round(end - start, 1), "seconds")
    return SIBC


def compute_sibc_for_batch(args):
    """
    Computes the spatial interaction betweenness centrality (SIBC) for a batch of nodes in the graph.

    This function performs a single-source shortest path computation for each node in the nodes list
    and accumulates the edge betweenness values based on the specified travel function and population data.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:

        graph (networkx.Graph): The input graph on which the SIBC is to be computed.
        nodes (list): A list of nodes for which the SIBC needs to be computed.
        weight (str): The edge weight attribute to use for computing the shortest paths. This could be attributes
                      such as 'length' or 'travel_time'.
        cutoff (float): The cutoff value for the maximum shortest path length. This helps in limiting the
                        computation to a certain distance.
        populations (dict): A dictionary with node identifiers as keys and respective populations as values.
                            This data is used in the computation of the SIBC.
        travel_function (function): A function to compute the travel time or cost between nodes based on the
                                   specified weight attribute.

    Returns
    -------
    dict
        A dictionary where keys are edges in the graph and values are dictionaries containing the accumulated
        spatial interaction betweenness centrality values for the respective edges.
    """
    graph, nodes, weight, cutoff, populations, travel_function = args
    SIBC = {e: 0 for e in graph.edges()}

    for node in nodes:
        seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
            graph, node, weight, cutoff=cutoff
        )
        SIBC = _accumulate_edges(
            SIBC,
            seen,
            pred,
            sigma,
            dists,
            populations,
            travel_function,
        )

    return np.array(list(SIBC.values()))


def _interaction_betweenness_centrality_multiprocessing(
    graph,
    weight="travel_time",
    normalized=False,
    cutoff=None,
    tasks_per_cpu=10,
    num_cpus=5,
    fit="weibull",
):
    """
    Computes the interaction betweenness centrality for all nodes in the graph using multiprocessing.

    This function divides the nodes into batches and distributes the computation of spatial interaction
    betweenness centrality across multiple CPUs to improve performance. It uses a mobility fit function
    determined by the specified weight attribute to guide the computation.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph on which the spatial interaction betweenness centrality is to be computed.

    weight : str, optional
        The edge weight attribute to use for computing the shortest paths. This could be attributes such as
        'length' or 'travel_time'. Default is 'travel_time'.

    normalized : bool, optional
        Flag indicating whether to normalize the betweenness centrality values. Currently not used in the
        function implementation. Default is False.

    cutoff : float, optional
        The cutoff value for the maximum shortest path length. This helps in limiting the computation to
        a certain distance. Default is None.

    tasks_per_cpu : int, optional
        The number of tasks assigned to each CPU during the parallel computation. Default is 10.

    num_cpus : int, optional
        The number of CPUs to use for parallel computation. Default is 5.

    Returns
    -------
    dict
        A dictionary merging the spatial interaction betweenness centrality values computed from each batch
        of nodes. The keys are edges in the graph and the values are dictionaries containing the accumulated
        spatial interaction betweenness centrality values for the respective edges.

    Notes
    -----
    - The function computes mobility fit parameters using a dataset located in "data/GMP/mobility/" and the
      specified weight attribute.
    - The function prints the computation time in seconds.
    """

    populations = nx.get_node_attributes(graph, "population")

    start = time.time()

    total_nodes = len(graph.nodes)
    # Calculate the batch size based on the number of CPUs and tasks per CPU
    batch_size = max(1, total_nodes // (num_cpus * tasks_per_cpu))

    nodes = list(graph.nodes)
    batches = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]

    args_list = [
        (graph, batch, weight, cutoff, populations, mobility_fit(weight, fit=fit))
        for batch in batches
    ]

    with Pool(processes=num_cpus) as pool:
        results = pool.map_async(
            compute_sibc_for_batch, args_list, chunksize=tasks_per_cpu
        )
        load_arr = np.sum(results.get(), axis=0)

    del results
    del args_list
    gc.collect()

    SIBC = dict(zip(dict.fromkeys(graph.edges()).keys(), load_arr))
    SIBC = _rescale_e(SIBC, populations, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)

    end = time.time()
    print("Multi-threaded Time:", round(end - start, 1), "seconds")

    return SIBC


"""
# %%
from src import RoadNetwork as rn

# Create a graph and set attributes
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

city = rn.RoadNetwork(
    osmpbf="germany.osm.pbf",
    highway_filter=f"w/highway={','.join(driving_tags)}",
    place="Bonn,Germany",
)
graph = city.graph

# %%
city.loads("travel_time", threads=5, jobs_per_cpu=5, cache=False)
sibc = city.edges.load
# %%

sibc_multi = interaction_betweenness_centrality(
    graph,
    weight="travel_time",
    normalized=True,
    num_cpus=5,
    tasks_per_cpu=5,
    cache=False,
    return_graph=False,
)


# %%

# compare sibc and sibc_multi
sum({e: sibc[e] - sibc_multi[e] for e in sibc_multi.keys()}.values())"""

# %%
