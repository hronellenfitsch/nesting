#!/usr/bin/env python

"""
    cycle_basis.py

    functions for calculating the cycle basis of a graph
"""

from numpy import *
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path

if matplotlib.__version__ >= '1.3.0':
    from matplotlib.path import Path
else:
    from matplotlib import nxutils

from itertools import chain
from itertools import ifilterfalse
from itertools import izip
from itertools import tee

from collections import defaultdict

import time

from helpers import *

class Cycle():
    """ Represents a set of nodes that make up a cycle in some
    graph. Is hashable and does not care about orientation or things
    like that, two cycles are equal if they share the same nodes.
    A cycle can be compared to a set or frozenset of nodes.
    path is a list of vertices describing a closed path in the cycle.
    if it is absent, a closed path will be calculated together with
    coordinates.
    coords is an array of x-y pairs representing the coordinates of
    the cycle path elements.
    """
    def __init__(self, graph, edges, coords=None):
        """ Initializes the Cycle with an edge list representing the
        cycle.
        All edges should be ordered such that a cycle is represented
        as

        (1,2)(2,3)(3,4)...(n-2,n-1)(n-1,1)

        Parameters:
            graph: The underlying graph object

            edges: The edge list making up the cycle.

            is_ordered: If set to false, will use the neighborhood
                information from graph to construct ordered edge set
                from unordered one.
                In case the unordered edge set is not a connected graph,
                e.g. when removing one cycle splits the surrounding
                one in half, the smaller connected component in terms
                of total length is thrown away. Since our cycles are
                typically convex, this means we use the outermost
                component.
        """
        self.graph = graph

        edges, self.total_area = self.ordered_edges(edges)

        self.path = zip(*edges)[0]
        if coords is None:
            self.coords = array([[graph.node[n]['x'], graph.node[n]['y']]
                    for n in self.path])
        else:
            self.coords = coords
        self.edges = edges

        # This allows comparisons
        self.edgeset = set([tuple(sorted(e)) for e in edges])
        self.com = mean(self.coords, axis=0)

        # This frozenset is used to compare/hash cycles.
        self._nodeset = frozenset(self.path)

    def ordered_edges(self, edges):
        """ Uses the graph associated to this cycle to order
        the unordered edge set.

        Also return the area of the cycle. This is defined as
        max(Areas of individual connected components) -
            (Areas of other connected components)

        This assumes that the cycle is one large cycle containing
        one or more smaller cycles.
        """
        # construct subgraph consisting of only the specified edges
        edge_graph = nx.Graph(edges)

        con = sorted_connected_components(edge_graph)

        # Calculate sorted edge list for each connected component
        # of the cycle
        component_sorted_edges = []
        areas = []
        G = self.graph
        for comp in con:
            # get ordered list of edges
            component_edges = comp.edges()
            n_edges = len(component_edges)
            sorted_edges = []
            start = component_edges[0][0]
            cur = start
            prev = None

            for i in xrange(n_edges):
                nextn = [n for n in comp.neighbors(cur)
                        if n != prev][0]
                sorted_edges.append((cur, nextn))

                prev = cur
                cur = nextn

            # coordinates of path
            coords = array([(G.node[u]['x'], G.node[u]['y'])
                    for u, v in sorted_edges] \
                            + [(G.node[sorted_edges[0][0]]['x'],
                                G.node[sorted_edges[0][0]]['y'])])

            areas.append(polygon_area(coords))
            component_sorted_edges.append(sorted_edges)

        if len(areas) > 1:
            areas = sorted(areas, reverse=True)
            total_area = areas[0] - sum(areas[1:])
        else:
            total_area = areas[0]

        return list(chain.from_iterable(
            sorted(component_sorted_edges, key=len, reverse=True))), \
                    total_area

    def intersection(self, other):
        """ Returns an edge set representing the intersection of
        the two cycles.
        """
        inters = self.edgeset.intersection(other.edgeset)

        return inters

    def union(self, other, data=True):
        """ Returns the edge set corresponding to the union of two cycles.
        Will overwrite edge/vertex attributes from other to this,
        so only use if both cycle graphs are the same graph!
        """
        union = self.edgeset.union(other.edgeset)
        return union

    def symmetric_difference(self, other, intersection=None):
        """ Returns a Cycle corresponding to the symmetric difference of
        the Cycle and other. This is defined as the set of edges which
        is present in either cycle but not in both.
        If the intersection has been pre-calculated it can be used.
        This will fail on non-adjacent loops.
        """
        new_edgeset = list(self.edgeset.symmetric_difference(
            other.edgeset))

        return Cycle(self.graph, new_edgeset)

    def area(self):
        """ Returns the area enclosed by the polygon defined by the
        Cycle. If the cycle contains more than one connected component,
        this is defined as the area of the largest area connected
        component minus the areas of the other connected components.
        """
        return self.total_area

    def radii(self):
        """ Return the radii of all edges in this cycle.
        """
        return array([self.graph[u][v]['conductivity']
            for u, v in self.edgeset])

    def __hash__(self):
        """ Implements hashing by using the internal set description's hash
        """
        return self._nodeset.__hash__()

    def __eq__(self, other):
        """ Implements comparison using the internal set description
        """
        if isinstance(other, Cycle):
            return self._nodeset.__eq__(other._nodeset)
        elif isinstance(other, frozenset) or isinstance(other, set):
            return self._nodeset.__eq__(other)
        else:
            return -1

    def __repr__(self):
        return repr(self._nodeset)

def polygon_area(coords):
    """ Return the area of a closed polygon
    """
    Xs = coords[:,0]
    Ys = coords[:,1]

    # Ignore orientation
    return 0.5*abs(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))

def traverse_graph(G, start, nextn):
    """ Traverses the pruned (i.e. ONLY LOOPS) graph G counter-clockwise
    in the direction of nextn until start is hit again.
    If G has treelike components this will fail and get stuck, there
    is no backtracking.

    Returns a list of nodes visited, a list of edges visited and
    an array of node coordinates.
    This will find (a) all internal
    smallest loops (faces of the planar graph) and (b) one maximal
    outer loop
    """
    start_coords = array([G.node[start]['x'], G.node[start]['y']])
    nodes_visited = [start]
    nodes_visited_set = set()
    edges_visited = []
    coords = [start_coords]

    prev = start
    cur = nextn

    while cur != start:
        cur_coords = array([G.node[cur]['x'], G.node[cur]['y']])
        # We ignore all neighbors we alreay visited to avoid multiple loops

        neighs = [n for n in G.neighbors(cur) if n != prev and n != cur]

        edges_visited.append((prev, cur))
        nodes_visited.append(cur)
        coords.append(cur_coords)

        n_neighs = len(neighs)
        if n_neighs > 1:
            # Choose path that keeps the loop closest on the left hand side
            prev_coords = array([G.node[prev]['x'], G.node[prev]['y']])
            neigh_coords = array([[G.node[n]['x'], G.node[n]['y']] \
                for n in neighs])

            ## Construct vectors and normalize
            u = cur_coords - prev_coords
            vs = neigh_coords - cur_coords

            # calculate cos and sin between direction vector and neighbors
            u /= sqrt((u*u).sum(-1))
            vs /= sqrt((vs*vs).sum(-1))[...,newaxis]

            coss = dot(u, vs.T)
            sins = cross(u, vs)

            # this is a function between -2 and +2, where the
            # leftmost path corresponds to -2, rightmost to +2
            # sgn(alpha)(cos(alpha) - 1)
            ranked = sign(sins)*(coss - 1.)

            prev = cur
            cur = neighs[argmin(ranked)]
        else:
            # No choice to make
            prev = cur
            cur = neighs[0]

        # Remove pathological protruding loops
        if prev in nodes_visited_set:
            n_ind = nodes_visited.index(prev)

            del nodes_visited[n_ind+1:]
            del coords[n_ind+1:]
            del edges_visited[n_ind:]

        nodes_visited_set.add(prev)

    edges_visited.append((nodes_visited[-1], nodes_visited[0]))

    return nodes_visited, edges_visited, array(coords)

def cycle_mtp_path(cycle):
    """ Returns a matplotlib Path object describing the cycle.
    """
    # Set up polygon
    verts = zeros((cycle.coords.shape[0] + 1, cycle.coords.shape[1]))
    verts[:-1,:] = cycle.coords
    verts[-1,:] = cycle.coords[0,:]

    codes = Path.LINETO*ones(verts.shape[0])
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    return Path(verts, codes)

def outer_loop(G, cycles):
    """ Detects the boundary loop in the set of fundamental cycles
    by noting that the boundary is precisely the one loop with
    maximum area (since it contains all other loops, they all must
    have smaller area)
    """
    return max([(c.area(), c) for c in cycles])[1]

def shortest_cycles(G):
    """ Returns a list of lists of Cycle objects belonging to the
    fundamental cycles of the pruned (i.e. there are no treelike
    components) graph G by traversing the graph counter-clockwise
    for each node until the starting node has been found.
    Also returns the outer loop.
    """
    cycleset = set()
    # Betti number counts interior loops, this algorithm finds
    # exterior loop as well!
    n_cycles = G.number_of_edges() - G.number_of_nodes() + 1

    # Count outer loop as well
    if n_cycles >= 2:
        n_cycles += 1

    print "Number of cycles including boundary: {}.".format(n_cycles)

    t0 = time.time()

    mst = nx.minimum_spanning_tree(G, weight=None)

    for u, v in G.edges_iter():
        if not mst.has_edge(u, v):
            # traverse cycle in both directions
            path, edges, coords = traverse_graph(G, u, v)
            cycleset.add(Cycle(G, edges, coords=coords))

            path, edges, coords = traverse_graph(G, v, u)
            cycleset.add(Cycle(G, edges, coords=coords))

    if len(cycleset) != n_cycles:
        print "WARNING: Found only", len(cycleset), "cycles!!"

    t1 = time.time()
    print "Detected fundamental cycles in {}s".format(t1 - t0)
    #print "Number of detected facets:", len(cycleset)
    return list(cycleset)

def find_neighbor_cycles(G, cycles):
    """ Returns a set of tuples of cycle indices describing
    which cycles share edges
    """
    n_c = len(cycles)

    # Construct edge dictionary
    edges = defaultdict(list)

    for i in xrange(n_c):
        for e in cycles[i].edges:
            edges[tuple(sorted(e))].append(i)

    # Find all neighboring cycles
    neighbor_cycles = set()

    for n in edges.values():
        neighbor_cycles.add(tuple(sorted(n)))

    return neighbor_cycles
