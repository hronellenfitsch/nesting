#!/usr/bin/env python

"""
    decomposer.py

    Contains functions that process leaf networks in a format
    intelligible to NetworkX.
    
    Most important functions:

    prune_graph: Removes all tree-like components from the given graph.
    shortest_cycles: Finds a basis of minimal cycles of a planar pruned graph.
        The minimal cycles correspond to the facets of the planar polygon
        corresponding to the graph.
    hierarchical_decomposition: Performs the hierarchical decomposition
        algorithm on a single connected component of a pruned graph.
        The connected component can be obtained using NetworkX,
        see the main function for example usage.

    All other functions should be treated as internal.

    2013 Henrik Ronellenfitsch
"""

from numpy import *
from numpy import ma
import numpy.random

import scipy
import scipy.sparse
import scipy.spatial

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

import random
import argparse
import os
import time
import sys

import storage
import plot

import seaborn as sns

from blist import sortedlist

from cycle_basis import *

class Filtration():
    """ Represents the filtration of a graph in a memory-efficient way
    by only storing changes between successive filtration steps.
    The filtration is constructed successively by supplying a start
    graph and then adding/removing edges and nodes.

    Certain steps in the filtration can be accessed in two ways:

    (a) by using array-index notation. This method constructs the
    filtration steps in-memory for each access. Thus for both f[14] and
    f[15], f[14] is built up from the ground!

    (b) by using the instance as an iterable. This method
    constructs the filtration successively, but only allows access
    to successive steps.

    Note that this implementation is "backwards" with respect to the
    usual mathematical definition which has
        {} = K^0 \subset K^1 \subset ... \subset K^{n-1} \subset K^n = X.
    """
    def __init__(self, base):
        self.base = base
        self.removed_nodes = [[]]
        self.removed_edges = [[]]
        self.step_nums = [0]
        self.iter_return_step_nums = False

    def add_step(self, step_num, removed_nodes, removed_edges):
        """ Adds a step to the filtration which removes the given
        edges and nodes from the graph.
        """
        self.removed_nodes.append(removed_nodes)
        self.removed_edges.append(removed_edges)
        self.step_nums.append(step_num)

    def set_iter_return_step_nums(return_step_nums):
        """ Determines whether iterating over the filtration
        also returns the actual step numbers of all steps (because
        external loops have not been removed.)

        After iteration over the whole filtration this variable is
        set to False.
        """
        self.iter_return_step_nums = return_step_nums
    
    def __len__(self):
        return len(self.removed_nodes)

    def __getitem__(self, key):
        """ Returns the accessed step in the filtration.
        f[0] returns the original graph,
        negative numbers as keys are possible
        """
        if not isinstance(key, int):
            raise TypeError()
        
        max_ind = self.__len__()
        if key >= max_ind:
            raise IndexError()
        
        if key < 0:
            key = max_ind - key - 2

        gen = self.__iter__()
        for i in xrange(key + 1):
            cur = gen.next()

        return cur

    def __iter__(self):
        """ Returns a generator that successively constructs the
        filtration.
        """
        cur = self.base.copy()

        if self.iter_return_step_nums:
            yield 0, cur
        else:
            yield cur
        
        # Perform filtration steps
        for nodes, edges, step in izip(self.removed_nodes[1:], \
                self.removed_edges[1:], self.step_nums[1:]):
            cur.remove_edges_from(edges)
            cur.remove_nodes_from(nodes)

            if self.iter_return_step_nums:
                yield step, cur
            else:
                yield cur
         
        self.iter_return_step_nums = False

def path_subgraph(G, path, edges):
    """ Returns the subgraph of G induced by the given path (ordered collection
    of nodes)
    """
    subgraph = G.subgraph(path).copy()
    edges = set(edges)
     
    to_remove = []
    for e in subgraph.edges_iter():
        if not e in edges and not e[::-1] in edges:
            to_remove.append(e)

    subgraph.remove_edges_from(to_remove)

    return subgraph

def prune_graph(G):
    """
        Return a graph describing the loopy part of G, which is
        implicitly described by the list of cycles.
        The loopy part does not contain any

        (a) tree subgraphs of G
        (b) bridges of G

        Thus pruning may disconnect the graph into several 
        connected components.
    """
    cycles = nx.cycle_basis(G)
    pruned = G.copy()
    cycle_nodes = set(chain.from_iterable(cycles))
    cycle_edges = []

    for c in cycles:
        cycle = c + [c[0]]
        a, b = tee(cycle)
        next(b, None)
        edges = izip(a, b)

        cycle_edges.append(edges)
    
    all_cycle_edges = set(tuple(sorted(e)) \
            for e in chain.from_iterable(cycle_edges))
    # remove treelike components and bridges by removing all
    # edges not belonging to loops and then all nodes not
    # belonging to loops.
    pruned.remove_edges_from(e for e in G.edges_iter() \
            if (not tuple(sorted(e)) in all_cycle_edges))
    pruned.remove_nodes_from(n for n in G if not n in cycle_nodes)

    return pruned

def connected_component_subgraphs_nocopy(G):
    """Return connected components as subgraphs. This is like
    networkx's standard routine, but does not perform a deep copy
    because of memory.
    """
    cc = nx.connected_components(G)
    graph_list = []
    for c in cc:
        graph_list.append(G.subgraph(c))
    return graph_list

def prune_dual(leaf, dual):
    """ Modifies both leaf and dual by removing all cycles not
    belonging to the largest connected component of dual.
    """

    con = connected_component_subgraphs_nocopy(dual)
    n_con = len(con)

    print "Dual connected components: {}.".format(n_con)

    if n_con == 1:
        return
    
    # These are the cycles we want to remove
    dual_nodes = list(chain.from_iterable(comp.nodes_iter() 
        for comp in con[1:]))

    nodes_to_rem = set()
    
    for n in dual_nodes:
        cy = dual.node[n]['cycle']
        # Remove edges from original graph
        leaf.remove_edges_from(cy.edges)
        
        for n in cy.path:
            nodes_to_rem.add(n)
    
    # Remove nodes from dual graph
    dual.remove_nodes_from(dual_nodes)

    # remove disconnected nodes from original graph
    nodes_to_rem = [n for n in nodes_to_rem if leaf.degree(n) == 0]
    leaf.remove_nodes_from(nodes_to_rem)

def cycle_dual(G, cycles, avg_fun=None):
    """
        Returns dual graph of cycle intersections, where each edge
        is defined as one cycle intersection of the original graph
        and each node is a cycle in the original graph.

        The general idea of this algorithm is:

        * Find all cycles which share edges by an efficient dictionary
          operation

        * Those edges which border on exactly two cycles are connected

        The result is a possibly disconnected version of the dual
        graph which can be further processed.

        The naive algorithm is O(n_cycles^2) whereas this improved 
        algorithm is better than O(n_cycles) in the average case.
    """
    if avg_fun == None:
        avg_fun = lambda c, w: average(c, weights=w)

    dual = nx.Graph()

    neighbor_cycles = find_neighbor_cycles(G, cycles)
    
    # Construct dual graph
    for ns in neighbor_cycles:
        # Add cycles
        for c, n in ((cycles[n], n) for n in ns):
            dual.add_node(n, x=c.com[0], y=c.com[1], cycle=c, \
                    external=False, cycle_area=c.area())
        
        # Connect pairs
        if len(ns) == 2:
            a, b = ns

            c_a = cycles[a]
            c_b = cycles[b]

            sect = c_a.intersection(c_b)

            wts = [G[u][v]['weight'] for u, v in sect]
            conds = [G[u][v]['conductivity'] for u, v in sect]

            wt = sum(wts)
            #cond = average(conds, weights=wts)
            #cond = min(conds)
            cond = avg_fun(conds, wts)

            dual.add_edge(a, b, weight=wt, 
                    conductivity=cond, intersection=sect)
    
    return dual

def remove_outer_from_dual(G, dual, outer, new_connections=True):
    """ Removes the outermost loop from the dual graph
    and creates new nodes for each loop bordering it.
    """
    # Only necessary if there is more than one loop
    if dual.number_of_nodes() <= 1:
        return

    # Find boundary nodes in dual
    outer_n = [n for n in dual.nodes_iter(data=True) \
        if n[1]['cycle'] == outer][0][0]
    boundary = [n for n in dual.nodes_iter() 
            if outer_n in dual.neighbors(n)]
    
    if new_connections:
        max_nodes = max(dual.nodes())
        k = 1
        for b in boundary:
            new = max_nodes + k

            # Construct outer point
            attrs = dual[outer_n][b]
            
            inter = attrs['intersection']
            # FIXME: Nicer positions.
            a = list(inter)[0][0]

            dual.add_node(new, x=G.node[a]['x'], 
                    y=G.node[a]['y'], 
                    external=True, cycle=outer, cycle_area=0.)
            dual.add_edge(b, new, **attrs)

            k = k + 1
    
    # Remove original boundary node
    dual.remove_node(outer_n)

def hierarchical_decomposition(leaf, avg_fun=None, 
        include_externals=False, remove_outer=True,
        filtration_steps=100):
    """
        Performs a variant of the algorithm 
        from Katifori, Magnasco, PLOSone 2012.
        Returns a NetworkX digraph (ordered edges) containing 
        the hierarchy tree as well as the root node in tree.graph['root'].
        Also returns a representation of the cycle dual graph
        and a list of graphs containing successive filtrations
        of the original.
        If include_externals == True, the filtration will include
        removing of external edges.
        The leaf must contain only one pruned connected component, otherwise
        the algorithm will fail and not correctly account for outer cycles
    """
    if avg_fun == None:
        avg_fun = lambda c, w: average(c, weights=w)

    # Preprocessing
    print "Detecting minimal cycles."
    cycles = shortest_cycles(leaf)

    print "Constructing dual."
    dual = cycle_dual(leaf, cycles, avg_fun=avg_fun)    

    print "Pruning dual."
    prune_dual(leaf, dual)

    print "Detecting outermost loop and rewiring."
    outer = outer_loop(leaf, cycles)

    remove_outer_from_dual(leaf, dual, outer, new_connections=remove_outer)

    dual_orig = dual.copy()

    print "Performing hierarchical decomposition."
    tree = nx.DiGraph()
    filtration = Filtration(leaf.copy())
    filtr_cur = leaf.copy()

    # Construct leaf nodes from cycles
    dual_nodes = dual.nodes()
    max_node = max(dual_nodes)
    tree.add_nodes_from(dual.nodes_iter(data=True))
    
    # Maintain a sorted collection of all intersections ordered
    # by conductivity
    sorted_edges = [tuple(sorted(e)) for e in dual.edges_iter()]
    s_edges = sortedlist(sorted_edges, key=lambda k: \
            dual[k[0]][k[1]]['conductivity'])

    # Work through all intersections
    #plt.figure()
    k = 1

    # Perform actual decomposition
    while dual.number_of_edges():
        #plt.clf()
        #plot.draw_leaf(filtr_cur)
        #plot.draw_dual(dual)
        #raw_input()

        # Find smallest intersection
        i, j = s_edges[0]
        del s_edges[0]

        dual_i, dual_j = dual.node[i], dual.node[j]
        dual_e_i, dual_e_j = dual[i], dual[j]

        intersection = dual_e_i[j]['intersection']
        
        # Save current step in filtration as subgraph (no copying!)
        if ((not dual_i['external'] and not dual_j['external']) \
                or include_externals):
            filtr_cur.remove_edges_from(intersection)
            
            if mod(k, filtration_steps) == 0 or k == max_node - 1:
                removed_nodes = [n for n, d in filtr_cur.degree_iter() \
                        if d == 0]

                filtr_cur.remove_nodes_from(removed_nodes)
                filtration.add_step(k, removed_nodes, intersection)

        # New tree node
        new = max_node + k
        tree.add_edges_from([(new, i), (new, j)])
        
        # a) Create new node in the dual with attributes of the
        #    symmetric difference of i and j
        
        # Contracted external nodes do not change the cycle of the result,
        # the resulting node keeps its cycle.
        # Since external nodes are always leaf nodes, they can only be
        # contracted with internal nodes.
        if dual_i['external']:
            new_cycle = dual_j['cycle']
        elif dual_j['external']:
            new_cycle = dual_i['cycle']
        else:
            new_cycle = \
                dual_i['cycle'].symmetric_difference(dual_j['cycle'])
        
        # Update contracted node properties
        dual.add_node(new, x=new_cycle.com[0], y=new_cycle.com[1], \
            cycle=new_cycle, cycle_area=new_cycle.area(), external=False)
        
        # Add tree attributes
        tree.add_node(new, cycle=new_cycle, cycle_area=new_cycle.area(),
                external=False, x=new_cycle.com[0], y=new_cycle.com[1])

        # b) Find all neighbors of the two nodes in the dual graph
        # we use a set in case i and j have the same neighbor
        # (triangle in the dual graph)
        neighbors_i = dual.neighbors(i)
        neighbors_j = dual.neighbors(j)

        neighbors_i.remove(j)
        neighbors_j.remove(i)

        neighbors = set(neighbors_i + neighbors_j)
        
        # connect all neighbors to the new node
        for n in neighbors:
            if n in neighbors_i and n in neighbors_j:
                # Recalculate attributes
                wts = [dual_e_i[n]['weight'], \
                    dual_e_j[n]['weight']]
                conds = [dual_e_i[n]['conductivity'], \
                    dual_e_j[n]['conductivity']]

                inter = dual_e_i[n]['intersection'].union(
                        dual_e_j[n]['intersection'])
                
                wt = sum(wts)
                cond = avg_fun(conds, wts)

                dual.add_edge(n, new, weight=wt, conductivity=cond,
                        intersection=inter)

            elif n in neighbors_i:
                dual.add_edge(n, new, **dual_e_i[n])
            elif n in neighbors_j:
                dual.add_edge(n, new, **dual_e_j[n])

            # Update sorted list
            s_edges.add((n, new))

        # Remove old nodes
        for n in neighbors_i:
            s_edges.remove(tuple(sorted([n, i])))

        for n in neighbors_j:
            s_edges.remove(tuple(sorted([n, j])))
        
        dual.remove_nodes_from([i, j])
        
        # Merge external neighbors of new node
        ext = [n for n in dual.neighbors(new) if dual.node[n]['external']]
        n_ext = len(ext)
        if n_ext > 1:
            # construct new attributes
            inter = reduce(lambda x, y:
                dual[new][x]['intersection'].union(
                dual[new][y]['intersection']), ext)
    
            wts = [dual[new][e]['weight'] for e in ext]
            conds = [dual[new][e]['conductivity'] for e in ext]

            wt = sum(wts)
            cond = avg_fun(conds, wts)
            
            # construct new external node
            dual.add_node(new + 1, x=dual.node[ext[0]]['x'],
                    y=dual.node[ext[0]]['y'],
                    cycle=dual.node[ext[0]]['cycle'], cycle_area=0.,
                    external=True)

            dual.add_edge(new, new + 1, weight=wt, conductivity=cond,
                    intersection=inter)
            
            # update tree information
            tree.add_node(new + 1, x=dual.node[ext[0]]['x'],
                    y=dual.node[ext[0]]['y'],
                    cycle=dual.node[ext[0]]['cycle'], cycle_area=0.,
                    external=True)

            k += 1
            
            # update sorted edge list
            s_edges.add((new, new + 1))
            for e in ext:
                s_edges.remove(tuple(sorted([new, e])))

            dual.remove_nodes_from(ext)
            tree.remove_nodes_from(ext)

        # Counter to index new nodes
        print "Step {}/{}\r".format(k, max_node),
        k += 1

    if k > 1:
        # The last loop is indeed external since it is the outer one
        tree.add_node(new, cycle=new_cycle, cycle_area=new_cycle.area(),
                external=True, x=new_cycle.com[0], y=new_cycle.com[1])

        tree.graph['root'] = new
    else:
        # There was only one loop.
        tree.graph['root'] = tree.nodes()[0]

    return tree, dual_orig, filtration

def apply_workaround(G):
    """ Applies a workaround to the graph which removes all
    exactly collinear edges.
    """

    removed_edges = []
    for n in G.nodes_iter():
        nei = G.neighbors(n)
        p1 = array([[G.node[m]['x'], G.node[m]['y']] \
                for m in nei])
        p0 = array([G.node[n]['x'], G.node[n]['y']])
        
        dp = p1 - p0
        dp_l = sqrt((dp*dp).sum(axis=1))
        dp_n = dp/dp_l[...,newaxis]

        coss = dot(dp_n, dp_n.T)
        
        tril_i = tril_indices(coss.shape[0])
        coss[tril_i] = 0.
        
        coll = abs(coss - 1.) < 1e-3
        for i in xrange(len(nei)):
            c = where(coll[:,i])[0]
            if len(c) > 0:
                edges = tuple((n, nei[cc]) for cc in c)
                dp_c = zip(dp_l[c], edges) + [(dp_l[i], (n, nei[i]))]
                max_v, max_e = max(dp_c)
                
                print "Found collinear edges:"
                print dp_c
                print "Removing offending edge {}.".format(max_e)
                G.remove_edge(*max_e)

                removed_edges.append(max_e)

    return removed_edges

# Code for intersection test taken from 
# http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def knbrs(G, start, k):
    """ Return the k-neighborhood of node start in G.
    """
    nbrs = set([start])
    for l in xrange(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs

def remove_intersecting_edges(G):
    """ Remove any two edges that intersect from G,
    correcting planarity errors.
    Since we cannot tell which one of the edges is the "correct" one,
    we remove both.
    """
    
    edges_to_rem = []
    edges = G.edges()
    for i in xrange(len(edges)):
        u1, v1 = edges[i]

        u1_x = G.node[u1]['x']
        u1_y = G.node[u1]['y']

        v1_x = G.node[v1]['x']
        v1_y = G.node[v1]['y']

        u1_vec = [u1_x, u1_y]
        v1_vec = [v1_x, v1_y]
        
        # look at order 5 neighbors subgraph (this is an approximation,
        # not guaranteed to work every single time! It is fast though.)
        neighs = knbrs(G, u1, 5)
        neighs.update(knbrs(G, v1, 5))
        sg = G.subgraph(neighs)

        for u2, v2 in sg.edges_iter():
            # If the edges have a node in common, disregard.
            if u2 == u1 or u2 == v1 or v2 == u1 or v2 == u2:
                continue

            u2_x = G.node[u2]['x']
            u2_y = G.node[u2]['y']

            v2_x = G.node[v2]['x']
            v2_y = G.node[v2]['y']
            
            u2_vec = [u2_x, u2_y]
            v2_vec = [v2_x, v2_y]
            
            if intersect(u1_vec, v1_vec, u2_vec, v2_vec):
                    edges_to_rem.append((u1, v1))
                    edges_to_rem.append((u2, v2))
                    #print (u1, v1), (u2, v2)
    
    G.remove_edges_from(edges_to_rem)

if __name__ == '__main__':
    sns.set(style='white', font_scale=1.5)
    params = {'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)

    plt.ion()
    parser = argparse.ArgumentParser("Leaf Decomposer.")
    parser.add_argument('INPUT', help="Input file in .gpickle format" \
    " containing the unpruned leaf data as a graph.")
    parser.add_argument('-s', '--save', help="Saves the hierarchical tree in" \
    " the given pickle file", type=str, default="")
    parser.add_argument('-p', '--plot', help="Plots the intermediate results.",\
    action='store_true')
    parser.add_argument('-a', '--average-intersection',
            help="Use average of edge conductivities instead of minimum",
            action="store_true")
    parser.add_argument('-e', '--no-external-loops',
            help='If set, do not assign virtual external loops',
            action='store_true')
    parser.add_argument('-w', '--workaround',
            help="Use workaround to remove spurious collinear edges.",
            action='store_true')
    parser.add_argument('-f', '--filtration-steps',
            help='Number of steps at which a new filtration should be stored', type=int, default=1000)
    parser.add_argument('-i', '--inverse-intersection', action='store_true',
            help='use inverse sum of edge conductivities')

    args = parser.parse_args()

    print "Loading file {}.".format(args.INPUT)
    leaf = nx.read_gpickle(args.INPUT)
    
    #leaf = test_loop()
    print "Removing intersecting edges."
    remove_intersecting_edges(leaf)

    print "Pruning."
    pruned = prune_graph(leaf)

    if args.workaround:
        print "Applying workaround to remove spurious collinear edges."
        removed_edges = apply_workaround(pruned)
        
        print "Pruning again."
        pruned = prune_graph(pruned)
    else:
        removed_edges = []
    
    con = nx.connected_component_subgraphs(pruned)
    print "Connected components:", len(con)

    if len(con) == 0:
        print "This graph is empty!!"
        print "Have a nice day."
        sys.exit(0)

    print "Decomposing largest connected component."
    
    if args.average_intersection:
        avg_fun = None
    elif args.inverse_intersection:
        avg_fun = lambda c, w: sum(1./asarray(c))
    else:
        avg_fun = lambda c, w: min(c)
    
    t0 = time.time()
    tree, dual, filtr = hierarchical_decomposition(con[0], 
            avg_fun=avg_fun, remove_outer=not args.no_external_loops,
            filtration_steps=args.filtration_steps)

    print "Decomp. took {}s.".format(time.time() - t0)
    print "Number of loops:", dual.number_of_nodes()
    print "Number of tree nodes:", tree.number_of_nodes()

    if args.save != "":
        print "Saving file."
        
        SAVE_FORMAT_VERSION = 5
        sav = {'version':SAVE_FORMAT_VERSION, \
                'leaf':leaf, 'tree':tree, 'dual':dual, \
                'filtration':filtr, 'pruned':pruned, \
                'removed-edges':removed_edges}
        
        storage.save(sav, args.save)
    
    print "Done."

    if args.plot:
        plt.figure()
        plot.draw_leaf(leaf, "Input leaf data")
        plt.figure()
        plot.draw_leaf(pruned, "Pruned leaf data and dual graph")
        plot.draw_dual(dual)
        plt.figure()
        plot.draw_tree(tree)
        plt.figure()
        plot.draw_filtration(filtr)
        plt.show()
