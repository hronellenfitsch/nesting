#!/usr/bin/env python
"""
    analyzer.py

    Contains functions which analyze tree graphs such as the ones
    obtained from decomposition.py.

    2013 Henrik Ronellenfitsch
"""

from numpy import *
from numpy import ma

import scipy.stats

import networkx as nx
import matplotlib.pyplot as plt

import argparse
import sys
import os
from itertools import izip
from collections import defaultdict

from decomposer import Cycle, Filtration
import decomposer
import plot
import storage

def edge_prune_graph(G):
    """ Take the graph G and edge prune, i.e. remove all
    nodes with degree 2, create new edges with appropriately
    averaged/summed weights.
    """

    def pruning_step(G):
        n_removed = 0
        for n in G.nodes():
            ns = G.neighbors(n)
            if len(ns) == 2:
                # calculate averaged attributes
                n1, n2 = ns
                l1 = G[n][n1]['weight']
                l2 = G[n][n2]['weight']

                c1 = G[n][n1]['conductivity']
                c2 = G[n][n2]['conductivity']

                l = l1 + l2
                c = (l1*c1 + l2*c2)/l
                
                # add new edge, remove old one
                G.add_edge(n1, n2, weight=l, conductivity=c)
                G.remove_node(n)

                n_removed += 1
        return n_removed
    
    print "Edge pruning..."
    while pruning_step(G) > 0:
        pass

    return G

def weighted_line_graph(G, average=False):
    """ Return a line graph of G where edge attributes are propagated
    properly. Node attributes are ignored.
    If average is set to True, perform an averaging over
    conductivities.
    """
    line_graph = nx.line_graph(G)
    line_graph.add_nodes_from((tuple(sorted((u, v))), d)
            for u, v, d in G.edges_iter(data=True))

    # average
    if average:
        new_node_conds = {}
        for n, d in line_graph.nodes_iter(data=True):
            neighbor_conds = mean([line_graph.node[m]['conductivity'] 
                    for m in line_graph.neighbors(n)])
            new_node_conds[n] = 0.5*(d['conductivity'] + 
                    neighbor_conds)

        for n, v in new_node_conds.iteritems():
            line_graph.node[n]['conductivity'] = v

    return line_graph

def topological_length(line_graph, e, G, mode='lt'):
    """ Find the topological length associated to node e
    in the line graph. Topological length is defined as
    in the comment to topological_length_statistics.
    """
    length = 0
    length_real = 0

    current_width = line_graph.node[e]['conductivity']
    current_node = e
    edges =  [e]
    
    if mode == 'lt':
        comp = lambda x, y: x < y
    elif mode == 'leq':
        comp = lambda x, y: x <= y

    while True:
        # find neighboring edges
        neighs_below = [(line_graph.node[n]['conductivity'], n)
               for n in line_graph.neighbors(current_node)
               if comp(line_graph.node[n]['conductivity'], current_width)
               and not n in edges]
        
        # edges in 2-neighborhood
        #neighs_below_2 = [(line_graph.node[n]['conductivity'], n)
        #       for n in decomposer.knbrs(line_graph, current_node, 2) 
        #       if line_graph.node[n]['conductivity'] < current_width]
                    
        length += 1
        length_real += G[current_node[0]][current_node[1]]['weight']
        
        # we're at the end
        if len(neighs_below) == 0:
            break

        # use best bet from both 2 and 1 neighborhood            
        max_neighs = max(neighs_below)

        current_width, current_node = max_neighs
        edges.append(current_node)
    
    # plot edges
    #print edges
    #plt.sca(self.leaf_subplot)
    #plot.draw_leaf_raw(G, edge_list=edges, color='r')
    #raw_input()

    return length, length_real, edges

def asymmetry(marked_tree):
    """ Returns the tree asymmetry after Van Pelt using
    the marked tree.
    """
    parts = [marked_tree.node[n]['partition-asymmetry']
            for n, d in marked_tree.degree_iter() if d > 1]

    weights = [marked_tree.node[n]['subtree-degree'] - 1
            for n, d in marked_tree.degree_iter() if d > 1]

    if len(weights) > 0:
        return average(parts, weights=weights)

def cum_size_distribution(areas, a):
    """ Return P[A > a] for the given set of areas.
    Extract areas from the hierarchical tree using

    areas = [tree.node[n]['cycle_area'] for n in tree.nodes_iter() \
            if not tree.node[n]['external']].
    """
    return sum([1. for A in areas if A > a])/len(areas)

def mark_subtrees(tree):
    """ Modifies the given tree by adding node attributes which
    contain the subtree degree (i.e. number of leaf nodes) for
    the subtree anchored at that particular node as well as the
    partition asymmetry.

    We define
        partition-asymmetry:        |r-l|/max(r, l)
        partition-asymmetry-1:      |r-l|/(r+l-1)
        partition-asymmetry-2:      |r-l|/(r+l-2)

    where partition-asymmetry-1,2 are only defined where the denominator
    does not vanish. The respective unweighted asymmetries are
    defined accordingly
    """
    for n in nx.dfs_postorder_nodes(tree):
        succ = tree.successors(n)

        if len(succ) == 0:
            tree.node[n]['subtree-degree'] = 1
            tree.node[n]['partition-asymmetry'] = None
            tree.node[n]['partition-asymmetry-1'] = None
            tree.node[n]['partition-asymmetry-2'] = None
            tree.node[n]['sub-partition-asymmetries'] = []
            tree.node[n]['asymmetry-simple-weights'] = []
            tree.node[n]['asymmetry-simple'] = 0.
            tree.node[n]['asymmetry-unweighted'] = 0.

            tree.node[n]['level'] = 0
        else:
            s0 = tree.node[succ[0]]
            s1 = tree.node[succ[1]]

            r = s0['subtree-degree']
            s = s1['subtree-degree']

            r_parts = s0['sub-partition-asymmetries']
            s_parts = s1['sub-partition-asymmetries']

            r_wts = s0['asymmetry-simple-weights']
            s_wts = s1['asymmetry-simple-weights']
            
            abs_degree_diff = abs(float(r) - s)
            degree = r + s

            my_part = abs_degree_diff/max(r, s)
            my_part_1 = abs_degree_diff/(degree - 1)

            if r + s > 2:
                my_part_2 = abs_degree_diff/(degree - 2)
            else:
                my_part_2 = None
            
            asym_simple_wts = r_wts + s_wts + [degree - 1]
            sub_part_asym = r_parts + s_parts + [my_part]
            
            tree.node[n]['subtree-degree'] = degree
            tree.node[n]['partition-asymmetry'] = my_part
            tree.node[n]['partition-asymmetry-1'] = my_part_1
            tree.node[n]['partition-asymmetry-2'] = my_part_2
            tree.node[n]['asymmetry-simple-weights'] = asym_simple_wts
            tree.node[n]['sub-partition-asymmetries'] = sub_part_asym

            tree.node[n]['asymmetry-simple'] = ma.average(
                    sub_part_asym, weights=asym_simple_wts)
            tree.node[n]['asymmetry-unweighted'] = ma.average(sub_part_asym)
             
            tree.node[n]['level'] = max(s0['level'], s1['level']) + 1

def remove_external_nodes(tree):
    """ Returns a tree that is equivalent to the given tree,
    but all external nodes are removed.
    """
    no_ext_tree = tree.copy()
    
    # Remove external nodes except for root which must be kept
    root = no_ext_tree.graph['root']
    no_ext_tree.remove_nodes_from(n for n in tree.nodes_iter()
            if tree.node[n]['external'] and n != root)
    
    internal_nodes = [n for n in no_ext_tree.nodes() if 
            len(no_ext_tree.successors(n)) == 1 \
            and len(no_ext_tree.predecessors(n)) == 1]
    
    for i in internal_nodes:
        pr = no_ext_tree.predecessors(i)[0]
        su = no_ext_tree.successors(i)[0]

        no_ext_tree.add_edge(pr, su)
        no_ext_tree.remove_node(i)

    # Handle case of root node
    root_succ = no_ext_tree.successors(root)

    if len(root_succ) == 1:
        s = root_succ[0]
        su = no_ext_tree.successors(s)

        no_ext_tree.add_edge(root, su[0])
        no_ext_tree.add_edge(root, su[1])
        no_ext_tree.remove_node(s)

    return no_ext_tree

def average_asymmetry(marked_tree, delta, Delta, attr='asymmetry-simple'):
    """ Returns the average asymmetry of all subtrees of tree whose
        degree is within Delta/2 from delta
    """
    asymmetries = array([marked_tree.node[n][attr] 
            for n in marked_tree.nodes_iter()
            if abs(marked_tree.node[n]['subtree-degree'] - delta) <=
            Delta/2.])
    
    if len(asymmetries) > 0:
        return mean(asymmetries)
    else:
        return float('NaN')

@plot.save_plot(name="average_asymmetry")
def avg_asymmetries_plot(marked_tree, Delta, 
        attr='asymmetry-simple', mode="default"):
    """ Makes a plot of several average asymmetries in a certain range
    with fixed delta
    """
    degree = marked_tree.node[marked_tree.graph['root']]['subtree-degree']
    degrees = array(sorted(list(set([marked_tree.node[n]['subtree-degree'] 
        for n in marked_tree.nodes_iter()]))))
    asyms = [average_asymmetry(marked_tree, d, Delta, attr=attr) 
            for d in degrees]

    plt.figure()
    plt.title("Average asymmetry, $\Delta = {}$, mode: {}".format(
        Delta, mode))
    plt.xlabel("Normalized subtree degree $\log(d)$")
    plt.ylabel("$\\bar Q(d)$")
    plt.plot(log(degrees/float(degree)), asyms, linewidth=2)

@plot.save_plot(name="cumulative_size_dist")
def cum_size_plot(areas):
    """ Makes a plot of the cumulative size distribution.
    areas will be normalized
    """
    plt.figure()
    plt.hist(areas/max(areas), normed=True, cumulative=-1, bins=100)
    plt.xlabel("Normalized area $a$")
    plt.ylabel("$P[A > a]$")
    plt.xlim(0)
    plt.title("Cumulative size distribution")

def areas_cum_Ps(tree, bins):
    """ cumulative size distribution
    """
    areas = [tree.node[n]['cycle_area'] for n in tree.nodes_iter() \
            if not tree.node[n]['external']]
    Ps = [cum_size_distribution(areas, a) \
            for a in linspace(min(areas), max(areas), num=bins)]

    return areas, Ps

def normalized_area_distribution(tree, bins):
    """ Returnes the set of normalized non-external areas associated
    to the hierarchical decomposition,
    the normalized probability distribution P[A = a] and the
    cumulative probability distribution P[A > a]
    """
    areas = array([tree.node[n]['cycle_area'] for n in tree.nodes_iter() \
            if not tree.node[n]['external']])

    areas /= areas.max()

    hist, bin_edges = histogram(areas, bins=bins, density=True)

    normed = hist/bins
    cumul = 1. - cumsum(normed)
    
    return areas, normed, cumul

def low_level_avg_asymmetries(tree, degree, Delta, 
        attr='asymmetry-simple'):
    """ Cuts the tree at given degree level and calculates the
    average asymmetries for the resulting subtrees.
    """
    tree_new = tree.copy()

    nodes_to_rem = [n for n in tree.nodes_iter() 
            if tree.node[n]['subtree-degree'] >= degree]
    tree_new.remove_nodes_from(nodes_to_rem)
    
    roots = [n for n in tree_new.nodes_iter() 
            if len(tree_new.predecessors(n)) == 0 and
            len(tree_new.successors(n)) == 2]
    
    return subtree_asymmetries(tree_new, roots, Delta, attr=attr)

def subtree_asymmetries(tree, roots, Delta, attr='asymmetry-simple'):
    """ Calculates the average asymmetry functions for the
    subtrees rooted at the nodes given in roots.
    """
    subtrees = [nx.DiGraph(tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]
    
    reslt = []
    for s, r in izip(subtrees, roots):
        s.graph['root'] = r
        
        degree = s.node[r]['subtree-degree']
        degrees = array(sorted(list(set([
            s.node[n]['subtree-degree'] 
            for n in s.nodes_iter()]))))

        reslt.append([degrees, 
            [average_asymmetry(s, d, Delta, attr=attr) 
                for d in degrees]])

    return reslt

def get_subtrees(tree, roots, mode='all', area=0, degree=0):
    """ Extracts the subtrees rooted at roots from tree.
    If a mode is given, further restricts to a sub-subtree which
    has some desired property.

    Parameters:
        tree: The hierarchical, marked tree we are interested in.

        roots: The root node ids of the subtrees we are interested in

        mode: 
            'all': extract full subtree
            'area': extract subtree whose loop area is closest to area
            'degree': extract subtree whose degree is closest to degree

        area: The area for the 'area' mode

        degree: The degree for the 'degree' mode

    Returns:
        subtrees: List of requested subtrees

        roots: List of roots of the requested subtrees
    """
    # Obtain subtrees as subgraphs and properly set root nodes
    subtrees = [nx.DiGraph(tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    if mode == 'area':
        roots = []
        for st in subtrees:
            # Find node with area closest to area
            ar, root = min([(abs(data['cycle_area'] - area), r) 
                for r, data in st.nodes_iter(data=True)])
            ar = st.node[root]['cycle_area']

            roots.append(root)

            print "Subtree closest to {} has area {}, degree {}, root {}".format(area,
                    ar, st.node[root]['subtree-degree'], root)
        
        # Recalculate subtrees
        subtrees = [nx.DiGraph(
            tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    elif mode == 'degree':
        roots = []
        for st in subtrees:
            # Find node with degree closest to degree
            de, root = min([(abs(data['subtree-degree'] - degree), r) 
                for r, data in st.nodes_iter(data=True)])
            de = st.node[root]['subtree-degree']

            roots.append(root)

            print "Subtree closest to {} has degree {}, area {}, root {}".format(
                    degree, de, st.node[root]['cycle_area'], root)
        
        # Recalculate subtrees
        subtrees = [nx.DiGraph(
            tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    # Save subtree roots in tree attributes
    for s, r in izip(subtrees, roots):
        s.graph['root'] = r
        s.node[r]['order'] = 0

    return subtrees, roots

def subtree_asymmetries_areas(tree, roots, attr='asymmetry-simple',
        area=0):
    """ Calculates the average asymmetry functions for the subtrees
    rooted at the nodes given in roots.
    Returns a list of lists (one for each subtree) of tuples
    of the form (asymmetry, degree, area)
    as well as the subtrees.

    This is the complete set of raw data from hierarchical
    decomposition.

    if area is equal to zero, returns the full selected subtrees.
    Otherwise, returns for each selected subtree that sub-subtree
    whose area is closest to the given area (in square pixels)
    """

    if area > 0:
        subtrees, roots = get_subtrees(tree, roots, mode='area',
                area=area)
    else:
        subtrees, roots = get_subtrees(tree, roots)
    
    reslt = []
    for s, r in izip(subtrees, roots):
        dist = [(s.node[n][attr], s.node[n]['subtree-degree'], 
            s.node[n]['cycle_area']) for n in s.nodes_iter()]

        dist = [(q, d, a) for q, d, a in dist if a > 0 and d > 0]

        reslt.append(dist)
    
    return reslt, subtrees

@plot.save_plot(name="low_level_avg_asymmetries")
def plot_low_level_avg_asymmetries(tree, frac, Delta,
        attr='asymmetry-simple', mode='default'):
    
    degree = frac*tree.node[tree.graph['root']]['subtree-degree']
    reslts = low_level_avg_asymmetries(tree, degree, Delta, attr=attr)

    plt.figure()
    plt.title("Average asymmetries, $\Delta = {}$, cut at: {}%, "
            "mode: {}".format(Delta, 100*frac, mode))
    plt.xlabel("Normalized subtree degree $\log(d)$")
    plt.ylabel("$\\bar Q(d)$")

    for degrees, asyms in reslts:
        plt.plot(log(degrees/float(max(degrees))), asyms, linewidth=2)

def load_graph(fname):
    """ Loads a graph from the given file
    """
    sav = storage.load(fname)

    ver = sav['version']
    
    SAVE_FORMAT_VERSION = 5
    if ver > SAVE_FORMAT_VERSION:
        print "File format version {} incompatible!".format(ver)
        sys.exit()

    leaf = sav['leaf']
    tree = sav['tree']
    filt = sav['filtration']
    remv = sav['removed-edges']
    prun = sav['pruned']

    return leaf, tree, filt, remv, prun

def thresholded_asymmetry(tree, thr):
    """ Return thresholded unweighted asymmetry
    """
    asyms = array([[d['subtree-degree'], d['partition-asymmetry']]
            for n, d in tree.nodes_iter(data=True)
            if d['subtree-degree'] > 1])

    degs = asyms[:,0]
    asyms_only = asyms[:,1]

    filtered = asyms_only[degs <= thr]

    return filtered.mean(), average(filtered, weights=degs[degs <= thr]), asyms

def analyze_tree(tree):
    # calculate metrics
    horton_strahler = 0
    shreve = 0
    
    print "Constructing marked trees."
    marked_tree = tree.copy()
    mark_subtrees(marked_tree)

    tree_no_ext = remove_external_nodes(tree)
    marked_tree_no_ext = tree_no_ext.copy()
    mark_subtrees(marked_tree_no_ext)
    
    print "Calculating tree asymmetry."
    tree_asymmetry = marked_tree.node[
            marked_tree.graph['root']]['asymmetry-unweighted']
    tree_asymmetry_no_ext = marked_tree_no_ext.node[
            marked_tree_no_ext.graph['root']]['asymmetry-unweighted']

    #areas, area_hist, cumul = normalized_area_distribution(tree, 100)
    areas = array([tree_no_ext.node[n]['cycle_area'] 
        for n in tree_no_ext.nodes_iter()])
    
    return horton_strahler, shreve, marked_tree, tree_no_ext, \
            marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
            areas

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Leaf Decomposition Analyzer.")
    parser.add_argument('INPUT', help="Input file in .gml format" \
    " containing the unpruned leaf data as a graph.")

    parser.add_argument('-p', '--plot', help="Plot results",
            action='store_true')
    parser.add_argument('-q', '--save-plots', help="Save plots to given"
            "files: -q bla.pdf will save to bla_plotname.pdf.",
            type=str, default="")
    parser.add_argument('-t', '--tree-heights', help="Use level in"
            " hierarchy to define height of tree nodes",
            action="store_true", default=False)
    parser.add_argument('-D', '--Delta', help="Window size for "
            "average tree asymmetry as a fraction of the "
            "total tree degree", type=float, default=0.1)
    parser.add_argument('-o', '--nice-tree-positions', action='store_true',
            help='Calculate nice nesting tree positions (slow!)')

    saveload_group = parser.add_mutually_exclusive_group()
    saveload_group.add_argument('-l', '--load', help="Load saved analyzed"
            " data instead of graph file", action='store_true')
    saveload_group.add_argument('-s', '--save', 
            help="Save analyzed data in pickle",
            type=str, default="")
    saveload_group.add_argument('-r', '--reanalyze', 
            help="Reanalyzes the given"
            " file, i.e. does everything except calculating"
            " the tree layout.", action='store_true')

    args = parser.parse_args()
    print "Loading file."
    
    if args.load or args.reanalyze:
        sav = storage.load(args.INPUT)

        horton_strahler = sav['horton-strahler-index']
        shreve = sav['shreve-index']
        tree_asymmetry = sav['tree-asymmetry']
        tree_asymmetry_no_ext = sav['tree-asymmetry-no-ext']
        areas = sav['tree-areas']
        marked_tree = sav['marked-tree']
        marked_tree_no_ext = sav['marked-tree-no-ext']
        tree_pos = sav['tree-positions']

        graph_file = sav['graph-file']

        leaf, tree, filt, remv, prun = load_graph(graph_file)
    else:
        graph_file = args.INPUT
        leaf, tree, filt, remv, prun = load_graph(graph_file)

        n_remv = len(remv)
        if n_remv > 0:
            print "Attention, workaround detected and"
            " removed {} collinear edges.".format(n_remv)
        
        horton_strahler, shreve, marked_tree, tree_no_ext, \
            marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
            areas = analyze_tree(tree)
        
        if args.nice_tree_positions:
            print "Calculating nice tree layout positions."
            tree_pos = nx.graphviz_layout(tree, prog='dot')
        else:
            print "Calculating standard tree layout positions."
            tree_pos = nx.spring_layout(tree, iterations=5)

    if args.reanalyze:
            horton_strahler, shreve, marked_tree, tree_no_ext, \
            marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
            areas = analyze_tree(tree)

            args.save = args.INPUT

    tree_pos_h = tree_pos.copy()
    for n in marked_tree.nodes_iter():
        tree_pos_h[n] = (tree_pos_h[n][0], marked_tree.node[n]['level'])

    # print info
    print "Whole tree unweighted nesting number:", 1-tree_asymmetry
    print "Whole tree unweighted nesting number without external nodes:", 1-tree_asymmetry_no_ext

    thresh_uw, thresh_wt, asyms_all = thresholded_asymmetry(marked_tree, 256)

    print "Thresholded (d=256) unweighted nesting number:", 1 - thresh_uw
    print "Thresholded (d=256) weighted nesting number:", 1 - thresh_wt

    # Save data
    if args.save != "":
        sav = { 'graph-file': graph_file,
                'horton-strahler-index': horton_strahler,
                'shreve-index': shreve,
                'tree-asymmetry': tree_asymmetry,
                'tree-asymmetry-no-ext': tree_asymmetry_no_ext,
                'tree-areas': areas,
                'marked-tree': marked_tree,
                'marked-tree-no-ext': marked_tree_no_ext,
                'tree-positions': tree_pos,
                }
        
        print "Saving analysis data."
        storage.save(sav, args.save)
        
        datadir = args.save + '_data'
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        savetxt(args.save + '_data/degrees_asymmetries.txt', asyms_all)
        print "Done."
    
    if args.plot or args.save_plots != "":
        print "Plotting/saving plots."
        fname = ""
        fname_no_ext = ""
        fext = ".png"
        if args.save_plots != "":
            fname, fext = os.path.splitext(args.save_plots)
            fname_no_ext = fname + "_no_ext"

        print fname, fext

        plt.figure()
        print "Drawing leaf."
        plot.draw_leaf(leaf, fname=fname, fext=fext, dpi=600)
        plt.figure()
        print "Drawing hierarchical trees."
        plot.draw_tree(marked_tree, pos=tree_pos, fname=fname, fext=fext)
        
        plt.figure()
        
        if args.tree_heights:
            pos = tree_pos_h
        else:
            pos = tree_pos

        plot.draw_tree(marked_tree_no_ext, pos=pos, 
                fname=fname_no_ext, fext=fext)

        plt.figure()
        print "Drawing filtration."
        plot.draw_filtration(filt, fname=fname, fext=fext, dpi=600)
        
        print "Calculating average asymmetries."
        Delta = args.Delta*marked_tree.node[marked_tree.graph['root']
                ]['subtree-degree']
        Delta_no_ext = args.Delta*marked_tree_no_ext.node[
                marked_tree_no_ext.graph['root']]['subtree-degree']

        avg_asymmetries_plot(marked_tree, Delta, fname=fname + 
                "_asym_{}".format(args.Delta), fext=fext)
        avg_asymmetries_plot(marked_tree_no_ext, Delta_no_ext, 
                mode="no-external", fname=fname_no_ext + 
                "_asym_{}".format(args.Delta), fext=fext)
        plot_low_level_avg_asymmetries(marked_tree, 
                0.75, Delta, fname=fname + "_asym_{}".format(args.Delta), 
                fext=fext)

        print "Cumulative size distribution."
        cum_size_plot(areas, fname=fname, fext=fext)

        print "Done."
    
    if args.plot:
        plt.show()
        raw_input()
