#!/usr/bin/env python

"""
tree_edit.py

Tool that reads data from analyzed leaf networks and allows for
graphcial selection of certain subtrees, followed by
averaging over the associated tree asymmetries.
Also exports all of the leaf metrics.

Henrik Ronellenfitsch 2013
"""

import os.path
import os
import sys

import csv
import argparse
from itertools import izip
from itertools import chain
from itertools import tee
from itertools import combinations
from collections import defaultdict

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.collections

from numpy import *
import numpy.linalg as linalg
import numpy.random

from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import scipy.interpolate
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg

import cvxopt as cvx
from cvxopt.modeling import variable, op

from decomposer import Cycle, Filtration
from tree_encode import canonize_tree, encode_tree, \
        random_binary_tree_bottomup, uniform_random_tree_sample
from fit_ellipse import *
import decomposer
import storage
import plot
import analyzer
from cycle_basis import polygon_area
from helpers import *

def sparse_laplacian(G, nodelist=None, weight='weight'):
    if nodelist is None:
        nodelist = G.nodes()
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n, m = A.shape
    diags = A.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
    return  D - A

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def minimum_spanning_edges(G,weight='weight',data=True):
    """Generate edges in a minimum spanning forest of an undirected 
    weighted graph.

    A minimum spanning tree is a subgraph of the graph (a tree)
    with the minimum sum of edge weights.  A spanning forest is a
    union of the spanning trees for each connected component of the graph.

    Parameters
    ----------
    G : NetworkX Graph
    
    weight : string
       Edge data key to use for weight (default 'weight').

    data : bool, optional
       If True yield the edge data along with the edge.
       
    Returns
    -------
    edges : iterator
       A generator that produces edges in the minimum spanning tree.
       The edges are three-tuples (u,v,w) where w is the weight.
    
    Examples
    --------
    >>> G=nx.cycle_graph(4)
    >>> G.add_edge(0,3,weight=2) # assign weight 2 to edge 0-3
    >>> mst=nx.minimum_spanning_edges(G,data=False) # a generator of MST edges
    >>> edgelist=list(mst) # make a list of the edges
    >>> print(sorted(edgelist))
    [(0, 1), (1, 2), (2, 3)]

    Notes
    -----
    Uses Kruskal's algorithm.

    If the graph edges do not have a weight attribute a default weight of 1
    will be used.

    Modified code from David Eppstein, April 2006
    http://www.ics.uci.edu/~eppstein/PADS/
    """
    # Modified code from David Eppstein, April 2006
    # http://www.ics.uci.edu/~eppstein/PADS/
    # Kruskal's algorithm: sort edges by weight, and add them one at a time.
    # We use Kruskal's algorithm, first because it is very simple to
    # implement once UnionFind exists, and second, because the only slow
    # part (the sort) is sped up by being built in to Python.
    from networkx.utils import UnionFind
    if G.is_directed():
        raise nx.NetworkXError(
            "Mimimum spanning tree not defined for directed graphs.")
    
    def kfun(t):
        v = t[2].get(weight,1)
        if v > 0:
            return 1./v
        else:
            return 1e10

    subtrees = UnionFind()
    edges = sorted(G.edges(data=True), key=kfun)
    for u,v,d in edges:
        if subtrees[u] != subtrees[v]:
            if data:
                yield (u,v,d)
            else:
                yield (u,v)
            subtrees.union(u,v)

def minimum_spanning_tree(G,weight='weight'):
    """Return a minimum spanning tree or forest of an undirected 
    weighted graph.

    A minimum spanning tree is a subgraph of the graph (a tree) with
    the minimum sum of edge weights.

    If the graph is not connected a spanning forest is constructed.  A
    spanning forest is a union of the spanning trees for each
    connected component of the graph.

    Parameters
    ----------
    G : NetworkX Graph
    
    weight : string
       Edge data key to use for weight (default 'weight').

    Returns
    -------
    G : NetworkX Graph
       A minimum spanning tree or forest. 
    
    Examples
    --------
    >>> G=nx.cycle_graph(4)
    >>> G.add_edge(0,3,weight=2) # assign weight 2 to edge 0-3
    >>> T=nx.minimum_spanning_tree(G)
    >>> print(sorted(T.edges(data=True)))
    [(0, 1, {}), (1, 2, {}), (2, 3, {})]

    Notes
    -----
    Uses Kruskal's algorithm.

    If the graph edges do not have a weight attribute a default weight of 1
    will be used.
    """
    T=nx.Graph(minimum_spanning_edges(G,weight=weight,data=True))
    # Add isolated nodes
    if len(T)!=len(G):
        T.add_nodes_from([n for n,d in G.degree().items() if d==0])
    # Add node and graph attributes as shallow copy
    for n in T:
        T.node[n]=G.node[n].copy()
    T.graph=G.graph.copy()
    return T


def lowres_graph(leaf, perc=70):
    """ Creates a low-resolution version of the given graph
    by removing the edges in the given percentile of the max."""
    lowres = nx.Graph(leaf)

    conds = [d['conductivity'] for a, b, d in lowres.edges_iter(data=True)]
    perccond = percentile(conds, perc)

    edges_to_rem = [(u, v) for u, v, d in lowres.edges_iter(data=True) 
            if d['conductivity'] < perccond]

    lowres.remove_edges_from(edges_to_rem)

    return lowres

def lowres_graph_edges(leaf, edges_to_rem=4000):
    """ Creates a low res version of the graph by plotting
    only the largest edges.
    """
    lowres = nx.Graph(leaf)
    if lowres.number_of_edges() <= edges_to_rem:
        return lowres

    edges = sorted([(d['conductivity'], (u, v))
            for u, v, d in lowres.edges_iter(data=True)])
    
    n_edges = len(edges)
    #edges_to_rem = 4*int(sqrt(n_edges))

    lowres.remove_edges_from(e[1] for e in edges[:n_edges-edges_to_rem])
    
    return lowres

def fix_artifacts_heuristic(G):
    """ Sloppy fix to remove artifacts from skeletonization at the major
    veins.
    """
    GG = G.subgraph(G.nodes())
    
    # find thick veins
    veins = sorted([(d['conductivity'], (u, v)) for u, v, d
            in GG.edges_iter(data=True)], reverse=True)

    n = len(veins)
    thick = veins[:10*int(sqrt(n))]

    _, edges = zip(*veins)
    
    edges_to_rem = []
    thr = 0.5
    for u, v in edges:
        # find neighbors
        n_u = [(u, n) for n in GG.neighbors(u) if not n == v]
        n_v = [(v, n) for n in GG.neighbors(v) if not n == u]
        ns = n_u + n_v
        
        d = GG[u][v]['conductivity']
        for w, x in ns:
            if GG[w][x]['conductivity'] < thr*d:
                edges_to_rem.append((w, x))

    GG.remove_edges_from(edges_to_rem)
    return GG

class TreeEditor(object):
    def __init__(self, fname, lowres=False, interactive=True, ext=True,
            segment_degree=250, fix_artifacts=False):
        self.lowres = lowres
        self.segment_degree=segment_degree

        self.load(fname, ext, fix_artifacts)

        if interactive:
            self.init_window()    
            self.edit_loop()

    def load(self, fname, ext, fix_artifacts):
        print "Loading file {}.".format(fname)
        self.filename = fname
        self.data_dir = self.filename + "_data/"
        self.stats_dir = self.filename + "_stats/"
        sav = storage.load(fname)

        self.horton_strahler = sav['horton-strahler-index']
        self.shreve = sav['shreve-index']
        self.tree_asymmetry = sav['tree-asymmetry']
        self.tree_asymmetry_no_ext = sav['tree-asymmetry-no-ext']
        self.areas = sav['tree-areas']
        self.marked_tree = sav['marked-tree']
        self.marked_tree_no_ext = sav['marked-tree-no-ext']
        self.tree_pos = sav['tree-positions']
        
        # If our data is just a leaf section, use tree without external
        # loops
        if not ext:
            print "Using nesting tree without external loops."
            self.marked_tree = self.marked_tree_no_ext
        
        # Recalculate heights
        for n in self.marked_tree.nodes_iter():
            self.tree_pos[n] = (self.tree_pos[n][0], 
                    self.marked_tree.node[n]['level'])

        self.graph_file = sav['graph-file']
        self.graph_name = os.path.basename(self.graph_file).split('.')[0]

        self.leaf, self.tree, self.filt, self.remv, self.prun = \
                analyzer.load_graph(self.graph_file)
        
        # fix artifacts for density measurement
        if fix_artifacts:
            self.fixed_leaf = fix_artifacts_heuristic(self.leaf)
        else:
            self.fixed_leaf = self.leaf
        
        # Calculate canonical ordering of tree
        canonize_tree(self.marked_tree)
        canonize_tree(self.marked_tree_no_ext)

        # Calculate loop paths
        self.loop_mtp_paths = [(n, decomposer.cycle_mtp_path( \
            self.marked_tree.node[n]['cycle'])) \
            for n in self.marked_tree.nodes_iter()
            if len(self.marked_tree.successors(n)) == 0
            and isinstance(self.marked_tree.node[n]['cycle'], Cycle)]

        # pre-calculate positions
        self.node_positions = [(n, self.tree_pos[n]) 
                for n in self.tree_pos.keys()]

        # low resolution graph if needed
        if self.lowres:
            self.lowres_graph = lowres_graph_edges(self.prun)
        else:
            self.lowres_graph = self.fixed_leaf

        self.selected_nodes = []
        self.selected_nodes_pts = {}
        self.selected_nodes_cycles = {}

    def init_window(self, draw_full=False):
        plt.ion()

        #plt.figure()
        #plot.draw_leaf(self.leaf)
        #plt.savefig("leaf.png", dpi=1600)
        #plt.close()
        #raw_input()

        self.fig = plt.figure(1)
        plt.clf()
        self.leaf_subplot = plt.subplot(211)
    
        if draw_full:
            plot.draw_leaf(self.fixed_leaf, title=self.graph_name)
        else:
            plot.draw_leaf(self.lowres_graph, title=self.graph_name)

        self.tree_subplot = plt.subplot(212)
        self.tree_pos, self.tree_edges = plot.draw_tree(self.marked_tree,
                pos=self.tree_pos, return_edges=True)

        plt.show()

    def show_sqrtn_thickest(self):
        """ Mark the sqrt(n) thickest edges in red to show that
        these are the "major" veins.
        """
        plt.subplot(211)
        
        edges = sorted([(d['conductivity'], (u, v))
                for u, v, d in self.leaf.edges_iter(data=True)],
                reverse=True)
        
        n = len(edges)
        Rs, thickest = zip(*edges[:int(10*sqrt(n))])
        
        plot.draw_leaf(self.leaf, edge_list=thickest, color='r')
        
        #extent = plt.gca().get_window_extent().transformed(
        #        self.fig.dpi_scale_trans.inverted())
        plt.savefig('{}_10sqrt(N).png'.format(self.graph_name),
                bbox_inches='tight', dpi=600)

    def on_click_select(self, event):
        if event.button == 1 and \
                event.xdata != None and event.ydata != None and \
                event.inaxes == self.tree_subplot:

            # Select a loop directly from the tree
            x, y = event.xdata, event.ydata

            x_disp, y_disp = self.tree_subplot.transData.transform((x,y))

            for n, (x_s, y_s) in self.node_positions:
                xs_disp, ys_disp = \
                        self.tree_subplot.transData.transform((x_s, y_s))

                dist = sqrt((xs_disp - x_disp)**2 + (ys_disp - y_disp)**2)
                if dist < 3:
                    self.select_node(n)
                    break
        elif event.button == 1 and \
                event.xdata != None and event.ydata != None and \
                event.inaxes == self.leaf_subplot:
            # Select a loop from the leaf representation
            x, y = event.xdata, event.ydata

            loops = [n for n, l in self.loop_mtp_paths
                    if l.contains_point((x, y))]

            if len(loops) == 1:
                self.select_node(loops[0])

    def on_key_press_select(self, event):
        if len(self.selected_nodes) > 0:
            last_sel = self.selected_nodes[-1]

            if event.key == 'u':
                pred = self.marked_tree.predecessors(last_sel)

                if len(pred) == 1 and self.select_node(pred[0]):
                    self.select_node(last_sel)
                    self.prev_selected_node = last_sel

            elif event.key == 'y':
                succ = self.marked_tree.successors(last_sel)

                if len(succ) == 2 and self.select_node(succ[0]):
                    self.select_node(last_sel)
                    self.prev_selected_node = last_sel

            elif event.key == 'x':
                succ = self.marked_tree.successors(last_sel)

                if len(succ) == 2 and self.select_node(succ[1]):
                    self.select_node(last_sel)
                    self.prev_selected_node = last_sel

            elif event.key == 'g':
                # Guess predecessor which is second order loop
                # based on differences of degrees of selected nodes
                
                # Go up one step up in the hierarchy to avoid
                # starting problems
                pred = self.marked_tree.predecessors(last_sel)

                if len(pred) != 1:
                    return

                cur_node = pred[0]
                
                # Do actual heuristics
                sel_degree = self.marked_tree.node[cur_node][
                        'subtree-degree']

                pred = self.marked_tree.predecessors(cur_node)

                if len(pred) != 1:
                    return

                next_node = pred[0]

                last_diff = self.marked_tree.node[next_node][
                        'subtree-degree'] - \
                        self.marked_tree.node[cur_node]['subtree-degree']
                
                cur_node = next_node
                pred = self.marked_tree.predecessors(cur_node)

                if len(pred) != 1:
                    return

                next_node = pred[0]

                cur_diff = self.marked_tree.node[next_node][
                        'subtree-degree'] - \
                        self.marked_tree.node[cur_node]['subtree-degree']
                
                while abs(cur_diff - last_diff) < 4*last_diff:
                    cur_node = next_node
                    pred = self.marked_tree.predecessors(cur_node)

                    if len(pred) != 1:
                        return

                    next_node = pred[0]

                    last_diff == cur_diff
                    cur_diff = self.marked_tree.node[next_node][
                            'subtree-degree'] - \
                            self.marked_tree.node[cur_node][
                                    'subtree-degree']

                self.select_node(last_sel)
                self.select_node(cur_node)
                self.prev_selected_node = last_sel

            elif event.key == 'b' and self.prev_selected_node != None:
                self.select_node(last_sel)
                self.select_node(self.prev_selected_node)

    def select_node(self, n):
        if n in self.selected_nodes:
            i = self.selected_nodes.index(n)

            self.selected_nodes_pts[n].remove()
            self.selected_nodes_cycles[n].remove()
            del self.selected_nodes[i]
            del self.selected_nodes_pts[n]
            del self.selected_nodes_cycles[n]

            plt.draw()
            return True
        else:
            cycle = self.marked_tree.node[n]['cycle']
            if not isinstance(cycle, Cycle):
                print "Selected external loop."
                return False

            self.selected_nodes.append(n)

            x, y = self.tree_pos[n]
            
            plt.subplot(212)
            self.selected_nodes_pts[n] = plt.plot(x, y, marker='o')[0]

            plt.subplot(211)

            cycle_edges = cycle.edges
            n_cycle_edges = len(cycle_edges)
            cy = plot.draw_leaf_raw(self.leaf, 
                    edge_list=cycle_edges, color='r', fixed_width=True,
                    title=self.graph_name)

            self.selected_nodes_cycles[n] = cy

            plt.draw()

            print "Selection area:", self.marked_tree.node[n]['cycle_area']
            
            radii = cycle.radii()
            print "Width dist:", radii.mean(), "+-", radii.std()
            
            return True
    
    def clean_selection(self):
        for n in self.selected_nodes:
            self.selected_nodes_pts[n].remove()
            self.selected_nodes_cycles[n].remove()
            del self.selected_nodes_pts[n]
            del self.selected_nodes_cycles[n]
        
        self.selected_nodes = []

        plt.show()

    def select_subtree(self):
        print """Select subtrees by clicking on the respective
        nodes or inside loops.
        Press 'u' to select the predecessor of the last selected node.
        Press 'y' to select the first successor.
        Press 'x' to select the second successor.
        Press 'g' to guess the next second-order loop.
        Press 'b' to undo.
        Press ENTER when done."""

        self.clean_selection()
        
        cid_click = self.fig.canvas.mpl_connect('button_press_event', 
                self.on_click_select)
        cid_press = self.fig.canvas.mpl_connect('key_press_event',
                self.on_key_press_select)

        raw_input()

        self.fig.canvas.mpl_disconnect(cid_click)
        self.fig.canvas.mpl_disconnect(cid_press)
        
        print "Selection:"
        print self.selected_nodes
        
        sa = array(self.selected_nodes)[newaxis,:]
        savetxt(self.filename + "_subtree_selection.txt",
            sa, fmt="%d", delimiter=", ")

    def select_subtree_indices(self):
        self.clean_selection()

        inds = raw_input("Enter list of indices separated by commas: ")
        reader = csv.reader([inds], skipinitialspace=True)

        iis = next(reader)
        
        for i in iis:
            self.select_node(int(i))
    
    def analyzer_asymmetry_to_xy(self, reslt):
        """ Takes the asymmetry data from the analyzer and converts
        it into a nice functional form by interpolating such that
        a large amount of degrees is covered.
        """
        # Interpolate to get as many data points as possible
        interps = [scipy.interpolate.interp1d(x, y, kind='nearest',
            bounds_error=False) 
                for x, y in reslt]

        xs = [x for x, y in reslt]

        xs = sorted(list(set(list(chain.from_iterable(xs)))))
        x_max = min([max(x) for x, y in reslt])
        
        xs = [x for x in xs if x <= x_max]

        ys = array([array([fun(x) for x in xs]) for fun in interps])     

        mean_ys = mean(ys, axis=0)
        std_ys = std(ys, axis=0)
        
        return xs, ys, mean_ys, std_ys

    def calculate_asymmetry(self, show_plot=True, interactive=True):
        if len(self.selected_nodes) == 0:
            print "Nothing selected! No can do."
            return

        asymmetries = [self.marked_tree.node[n]['asymmetry-simple'] 
                for n in self.selected_nodes]

        #all_partition_asymmetries = [d['partition-asymmetry'] 
        #        for n, d in self.marked_tree.nodes_iter(data=True)
        #        if d['partition-asymmetry'] != None]

        mean_asym = mean(asymmetries)
        std_asym = std(asymmetries)

        print "Average tree asymmetry: {} +- {}".format(mean_asym,
                std_asym)
        
        Delta = 20
        
        if interactive:
            inp = raw_input("Delta [{}]: ".format(Delta))
            if inp != "":
                Delta = int(inp)
        
        # Calculate asymmetry curves, weighted and unweighted
        reslt = analyzer.subtree_asymmetries(self.marked_tree,
                self.selected_nodes, Delta)

        xs, ys, mean_ys, std_ys = self.analyzer_asymmetry_to_xy(reslt)

        reslt_unweighted = analyzer.subtree_asymmetries(self.marked_tree,
                self.selected_nodes, Delta, attr='asymmetry-unweighted')

        xs_u, ys_u, mean_ys_u, std_ys_u = self.analyzer_asymmetry_to_xy(
                reslt_unweighted)
        
        # Full raw data for later use...
        raw_weighted_segments, segment_subtrees = \
                analyzer.subtree_asymmetries_areas(
                self.marked_tree,
                self.selected_nodes)
        
        raw_unweight_segments, _ = analyzer.subtree_asymmetries_areas(
                self.marked_tree,
                self.selected_nodes, attr='asymmetry-unweighted')

        raw_weighted = array(list(chain.from_iterable(
            raw_weighted_segments)))
        raw_unweight = array(list(chain.from_iterable(
            raw_unweight_segments)))

        full_raw_weighted = array(analyzer.subtree_asymmetries_areas(
                self.marked_tree, [self.marked_tree.graph['root']])[0][0])

        full_raw_unweight = array(analyzer.subtree_asymmetries_areas(
                self.marked_tree, [self.marked_tree.graph['root']],
                attr='asymmetry-unweighted')[0][0])

        # Plot everything
        if show_plot:
            plt.figure()
            plt.title("Average asymmetries of "\
                    "selected subtrees ($\Delta={}$), weighted".format(
                        Delta))
            plt.xlabel("Subtree degree $\delta$")
            plt.ylabel("Average asymmetry $\\bar Q(\delta)$")

            for i, (x, y) in izip(xrange(len(reslt)), reslt):
                plt.plot(x, y)
            
            plt.figure()
            plt.title("Averaged average asymmetries of"\
                    " selected subtrees ($\Delta={}$)".format(Delta))
            plt.xlabel("Subtree degree $\delta$")
            plt.ylabel("Average asymmetry $\\bar Q(\delta)$")

            plt.plot(xs, mean_ys, label="weighted")
            plt.fill_between(xs, mean_ys + std_ys, mean_ys - std_ys,
                    alpha=0.25)
           
            plt.plot(xs_u, mean_ys_u, label="unweighted", color='r')
            plt.fill_between(xs_u, mean_ys_u + std_ys_u, 
                    mean_ys_u - std_ys_u, alpha=0.25, facecolor='red')

            plt.legend(loc='lower right')
            
            #plt.figure()
            #plt.hist(all_partition_asymmetries)
            #plt.show()

            # DEBUG
            #plt.figure()
            #plt.title("Average asymmetry as"
            #        " function of loop area (unweighted)")
            #plt.xlabel("Area covered by loop $A/A_0$")
            #plt.ylabel("Average tree asymmetry $\\bar Q$")
            #reslt = analyzer.subtree_asymmetries_areas(self.marked_tree,
            #        self.selected_nodes, attr='asymmetry-unweighted')
            #
            #a0 = max([d['cycle_area']
            #        for n, d in self.marked_tree.nodes_iter(data=True)])
            #for dist in reslt:
            #    qs, ds, aas = zip(*dist)
            #    plt.plot(aas/a0, qs, 'o')
            
            # Asymmetry statistics

            #plt.figure()
            #plt.title("Unweighted asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Probability density $\mu_Q(q)$")
            #plt.hist(raw_unweight[:,0], bins=50, normed=True, 
            #        range=(0.,1.))
            #plt.xlim(0, 1)

            #plt.figure()
            #plt.title("Weighted asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Probability density $\mu_Q(q)$")
            #plt.hist(raw_weighted[:,0], bins=50, normed=True, 
            #        range=(0.,1.))
            #plt.xlim(0, 1)

            #plt.figure()
            #plt.title("Unweighted subtree asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Subtree degree $\delta$")
            #plt.hist2d(raw_unweight[:,0], raw_unweight[:,1],
            #        bins=[50, 100], normed=True,
            #        range=((0, 1), (0.5, 100+0.5)))

            #plt.figure()
            #plt.title("Weighted subtree asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Subtree degree $\delta$")
            #plt.hist2d(raw_weighted[:,0], raw_weighted[:,1],
            #        bins=[50, 100], normed=True,
            #        range=((0, 1), (0.5, 100+0.5)))

            #plt.figure()
            #plt.title("Unweighted full asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Subtree degree $\delta$")
            #plt.hist2d(full_raw_unweight[:,0], full_raw_unweight[:,1],
            #        bins=[50, 100], normed=True,
            #        range=((0, 1), (0.5, 100+0.5)))

            #plt.figure()
            #plt.title("Weighted full asymmetry probability distribution")
            #plt.xlabel("Asymmetry $q$")
            #plt.ylabel("Subtree degree $\delta$")
            #plt.hist2d(full_raw_weighted[:,0], full_raw_weighted[:,1],
            #        bins=[50, 100], normed=True,
            #        range=((0, 1), (0.5, 100+0.5)))

            plt.show()

        # Make sure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Save everything for later easy use
        savetxt(self.data_dir + "segment_average.txt", 
                array([xs, mean_ys, std_ys]).T)

        savetxt(self.data_dir + "segment_data.txt",
                concatenate((array([xs]), ys)).T)

        savetxt(self.data_dir + "segment_average_unweighted.txt", 
                array([xs_u, mean_ys_u, std_ys_u]).T)

        savetxt(self.data_dir + "segment_data_unweighted.txt",
                concatenate((array([xs_u]), ys_u)).T)

        savetxt(self.data_dir + "segment_raw_weighted.txt",
                raw_weighted)

        savetxt(self.data_dir + "segment_raw_unweighted.txt",
                raw_unweight)

        savetxt(self.data_dir + "full_raw_weighted.txt",
                full_raw_weighted)

        savetxt(self.data_dir + "full_raw_unweighted.txt",
                full_raw_unweight)
        
        # Save all segments
        for segment, i in zip(raw_weighted_segments, 
                range(len(raw_weighted_segments))):
            savetxt(self.data_dir + \
                    "raw_weighted_segment_{:0>2}.txt".format(i+1),
                    array(segment))
        
        for segment, i in zip(raw_unweight_segments, 
                range(len(raw_unweight_segments))):
            savetxt(self.data_dir + \
                    "raw_unweighted_segment_{:0>2}.txt".format(i+1),
                    array(segment))
        
        # Save the full bare tree structure as a bit string
        self.save_full_tree()
        
        # Fixed area subtrees
        for subtree, i in izip(segment_subtrees, 
                xrange(len(segment_subtrees))):
            self.save_tree(subtree, self.data_dir + \
                    "segment_{:0>2}_ar_1e6_tree_enc.txt".format(i))

        # Fixed degree subtrees
        fixed_degree_strees, roots = analyzer.get_subtrees(
                self.marked_tree, self.selected_nodes, 
                mode='degree', degree=self.segment_degree)

        for subtree, i in izip(fixed_degree_strees, 
                xrange(len(fixed_degree_strees))):
            self.save_tree(subtree, 
                    "segment_{:0>2}_deg_{}_tree_enc.txt".format(i, 
                        self.segment_degree))

        # Back to main figure
        plt.figure(1)
    
    def save_tree(self, tree, fname):
        """ Saves the given tree into the given file name.

        Parameters:
            tree: NetworkX DiGraph tree.

            fname: File name to save encoded tree to.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        bare = encode_tree(tree)
        
        if len(bare) > 1:
            with open(fname, 'w') as f:
                f.write(bare)
        else:
            print "tree is empty, not saving."
            sys.exit(0)

    def save_full_tree(self):
        """ Saves the full tree into a default file.
        """
        print "Saving full tree with/without externals."

        self.save_tree(self.marked_tree,
                self.data_dir + "tree_enc.txt")

        self.save_tree(self.marked_tree_no_ext,
                self.data_dir + "tree_enc_no_ext.txt")


    def subgraph_inside_cycle(self, cycle):
        """ Returns the subgraph of self.leaf that is inside the
        cycle.

        Parameters:
            cycle: The Cycle object that defines the subgraph

        Returns:
            S: subgraph with all nodes inside the cycle.
        """

        p = decomposer.cycle_mtp_path(cycle)

        nodes_coords = [(n, (d['x'], d['y']))
                for n, d in self.leaf.nodes_iter(data=True)]
        
        nodes, coords = zip(*nodes_coords)

        in_cycle = p.contains_points(coords, radius=1.)
        inside_bunch = [nodes[i] for i in xrange(len(in_cycle))
                if in_cycle[i] == True]
        
        S = self.leaf.subgraph(inside_bunch)
        return S
    
    def follow_highest_width_path(self, G, edge, reverse=False):
        if reverse:
            return self.follow_highest_width_path(G, (edge[1], edge[0]))
        
        u, v = edge
        visited = set([u])
        path = [u]
        path_edges = [(u, v)]

        while True:
            path.append(v)
            visited.add(v)

            nexts = sorted([(G[v][n]['conductivity'], n)
                for n in G.neighbors_iter(v)
                if not n in visited])
            
            if len(nexts) > 0:
                v = nexts[-1][1]
                path_edges.append((path[-1], v))
            else:
                break

        return path, path_edges

    def follow_shallowest_descent_path(self, G, edge, reverse=False):
        if reverse:
            return self.follow_highest_width_path(G, (edge[1], edge[0]))
        
        u, v = edge
        visited = set([u])
        path = [u]
        path_edges = [(u, v)]

        while True:
            path.append(v)
            visited.add(v)

            a, b = path_edges[-1]
            cur_width = G[a][b]['conductivity']

            nexts = sorted([(G[v][n]['conductivity'], n)
                for n in G.neighbors_iter(v)
                if not n in visited and 
                G[v][n]['conductivity'] < cur_width])
            
            if len(nexts) > 0:
                v = nexts[-1][1]
                path_edges.append((path[-1], v))
            else:
                break

        return path, path_edges

    def smoothen_path(self, verts, reps=50):
        for i in xrange(reps):
            avgs = 0.25*(verts[:-2] + 2*verts[1:-1] + verts[2:])
            verts[1:-1] = avgs
    
    def fit_ellipse_to_points(self, x, y):
        """ Fit an ellipse to the given point set,
        return semimajor and semiminor axes
        """
        aa = fit_ellipse(x, y)
        a, b = ellipse_axis_length(aa)
        x0, y0 = ellipse_center(aa)
        phi = ellipse_angle_of_rotation(aa)
         
        return a, b, x0, y0, phi

    def main_vein_length_by_ellipse(self, show_plot=True):
        # estimate main vein length by fitting an ellipse to the
        # leaf margin and using twice the semimajor axis as length
        # estimate
        cy = self.marked_tree.node[self.marked_tree.graph['root']]['cycle']
        
        a, b, x0, y0, phi = \
                self.fit_ellipse_to_points(cy.coords[:,0], cy.coords[:,1])
        
        print "Fitted ellipse axes: {}, {}".format(2*a, 2*b)

        if show_plot:
            plt.figure()
            plot.draw_leaf(self.lowres_graph)
            plt.scatter(cy.coords[:,0], cy.coords[:,1])
            
            phis = arange(0, 2*pi, 0.01)

            xx = x0 + a*cos(phis)*cos(phi) - b*sin(phis)*sin(phi)
            yy = y0 + a*cos(phis)*sin(phi) + b*sin(phis)*cos(phi)
            plt.plot(xx, yy)
            plt.show()

        return 2*max(a, b)

    def main_vein_length_by_largest_pdist(self):
        # Calculate largest distance between any two points
        # on the leaf margin
        cy = self.marked_tree.node[self.marked_tree.graph['root']]['cycle']
        length = max(pdist(cy.coords))

        print "Largest distance between any two points on margin:", length

        return length

    def main_vein_length(self, G, show_plot=True):
        # Use only largest connected component, assuming it contains
        # the leaf
        G = G.subgraph(sorted_connected_components_copy(G)[0])
        
        # Now that this is fixed, onto the actual main veins
        veins = sorted([(d['conductivity'], d['weight'],
            sqrt((G.node[u]['x'] - G.node[v]['x'])**2 +
                (G.node[u]['y'] - G.node[v]['y'])**2), (u, v))
                for u, v, d in G.edges_iter(data=True)])

        lengths = [l for d, l, r, e in veins]
        geom_lengths = [r for d, l, r, e in veins]
        
        print "Fixing main vein length..."
        len_ellipse = self.main_vein_length_by_ellipse(show_plot=show_plot)
        len_pdist = self.main_vein_length_by_largest_pdist()

        # Sophisticated fix necessary.
        # Idea: find longest paths of highest width elements
        # and straighten them out.
        major_nodes = array([e for d, l, r, e in veins]).flatten()
        
        # We must not allow loops. This modified version of
        # MST finds a spanning tree including the largest edges,
        # so we always get the main vein
        G_major = minimum_spanning_tree(G.subgraph(major_nodes),
                weight='conductivity')
        
        highest_edge = veins[-1][3]
        
        # Detect main vein.
        p1, pe1 = self.follow_highest_width_path(G_major, highest_edge)
        p2, pe2 = self.follow_highest_width_path(G_major, highest_edge,
                reverse=True)
        
        # Find coordinates and smoothen them out
        main_vein_path = p2[::-1] + p1[2:]
        main_vein_coords = array([[G.node[n]['x'], G.node[n]['y']]
            for n in main_vein_path])
 
        self.smoothen_path(main_vein_coords)

        # Calculate length and save
        dx = diff(main_vein_coords, axis=0)
        lens = sqrt((dx**2).sum(-1)[...,newaxis])
        length = sum(lens)
        print "Main vein length (smoothened):", length
        
        diams = array([G[u][v]['conductivity'] for u, v in pe1] + \
                [G[u][v]['conductivity'] for u, v in pe2[1:]])
        
        if show_plot:
            plt.figure()
            plot.draw_leaf(self.lowres_graph)
            plot.draw_leaf(self.leaf, edge_list=pe1, color='r')
            plot.draw_leaf(self.leaf, edge_list=pe2, color='g')
            plt.show()

        # fix vein lengths
        #print main_vein_path
        for (u, v), fl in izip(pairwise(main_vein_path), lens):
            G[u][v]['weight'] = fl

        return lens, diams, len_ellipse, len_pdist
        
    def vein_stats(self, G):
        """ Returns vein statistics of graph G.

        Parameters:
            G: A networkx graph with edge attributes 'conductivity'
                and 'weight'

        Returns:
            veins: (conductivity, weight) pairs
            minor_vein_thresh: Threshold width for minor veins
            minor_vein_diameters
            minor_vein_lengths
        """
        # Vein statistics
        veins = np.array([[d['conductivity'], d['weight']]
                for u, v, d in G.edges_iter(data=True)])

        lengths = [l for d, l in veins]
        diameters = [d for d, l in veins]

        # minor veins - lower 95% of vein diameters
        minor_vein_thresh = percentile(diameters, q=95)

        minor_vein_lengths = [l for l in lengths
            if l > minor_vein_thresh]

        minor_vein_diameters = [d['conductivity'] 
            for u, v, d in G.edges_iter(data=True)
            if d['conductivity'] < minor_vein_thresh]

        return veins, minor_vein_thresh, minor_vein_diameters, \
               minor_vein_lengths
    
    def calculate_vein_distances(self):
        """ approximate vein distances by fitting ellipses
        to the areoles, and taking the semiminor axis as an
        estimate for the incircle radius
        """
        distances = []
        for n, d in self.marked_tree_no_ext.degree_iter():
            if d == 1:
                coords = self.marked_tree_no_ext.node[n]['cycle'].coords
                a, b, x0, y0, phi = self.fit_ellipse_to_points(
                        coords[:,0], coords[:,1])
                distances.append(min(a, b))
        
        distances = real(array(distances))
        distances = distances[logical_not(isnan(distances))]
        return distances
    
    def calculate_vein_distances_chebyshev(self):
        """ approximate vein distances by finding the chebyshev
        centers of the areoles, and taking the radii.
        """
        distances = []
        cvx.solvers.options['show_progress'] = False

        for n, d in self.marked_tree_no_ext.degree_iter():
            if d == 1:
                coords = self.marked_tree_no_ext.node[n]['cycle'].coords

                # find convex hull to make approximation
                # possible
                hull = ConvexHull(coords)
                coords = coords[hull.vertices,:]

                # shift to zero center of gravity
                cog = coords.mean(axis=0)
                
                coords -= cog
                # append last one
                coords = vstack((coords, coords[0,:]))
                
                # Find Chebyshev center
                X = cvx.matrix(coords)
                m = X.size[0] - 1

                # Inequality description G*x <= h with h = 1
                G, h = cvx.matrix(0.0, (m,2)), cvx.matrix(0.0, (m,1))
                G = (X[:m,:] - X[1:,:]) * cvx.matrix([0., -1., 1., 0.],
                        (2,2))
                h = (G * X.T)[::m+1]
                G = cvx.mul(h[:,[0,0]]**-1, G)
                h = cvx.matrix(1.0, (m,1))
                
                # Chebyshev center
                R = variable()
                xc = variable(2)
                lp = op(-R, [ G[k,:]*xc + R*cvx.blas.nrm2(G[k,:]) <= h[k] 
                    for k in xrange(m) ] +[ R >= 0] )

                lp.solve()
                R = R.value
                xc = xc.value           
                
                #plt.figure(facecolor='w')

                ## polyhedron
                #for k in range(m):
                #    edge = X[[k,k+1],:] + 0.1 * cvx.matrix([1., 0., 0., -1.], (2,2)) * \
                #        (X[2*[k],:] - X[2*[k+1],:])
                #    plt.plot(edge[:,0], edge[:,1], 'k')


                ## 1000 points on the unit circle
                #nopts = 1000
                #angles = cvx.matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
                #circle = cvx.matrix(0.0, (2,nopts))
                #circle[0,:], circle[1,:] = R*cvx.cos(angles), R*cvx.sin(angles)
                #circle += xc[:,nopts*[0]]

                ## plot maximum inscribed disk
                #plt.fill(circle[0,:].T, circle[1,:].T, facecolor = '#F0F0F0')
                #plt.plot([xc[0]], [xc[1]], 'ko')
                #plt.title('Chebyshev center (fig 8.5)')
                #plt.axis('equal')
                #plt.axis('off')
                #plt.show()

                if lp.status == 'optimal':
                    distances.append(R[0])

        return array(distances)
    
    def width_degree_distribution(self, show_plot=True):
        """ Calculate the vein width as a function of degree
        """
        widths_radii = []

        for n, d in self.marked_tree.nodes_iter(data=True):
            if d['cycle'] != None:
                rads = d['cycle'].radii()
                deg = d['subtree-degree']
                #w_r = [[deg, r] for r in rads]
                #widths_radii.extend(w_r)
                widths_radii.append([deg, rads.mean()])
        
        widths_radii = array(widths_radii)

        if show_plot:
            plt.figure()
            plt.scatter(widths_radii[:,0], widths_radii[:,1])
            plt.xlabel('subtree degree')
            plt.ylabel('mean vein radii')
            plt.show()

    def calculate_statistics(self, show_plot=True, interactive=True):
        """ Calculates vein statistics for the given leaf.
        """
        # widths-degrees
        self.width_degree_distribution(show_plot=show_plot)

        # Fix main vein lengths.
        main_lens, main_diams, main_len_ellipse, main_len_pdist = \
                self.main_vein_length(self.leaf, show_plot=show_plot)

        veins, minor_vein_thresh, minor_vein_diameters, \
                minor_vein_lengths = self.vein_stats(self.fixed_leaf)
        
        # largest loop (outer loop)
        leaf_area = self.marked_tree.node[self.marked_tree.graph['root']]['cycle_area']

        # leaf area
        #points = np.array([(d['x'], d['y']) 
        #    for n, d in self.fixed_leaf.nodes_iter(data=True)])
        #ch = ConvexHull(points)
        #
        #pts_closed = np.array(list(points[ch.vertices,:]) + 
        #        list([points[ch.vertices[0],:]]))
        #leaf_area = polygon_area(pts_closed)

        minor_vein_density = sum(minor_vein_lengths)/leaf_area

        minor_vein_diameter = mean(minor_vein_diameters)
        minor_vein_diameter_std = std(minor_vein_diameters)
        
        lengths = array([l for d, l in veins])
        diameters = array([d for d, l in veins])

        print "Minor vein diameter threshold:", minor_vein_thresh
        print "Minor vein density: {} 1/px".format(minor_vein_density)
        print "Minor vein diameter: {} +- {} px".format(minor_vein_diameter,
                minor_vein_diameter_std)
        print "Total vein density:", veins[:,1].sum()/leaf_area
        
        areole_areas = []
        for n, d in self.marked_tree_no_ext.degree_iter():
            if d == 1:
                areole_areas.append(
                        self.marked_tree_no_ext.node[n]['cycle_area'])
        
        num_areoles = len(areole_areas)
        print "Number of areoles:", num_areoles
        
        vein_distances = self.calculate_vein_distances_chebyshev()
        print "# Vein distances:", len(vein_distances)
        print "Avg. vein distance:", vein_distances.mean()

        if show_plot:
            # Vein statistics
            plt.figure()
            plt.title("Vein diameters")
            plt.xlabel("Vein diameter (px)")
            plt.ylabel("Number of veins")
            plt.hist(diameters, bins=50)
            plt.axvline(x=mean(diameters), color='r', linewidth=2)
            plt.axvline(x=median(diameters), color='g', linewidth=2)
            
            plt.figure()
            plt.title("Weighted vein diameters")
            plt.xlabel("diameter $\\times$ length")
            plt.hist(diameters*lengths, bins=50)

            plt.figure()
            plt.title("Areole areas")
            plt.xlabel("areole area ($\mathrm{px}^2$)")
            plt.hist(areole_areas, bins=50)
            plt.axvline(x=mean(areole_areas), color='r', linewidth=2)
            plt.axvline(x=median(areole_areas), color='g', linewidth=2)

            plt.figure()
            plt.title("Vein lengths")
            plt.xlabel("Vein length (px)")
            plt.ylabel("Number of veins")
            plt.hist(lengths, bins=50)
            plt.axvline(x=mean(lengths), color='r', linewidth=2)
            plt.axvline(x=median(lengths), color='g', linewidth=2)

            plt.figure()
            plt.title("Vein distances")
            plt.xlabel("Vein distance (px)")
            plt.ylabel("Number of areoles")
            plt.hist(vein_distances, bins=50)
            plt.axvline(x=mean(vein_distances), color='r', linewidth=2)
            plt.axvline(x=median(vein_distances), color='g', linewidth=2)

            plt.show()

        # Save statistics
        # Make sure stats directory exists
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        
        # Save stuff
        savetxt(self.stats_dir + 'leaf_area.txt',
                array([leaf_area]))

        savetxt(self.stats_dir + 'minor_vein_threshold.txt',
                array([minor_vein_thresh]))

        savetxt(self.stats_dir + 'minor_vein_diameters.txt',
                array(minor_vein_diameters))

        savetxt(self.stats_dir + 'minor_vein_lengths.txt',
                array(minor_vein_lengths))

        savetxt(self.stats_dir + 'vein_diameters_lengths.txt',
                array(veins))

        savetxt(self.stats_dir + 'number_areoles.txt', 
                array([num_areoles]))

        savetxt(self.stats_dir + 'areole_areas.txt', 
                array(areole_areas))

        savetxt(self.stats_dir + 'vein_distances.txt', 
                vein_distances)

        savetxt(self.stats_dir + 'main_vein_length.txt',
                array([[sum(main_lens)], [main_len_ellipse], 
                    [main_len_pdist]]).T)

        savetxt(self.stats_dir + 'main_vein_lengths_diameters.txt',
                array([main_lens, main_diams]).T)

        # Statistics for selected subgraphs
        for n, i in zip(self.selected_nodes, 
                range(len(self.selected_nodes))):
            cy = self.marked_tree.node[n]['cycle']
            S = self.subgraph_inside_cycle(cy)
            
            veins, minor_vein_thresh, minor_vein_diameters, \
                minor_vein_lengths = self.vein_stats(S)

            area = self.marked_tree.node[n]['cycle_area']

            minor_vein_density = sum(minor_vein_lengths)/area

            minor_vein_diameter = mean(minor_vein_diameters)
            minor_vein_diameter_std = std(minor_vein_diameters)
            
            print "Segment", i
            print "Minor vein diameter threshold:", minor_vein_thresh
            print "Minor vein density: {} 1/px".format(minor_vein_density)
            print "Minor vein diameter: {} +- {} px".format(\
                    minor_vein_diameter, minor_vein_diameter_std)


                  # Save stuff
            savetxt(self.stats_dir + 'segment_{:0>2}_area.txt'.format(i),
                    array([area]))

            savetxt(self.stats_dir + \
                    'segment_{:0>2}_minor_vein_threshold.txt'.format(i),
                    array([minor_vein_thresh]))

            savetxt(self.stats_dir + \
                    'segment_{:0>2}_minor_vein_diameters.txt'.format(i),
                    array(minor_vein_diameters))

            savetxt(self.stats_dir + \
                    'segment_{:0>2}_minor_vein_lengths.txt'.format(i),
                    array(minor_vein_lengths))

            savetxt(self.stats_dir + \
                    'segment_{:0>2}_vein_diameters_lengths.txt'.format(i),
                    array(veins))
    
    def calculate_angle_statistics(self, interactive=True, show_plot=True):
        """ Finds the distributions of angles between neighboring veins
        from the geometric data in the graph.
        """
        angle_data = []
        for n, d in self.leaf.nodes_iter(data=True):
            pos = array([d['x'], d['y']])

            neigh_pos = array([[self.leaf.node[m]['x'], 
                self.leaf.node[m]['y']]
                for m in self.leaf.neighbors_iter(n)])

            # Check if there are (more than) 2 neighbors
            # and calculate the angles
            n_neigh = len(neigh_pos)
            if n_neigh >= 2: 
                neigh_vecs = neigh_pos - pos
                
                # Absolute angls w.r.t. x axis
                abs_angles = arctan2(neigh_vecs[:,1], neigh_vecs[:,0])
                abs_angles[abs_angles < 0] += 2*pi
                
                # Sorting by absolute angle finds neighboring vein vectors
                abs_angles = sorted(abs_angles)
                
                # Relative angles between neighboring vein vectors
                rel_angles = zeros(n_neigh)
                rel_angles[:-1] = diff(abs_angles)
                rel_angles[-1] = 2*pi - (abs_angles[-1] - abs_angles[0])
                
                angle_data.extend(rel_angles)

        angle_data = array(angle_data)

        if show_plot:
            plt.figure()
            plt.title("Angular distribution")
            plt.xlabel("Angle (deg)")
            plt.ylabel("$N$")
            plt.hist(180./pi*angle_data, bins=200)
            plt.show()
        
        # Bimodality coefficient
        n = len(angle_data)
        beta = (1. + scipy.stats.skew(angle_data)**2)/\
                (scipy.stats.kurtosis(angle_data) + \
                3.*(n-1.)**2/((n-2.)*(n-3.)))
        
        print "Bimodality coefficient:", beta
        
        # Make sure stats directory exists
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        
        # Save stuff
        savetxt(self.stats_dir + \
                'branching_angles.txt', angle_data)

    def get_degree_area_distribution(self):
        # Area distribution as a function of subtree degree
        areas = [(self.marked_tree.node[n]['cycle_area'], \
                self.marked_tree.node[n]['subtree-degree'])
                for n in self.marked_tree.nodes_iter()]
        areas = [(a, d) for a, d in areas if a > 0]    
        
        return areas

    def degree_area_distribution(self, show_plot=True):
        """ Calculates the distribution of degrees and the
        loop areas.
        """
        # Make sure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        areas = self.get_degree_area_distribution()
        savetxt(self.data_dir + "segment_areas.txt",
                array(areas))
        
        aas, ds = zip(*areas)

        # Linear fit
        p = polyfit(ds, aas, 1)
        print "Linear fit: A = {} + {}*d".format(*p[::-1])

        if show_plot:
            plt.figure()
            plt.xlabel("Subtree degree $\delta$")
            plt.ylabel("Loop area (sqr. pixels)")
            plt.plot(ds, aas, 'o')

            x = linspace(min(ds), max(ds), 100)
            plt.plot(x, p[0]*x + p[1])
            plt.show()
    
    def window_average_plot(self, tree, title="", size=50,
            attr='partition-asymmetry', ylabel='partition asymmetry',
            fig=None):
        """ Shows a plot of the asymmetry of the given marked tree
        """
        maxdeg = tree.node[tree.graph['root']]['subtree-degree']
        qs = [(d[attr], d['subtree-degree']) 
                for n, d in tree.nodes_iter(data=True)
                if d['partition-asymmetry'] != None]

        #pas = [[d[attr] for n, d in node] 
        #        for node in nodes]

        #qs = array([mean(pa) for pa in pas])
        #q_stds = array([std(pa) for pa in pas])
        #
        qw = tree.node[tree.graph['root']]['asymmetry-simple']
        qu = tree.node[tree.graph['root']]['asymmetry-unweighted']

        fig_hist2d = plt.figure()
        x, y = zip(*qs)
        plt.title(title)
        plt.hist2d(log2(y), x, bins=50, norm=LogNorm())
        plt.plot([0, log2(maxdeg)], 2*[qw], linewidth=4, alpha=0.7, 
            ls=':', color='k')
        plt.plot([0, log2(maxdeg)], 2*[qu], linewidth=4, alpha=0.7, 
            ls='--', color='k')
        plt.colorbar()
        plt.xlabel('log_2(subtree degree)')
        plt.ylabel(ylabel)

        #plt.figure()
        #plt.title(title)
        #plt.xlabel(ylabel)
        #plt.ylabel("Cumulative probability distribution")
        #plt.hist(x, bins=50, cumulative=True, normed=True)
        #plt.show()

        fig_hist = plt.figure()
        plt.xlabel(ylabel)
        plt.ylabel("Probability density")
        hst = plt.hist(x, bins=50, normed=True, log=True, range=(0,1),
                alpha=0.7, label="real leaf")
        plt.xlim(0, 1)
        plt.show()

        return array(qs), hst, fig_hist, fig_hist2d

    def window_average(self, show_plot=True):
        """ Performs a running average with fixed degree window
        over the hierarchical tree.
        """
        attr='partition-asymmetry'
        label='partition asymmetry'
        #print "Calculating average over actual tree"
        # Show actual tree
        qs_leaf, hst_leaf, fig_hist, fig_hist2d = self.window_average_plot(
                self.marked_tree_no_ext,
                "Leaf", attr=attr, ylabel=label)
        
        areole_areas = array(
                [self.marked_tree_no_ext.node[n]['cycle_area'] 
                for n, d in self.marked_tree_no_ext.degree_iter()
                if d == 1])
        
        num_areoles = len(areole_areas)

        qs = []
        
        # Random binary tree, bottom up approach
        rt = raw_input("Random tree (B)ottom up, (u)niform? ")
        if rt == 'u':
            random_tree_mode = 'uniform sample'
            samp, degs = uniform_random_tree_sample(num_areoles,
                    25, 0.01*num_areoles)
            print degs

        else:
            random_tree_mode = 'bottom-up sample'
            samp = [random_binary_tree_bottomup(num_areoles) for i in
                    xrange(25)]
        
        for rand_tree in samp:
            # Random binary tree, uniformly chosen
            analyzer.mark_subtrees(rand_tree)
            qs.extend([(d[attr], d['subtree-degree'])
                    for n, d in rand_tree.nodes_iter(data=True)
                    if d['partition-asymmetry'] != None])

        x, y = zip(*qs)
        # Plot both p.a. histograms into one figure
        plt.figure(fig_hist.number)
        plt.xlabel(label)
        plt.ylabel("Probability density")
        hst_rand = plt.hist(x, bins=50, normed=True, 
                log=True, range=(0,1), alpha=0.7, 
                label="random tree ({})".format(random_tree_mode))
        plt.xlim(0, 1)
        plt.legend()
        plt.show()
        
        # Random tree total histogram
        plt.figure()
        plt.hist2d(log2(y), x, bins=50, norm=LogNorm())
        plt.colorbar()
        plt.xlabel('log_2(subtree degree)')
        plt.ylabel(label)
        plt.title("Random tree ({})".format(random_tree_mode))
        
        # difference histogram
        leaf_n = hst_leaf[0]
        rand_n = hst_rand[0]
        diffs = leaf_n - rand_n

        plt.figure()
        plt.xlabel(label)
        plt.ylabel("(real leaf $-$ random tree)/real leaf")
        bins = hst_leaf[1][:-1]
        plt.bar(bins, diffs/leaf_n, width=0.8*(bins[1]-bins[0]))

        plt.figure()
        plt.xlabel(label)
        plt.ylabel("real leaf $-$ random tree")
        plt.bar(bins, diffs, width=0.8*(bins[1]-bins[0]))
         
        print "KS Test random vs real", \
                scipy.stats.ks_2samp(qs_leaf[:,0], x)
        
        # Plot difference between heat maps
        plt.figure()

        max_d_leaf = amax(qs_leaf[:,1])
        max_d_rand = amax(x)

        max_d = max(max_d_leaf, max_d_rand)

        hst2d_leaf = histogram2d(log2(qs_leaf[:,1]), qs_leaf[:,0],
                bins=50, normed=True, range=((1, log2(max_d)), (0, 1)))
        hst2d_rand = histogram2d(log2(y), x, bins=50, normed=True,
                range=((1, log2(max_d)), (0, 1)))
        
        diff2 = abs(hst2d_leaf[0] - hst2d_rand[0])
        # Nicer plots this way.
        diff2[diff2 < 1e-10] = 0
        
        n = diff2.shape[0]
        X, Y = meshgrid(linspace(1, log2(max_d), n), 
            linspace(0, 1, n))
        
        # set up transparent colormap
        #cmap = plt.get_cmap('bwr')
        #cmap._init()
        #alphas = abs(linspace(-1, 1, cmap.N))
        #cmap._lut[:-3,-1] = alphas

        nmax = amax(abs(diff2))
        plt.pcolormesh(X, Y, diff2.T, norm=LogNorm())
                #vmin=0, vmax=nmax)
        plt.colorbar()
        plt.xlabel("$\log_2$(subtree degree)")
        plt.ylabel("partition asymmetry")
        plt.title("|real leaf $-$ random tree|")
        plt.show()
        #print "Calculating average over random tree"
        #self.window_average_plot(rand_tree, "Random tree",
        #        attr=attr, ylabel=label)
        
    def plot_qds_attribute(self, attribute, name, tree=None):
        """
            Returns vector of asymmetry, degree data
            for given attribute,
            plots it.
        """
        if tree == None:
            tree = self.marked_tree

        qds = array([(d[attribute], d['subtree-degree'])
            for n, d in tree.nodes_iter(data=True)
            if d['subtree-degree'] > 0])
        
        qs = qds[:,0]
        ds = qds[:,1]

        plt.figure()
        plt.scatter(ds, qs)
        plt.xlabel('subtree degree')
        plt.ylabel(name)
        plt.show()

    def topological_length_up(self, line_graph, e, G):
        """ Find the topological length associated to node e
        in the line graph. Topological length is defined as
        in the comment to topological_length_statistics, but
        instead of walking down the least steep path, we
        try to walk up the steepest path.
        """
        length = 0
        length_real = 0

        current_width = line_graph.node[e]['conductivity']
        current_node = e
        edges =  [e]
        while True:
            # find neighboring edges
            neighs_above = [(line_graph.node[n]['conductivity'], n)
                   for n in line_graph.neighbors(current_node)
                   if line_graph.node[n]['conductivity'] > current_width]
            
            # edges in 2-neighborhood
            #neighs_below_2 = [(line_graph.node[n]['conductivity'], n)
            #       for n in decomposer.knbrs(line_graph, current_node, 2) 
            #       if line_graph.node[n]['conductivity'] < current_width]
                        
            length += 1
            length_real += G[current_node[0]][current_node[1]]['weight']
            
            # we're at the end
            if len(neighs_above) == 0:
                break

            # use best bet from both 2 and 1 neighborhood            
            max_neighs = max(neighs_above)

            current_width, current_node = max_neighs
            edges.append(current_node)
        
        # plot edges
        #print edges
        #plt.sca(self.leaf_subplot)
        #plot.draw_leaf_raw(G, edge_list=edges, color='r')
        #raw_input()

        return length, length_real, edges

    def topological_lengths_backbone(self, G):
        """ Return the topological lengths of G obtained by first
        creating a backbone spanning tree (MST unsing 1/width as
        weights), then walking on this tree as long as possible.
        """
        # Use only largest connected component, assuming it contains
        # the leaf
        G = G.subgraph(sorted_connected_components_copy(G)[0])
                
        # We must not allow loops. This modified version of
        # MST finds a spanning tree including the largest edges,
        # so we always get the backbone structure
        G_major = minimum_spanning_tree(G, weight='conductivity')
                
        # Calculate lengths of paths following highest widths
        # for each edge in both directions
        lengths = []
        for e in G_major.edges_iter():
            #p1, pe1 = self.follow_highest_width_path(G_major, e)
            #p2, pe2 = self.follow_highest_width_path(G_major, e,
            #        reverse=True)

            p1, pe1 = self.follow_shallowest_descent_path(G_major, e)
            p2, pe2 = self.follow_shallowest_descent_path(G_major, e,
                    reverse=True)

            lengths.append(len(pe1))
            lengths.append(len(pe2))
 
        return array(lengths), G_major.number_of_edges()

    def topological_length_randomized(self, line_graph, e, G):
        """ Find the topological length associated to node e
        in the line graph. Topological length is defined as
        in the comment to topological_length_statistics.
        """
        length = 0

        current_width = line_graph.node[e]['conductivity']
        current_node = e
        edges =  [e]
        while True:
            # find neighboring edges
            neighs_below = [(line_graph.node[n]['conductivity'], n)
                   for n in line_graph.neighbors(current_node)
                   if line_graph.node[n]['conductivity'] < current_width]

            neighs_above = [(line_graph.node[n]['conductivity'], n)
                   for n in line_graph.neighbors(current_node)
                   if line_graph.node[n]['conductivity'] > current_width
                   and n not in edges]
            
            # edges in 2-neighborhood
            #neighs_below_2 = [(line_graph.node[n]['conductivity'], n)
            #       for n in decomposer.knbrs(line_graph, current_node, 2) 
            #       if line_graph.node[n]['conductivity'] < current_width]
                        
            length += 1
            
            # we're at the end
            if len(neighs_below) == 0:
                break
            
            if len(neighs_above) > 0:
                m = min(neighs_above)
                
                if 4*numpy.random.random() < 1. - current_width/m[0]:
                    max_neighs = m
                else:
                    max_neighs = max(neighs_below)
            else:
                # use best bet from both 2 and 1 neighborhood            
                max_neighs = max(neighs_below)

            current_width, current_node = max_neighs
            edges.append(current_node)
        
        # plot edges
        #print edges
        #plt.sca(self.leaf_subplot)
        #plot.draw_leaf_raw(G, edge_list=edges, color='r')
        #raw_input()

        return length, edges

    def topological_length_node(self, G, n):
        """ Find the topological length associated to edge e in
        graph G by walking from node to node, instead of edge to
        edge.
        """
        length = 0
        current_width = 1e20 # a big number...
        current_node = n
        visited = [n]
         
        while True:
            neighs = [(G[current_node][m]['conductivity'], m) 
                for m in G.neighbors(current_node) if not m in visited]

            neighs_below = [(c, n) for c, n in neighs if c < current_width]

            if len(neighs_below) == 0:
                break

            current_width, current_node = max(neighs_below)
            visited.append(current_node)
        
        return len(visited) - 1

    def topological_length_statistics(self, G):
        """ Calculate the topological length statistics of veins
        in the pruned network G by the following procedure:

        (1) Remove all nodes with degree 2, average over widths
        (2) Take any edge, follow the next smaller edge until
            there is no smaller one left.
            The lengths of these paths are our statistics
        """
        G = analyzer.edge_prune_graph(G.copy())
        line_graph = analyzer.weighted_line_graph(G)
        topol_lengths = [analyzer.topological_length(line_graph, e, G)[:2]
                for e in line_graph.nodes_iter()]

        topol_lengths_up = [self.topological_length_up(
            line_graph, e, G)[:2]
                for e in line_graph.nodes_iter()]

        #topol_lengths = [self.topological_length_node(G, n)
        #        for n in G.nodes_iter()]

        return zip(*topol_lengths), zip(*topol_lengths_up)

    def global_topological_stats(self, show_plot=True):
        """ Calculate global topological statistics of the network
        """
        # Topological lengths
        (topol_lengths, topol_lengths_real), \
                (topol_lengths_up, topol_lengths_up_real)= \
                    self.topological_length_statistics(self.prun)
        avg_topol_len = mean(topol_lengths)
        avg_topol_len_up = mean(topol_lengths_up)

        print "Average topological length:", avg_topol_len
        print "Std dev topological length:", std(topol_lengths)

        print "Average topological length (up):", avg_topol_len_up
        print "Std dev topological length (up):", std(topol_lengths_up)

        # topological lengths on backbone spanning tree
        topol_backbone, backbone_edges = \
                self.topological_lengths_backbone(self.leaf)
        avg_topol_backbone = mean(topol_backbone)
        std_topol_backbone = std(topol_backbone)

        print "Backbone average topol length:", avg_topol_backbone
        print "Backbone std dev topol length:", std_topol_backbone

        #print "Backbone average topol length/edges:", \
        #        avg_topol_backbone/backbone_edges


        if show_plot:
            plt.figure()
            plt.hist(topol_lengths, bins=max(topol_lengths), 
                    normed=True, label='descending')
            plt.hist(topol_lengths_up, bins=max(topol_lengths_up), 
                    normed=True, label='ascending', alpha=.7)
            plt.xlabel('topological length')
            plt.ylabel('probability density')
            plt.legend()

            plt.figure()
            plt.hist(topol_backbone, bins=max(topol_backbone), 
                    normed=True, label='descending')
            plt.xlabel('topological length (backbone)')
            plt.ylabel('probability density')
            plt.show()

        # Save statistics
        # Make sure stats directory exists
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        
        # Save stuff
        savetxt(self.stats_dir + 'topological_lengths.txt',
                array([topol_lengths]).T, fmt='%u')
        savetxt(self.stats_dir + 'topological_lengths_ascending.txt',
                array([topol_lengths_up]).T, fmt='%u')
        savetxt(self.stats_dir + 'topological_lengths_backbone.txt',
                array([topol_backbone]).T, fmt='%u')
    
    def subtree_degree_distribution(self, show_plot=True):
        """ Calculate and plot the subtree degree distribution
        of the decomposition tree.
        """

        degrees = [d['subtree-degree'] 
                for n, d in self.marked_tree.nodes_iter(data=True)
                if d['subtree-degree'] > 1]

        degrees_no_ext = [d['subtree-degree'] 
                for n, d in self.marked_tree_no_ext.nodes_iter(data=True)
                if d['subtree-degree'] > 1]

        def count_degrees(deg):
            degs = array(sorted(list(set(deg))))
            counts = array([(deg == d).sum() for d in degs])

            return degs, counts
        
        #counts, bins = histogram(degrees, bins=10000)
            
        x, counts = count_degrees(degrees)

        logx = log(x)
        logy = log(counts)
        
        logx = logx[logx <= 4]
        logy = logy[logx <= 4]

        z = polyfit(logx, logy, 1)

        def nonlin_fit(x, y):
            def func(x, a, b):
                return a*x**b

            return curve_fit(func, x, y, p0=[y[0], -1.0])

        print "Nonlinear LSQ fit power:", nonlin_fit(x[logx <= 4], 
                counts[logx <= 4])[0][1]

        if show_plot:
            plt.figure()
            plt.loglog(x, counts, 'o', basex=2, basey=2)
            xx = linspace(1, 2**7, 1000)
            plt.loglog(xx, exp(z[1])*xx**z[0], basex=2, basey=2)
            
            plt.text(x[0] + 0.3, counts[0] + 0.3, 
                    '$N \sim d^{{{:.2}}}$'.format(z[0]))

            plt.xlabel('log(subtree degree)')
            plt.ylabel('log(count)')
            plt.show()

        # Save statistics
        # Make sure stats directory exists
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        
        savetxt(self.stats_dir + 'subtree_degrees.txt', degrees)
        savetxt(self.stats_dir + 'subtree_degrees_no_ext.txt', 
                degrees_no_ext)
        savetxt(self.stats_dir + 'subtree_degree_dist_fit.txt', z)

    def print_help_message(self):
        print """ TreeEdit - edits analyzed hierarchical trees.
        Available commands:
        
        s - Select subtrees by clicking
        i - Select subtrees by indices
        f - Save full tree data into default files
        a - Tree asymmetry over selected subtrees
        d - Degree-area distribution
        v - Vein statistics
        g - Angle statistics
        t - Running averaging window over asymmetry tree
        l - Global topological statistics of the graph
        q - Subtree degree distribution
        w - Show 10*sqrt[n] thickest edges
        r - Show window again
        x - Exit
        """

    def edit_loop(self):

        self.print_help_message()
        while True:
            cmd = raw_input("TE> ")

            if cmd == 'x':
                return
            elif cmd == 's':
                self.select_subtree()
            elif cmd == 'i':
                self.select_subtree_indices()
            elif cmd == 'a':
                self.calculate_asymmetry()
            elif cmd == 'v':
                self.calculate_statistics()
            elif cmd == 'd':
                self.degree_area_distribution()
            elif cmd == 'g':
                self.calculate_angle_statistics()
            elif cmd == 't':
                self.window_average()
            elif cmd == 'r':
                self.init_window()
            elif cmd == 'l':
                self.global_topological_stats()
            elif cmd == 'q':
                self.subtree_degree_distribution()
            elif cmd == 'f':
                self.save_full_tree()
            elif cmd == 'w':
                self.show_sqrtn_thickest()
            else:
                print "Command not recognized."

def load_selection(fname):
    """ Tries to load the given file name as a list of subtrees
    that should be preselected.
    If the file does not exist, no subtrees will be preselected.

    Parameters:
        fname: File name of the selected tree nodes

    Returns:
        sa: List of selected nodes.
    """
    if os.path.exists(fname):
        sa = loadtxt(edt.filename + "_subtree_selection.txt",
            delimiter=", ", dtype=int)

        sa = list(array([sa]).flatten())
    else:
        sa = []

    return sa

if __name__ == '__main__':
    # Nice plots

    # Argument parser
    parser = argparse.ArgumentParser("tree_edit.py")
    parser.add_argument('INPUT', help="Input file in .pkl.bz2 format")
    parser.add_argument('-s', '--select', help="Uses selection file"
            " to pre-select nodes and automatically processes the leaf", 
            action="store_true")

    parser.add_argument('-v', '--vein-statistics', help="Saves the vein"
            " statistics for the leaf", action='store_true')
    parser.add_argument('-a', '--angle-statistics',
            help="Saves the angle statistics of the leaf",
            action='store_true')
    parser.add_argument('-e', '--no-external-loops',
            help='Always use the nesting tree without external loops',
            action='store_true')
    parser.add_argument('-d', '--segment-degree',
            help="The degree of segments to be saved separately",
            default=250, type=int)
    parser.add_argument('-f', '--save-tree', action='store_true',
            help="Save full tree data into default dir")
    parser.add_argument('-l', '--global-length-stats', action='store_true',
            help='Save topological length statistics')
    parser.add_argument('-r', '--fix-artifacts', action='store_true',
            help='Heuristic to fix artifacts from thick veins')

    args = parser.parse_args()
    
    use_ext = not args.no_external_loops
    
    if args.select or args.vein_statistics or args.angle_statistics \
            or args.save_tree or args.global_length_stats:
        edt = TreeEditor(args.INPUT, interactive=False, 
                ext=use_ext, segment_degree=args.segment_degree,
                fix_artifacts=args.fix_artifacts)

    if args.select:
        sa = load_selection(edt.filename + "_subtree_selection.txt")

        edt.selected_nodes = sa
        edt.calculate_asymmetry(show_plot=False, interactive=False)
        edt.degree_area_distribution(show_plot=False)

    if args.save_tree:
        edt.save_full_tree()
        edt.subtree_degree_distribution(show_plot=False)
   
    if args.vein_statistics:
        sa = load_selection(edt.filename + "_subtree_selection.txt")

        edt.selected_nodes = sa
        edt.calculate_statistics(show_plot=False, interactive=False)

    if args.angle_statistics:
        edt.calculate_angle_statistics(show_plot=False, interactive=False)

    if args.global_length_stats:
        edt.global_topological_stats(show_plot=False)

    if not args.select and not args.vein_statistics \
            and not args.angle_statistics and not args.save_tree \
            and not args.global_length_stats:
        edt = TreeEditor(args.INPUT, ext=use_ext,
                segment_degree=args.segment_degree,
                fix_artifacts=args.fix_artifacts)
