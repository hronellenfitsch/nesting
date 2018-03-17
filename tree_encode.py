#!/usr/bin/env python

"""
    tree_encode.py

    Contains methods that convert a binary tree (NetworkX DiGraph)
    into a bit string.

    Henrik Ronellenfitsch 2013
"""

#import sys
#sys.path.append('../')
#import analyzer

import networkx as nx
import matplotlib.pyplot as plt

import numpy.random
from numpy import *

import bz2
import zlib
import pylzma

import zss

import seaborn as sns

def pargsort(seq):
    """ Like numpy's argsort, but works on python lists.
    """
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key = seq.__getitem__)

def dfs_preorder_nodes_ordered(G,source=None):
    """Produce nodes in a depth-first-search pre-ordering starting at source."""
    pre=(v for u,v,d in dfs_labeled_edges_ordered(G,source=source) 
         if d['dir']=='forward')
    # chain source to beginning of pre-ordering
#    return chain([source],pre)
    return pre

def dfs_labeled_edges_ordered(G,source=None):
    """Produce edges in a depth-first-search starting at source and
    labeled by direction type (forward, reverse, nontree).
    Supportes ordered traversal of graph.
    """
    # Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    # by D. Eppstein, July 2004.
    if source is None:
        # produce edges for all components
        nodes=G
    else:
        # produce edges for components with source
        nodes=[source]

    orderfun = lambda x: G.node[x]['order']
    visited=set()
    for start in nodes:
        if start in visited:
            continue
        yield start,start,{'dir':'forward'}
        visited.add(start)

        stack = [(start, iter(sorted(G[start], key=orderfun)))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                if child in visited:
                    yield parent,child,{'dir':'nontree'}
                else:
                    yield parent,child,{'dir':'forward'}
                    visited.add(child)
                    stack.append((child, 
                        iter(sorted(G[child], key=orderfun))))
            except StopIteration:
                stack.pop()
                if stack:
                    yield stack[-1][0],parent,{'dir':'reverse'}
        yield start,start,{'dir':'reverse'}

def encode_tree(G, root=None):
    """ Encodes the binary tree G into a string containing only
    zeros and ones.
    All attributes are ignored, we only encode the pure tree structure.
    However, each node must have the 'order' attribute indicating
    whether it is the "left" or "right" subtree by being set to
    some number. The larger number indicated the right subtree.

    Parameters:
        G: NetworkX DiGraph

        root: Root node to use. If set to none, will use
            G.graph['root']

    Returns:
        s: String which corresponds to the encoded tree.
    """
    if root == None:
        root = G.graph['root']

    s = ''
    for node in dfs_preorder_nodes_ordered(G, source=root):
        if len(G.successors(node)) == 2:
            s += '1'
        else:
            s += '0'
        
    return s

def canonize_tree(G):
    """ Orders the binary tree G by bringing it into AHU canonical
    form. Effectively, we calculate the complete tree isomorphism
    invariant, and then reconstruct the canonical tree using that
    description. The algorithm loosely follows

    Aho, Hopcroft, Ullman; The Design and Analysis of Computer Algorithms,
    Addison-Wesley 1974

    The graph G is ordered in-place by adding 'order' node attributes,
    such that it can be traversed using dfs_labeled_egdes_ordered

    Parameters:
        G: NetworkX DiGraph binary tree.
        The root's node label should be in G.graph['root']

    Returns:
        cname: The canonical name (string) associated to the graph G.
    """
    cname = ''
    # Assign canonical names
    for node in nx.dfs_postorder_nodes(G, source=G.graph['root']):
        succ = G.successors(node)

        if len(succ) == 0:
            G.node[node]['canonical-name'] = '10'
        else:
            cname0 = G.node[succ[0]]['canonical-name']
            cname1 = G.node[succ[1]]['canonical-name']
            
            # Sort children's canonical names
            children_cnames = [cname0] + [cname1]
            cinds = pargsort(children_cnames)

            cname = '1' + children_cnames[cinds[0]] + \
                    children_cnames[cinds[1]] + '0'

            G.node[node]['canonical-name'] = cname
            G.node[succ[cinds[0]]]['order'] = 1
            G.node[succ[cinds[1]]]['order'] = 2
    
    G.node[node]['order'] = 0
    return cname

def decode_tree(s):
    """ Decodes the string s, which contains only zeros and ones,
    into a NetworkX DiGraph corresponding to the encoded binary tree.

    Parameters:
        s: A string containing zeros and ones, describing a binary tree.

    Returns:
        G: NetworkX DiGraph corresponding to s.
    """
    G = nx.DiGraph()
    G.add_node(0, order=0)
    G.graph['root'] = 0

    not_yet_visited = []
    max_node_id = 0
    next_node_id = 0

    # Construct the tree
    for i in xrange(len(s) - 1):
        char = s[i]

        if char == '1':
            # Create new junction
            G.add_edge(next_node_id, max_node_id + 1, order=1)
            G.add_edge(next_node_id, max_node_id + 2, order=2)

            G.add_node(max_node_id + 1, order=1)
            G.add_node(max_node_id + 2, order=2)

            next_node_id = max_node_id + 1
            not_yet_visited.append(max_node_id + 2)

            max_node_id += 2
        elif char == '0':
            # Reached leaf node. Next node to visit is on stack.
            next_node_id = not_yet_visited.pop()
    
    return G

def decode_tree_zss(s):
    """ Decodes the string s containing zeros and ones
    into a tree object compatible with the zss library
    implementing the Zhang & Shasha (1989) tree edit distance algorithm.
    The node labels are strings containing integers ordered according to
    pre-order tree traversal.

    Parameters:
        s: A string containing zeros and ones, describing a binary tree.
    
    Returns:
        G: A zss.Node object representing the tree stored in s.
    """
    G = zss.Node('0')
    
    not_yet_visited = []
    max_node_id = 0
    next_node = G

    # Construct the tree
    for i in xrange(len(s) - 1):
        char = s[i]

        if char == '1':
            left = zss.Node(str(max_node_id + 1))
            right = zss.Node(str(max_node_id + 2))

            next_node.addkid(left)
            next_node.addkid(right)

            next_node = left
            not_yet_visited.append(right)

            max_node_id += 2
        else:
            next_node = not_yet_visited.pop()

    return G

def _random_binary_tree(G, root, depth, p):
    """ Recursively constructs a random binary tree.
    """
    if depth == 0:
        return

    i = G.number_of_nodes()
    j = i + 1

    if numpy.random.random() < p:
        G.add_edge(root, i)
        G.add_edge(root, j)

        _random_binary_tree(G, i, depth - 1, p)
        _random_binary_tree(G, j, depth - 1, p)

def random_binary_tree(depth, p=.8):
    """ Constructs a random binary tree with given maximal depth
    and probability p of bifurcating.

    Parameters:
        depth: The maximum depth the tree can have (might not be reached
        due to probabilistic nature of algorithm)

        p: Probability that any node bifurcates.

    Returns:
        G: Random binary tree NetworkX DiGraph
    """
    G = nx.DiGraph()
    G.add_node(0)
    G.graph['root'] = 0

    _random_binary_tree(G, 0, depth, p)

    return G

def random_binary_tree_bottomup(degree):
    """ Constructs a random binary tree with given degree by
    randomly connecting nodes bottom up.
    This models a completely random hierarchical tree.

    Parameters:
        degree: The degree of the final tree graph.

    Returns:
        T: the final random tree graph.
    """
    T = nx.DiGraph()

    cur_nodes = range(degree)

    for i in xrange(degree - 1):
        # Connect two random nodes
        j1 = numpy.random.randint(degree - i)
        n1 = cur_nodes[j1]
        del cur_nodes[j1]

        j2 = numpy.random.randint(degree - i - 1)
        n2 = cur_nodes[j2]
        del cur_nodes[j2]
        
        new_node = degree + i
        T.add_edge(new_node, n1)
        T.add_edge(new_node, n2)
        
        cur_nodes.append(new_node)

    T.graph['root'] = new_node

    return T

def random_binary_tree_bottomup_preferential(degree):
    """ Constructs a random binary tree with given degree by
    randomly connecting nodes bottom up.
    This models a completely random hierarchical tree, but with
    preferential attachment proportional to subtree degree

    Parameters:
        degree: The degree of the final tree graph.

    Returns:
        T: the final random tree graph.
    """
    T = nx.DiGraph()

    cur_nodes = range(degree)

    for i in xrange(degree - 1):
        # Connect two random nodes 
        j1 = numpy.random.randint(degree - i)
        n1 = cur_nodes[j1]
        del cur_nodes[j1]

        j2 = numpy.random.randint(degree - i - 1)
        n2 = cur_nodes[j2]
        del cur_nodes[j2]
        
        new_node = degree + i
        T.add_edge(new_node, n1)
        T.add_edge(new_node, n2)
        
        cur_nodes.append(new_node)

    T.graph['root'] = new_node

    return T


def _random_binary_tree_uniform(root, degree_max, eps, p):
    """ Recursively constructs a random binary tree.
    Stops if number of nodes exceeds nodes
    """
    G = nx.DiGraph()
    G.add_node(0)
    G.graph['root'] = 0

    stack = [root]
    cur_degree = 1

    while stack:
        cur_root = stack.pop()

        if cur_degree > degree_max + eps:
            return G

        i = G.number_of_nodes()
        j = i + 1
        
        if numpy.random.random() < p:
            G.add_edge(cur_root, i)
            G.add_edge(cur_root, j)

            stack.append(i)
            stack.append(j)

            cur_degree += 1

    return G

def uniform_random_tree_sample(degree, size, eps):
    """ Obtains a uniformly random sample of binary
    tree of desired size and degree.
    Uses a Boltzmann sampling algorithm.

    Parameters:
        degree: The degree of trees to sample (i.e. number of leaf nodes)

        size: The sample size
        
        eps: The allowed error in degree

    Returns:
        sample: List of NetworkX digraphs with the desired sample

        degs: List of degrees of the sample
    """
    print "Sampling uniform {} random trees with degree {}, eps={}".format(
            size, degree, eps)

    sample = []
    degs = []

    while len(sample) < size:
        # Try to generate a new tree

        G = _random_binary_tree_uniform(0, degree, eps, 0.5)
        
        # Count degree in horribly inefficient way
        deg = 0
        for n, d in G.degree_iter():
            if d == 1:
                deg += 1

        if deg >= degree - eps and deg <= degree + eps:
            sample.append(G)
            degs.append(deg)

            print "Found {}/{} sample trees.".format(len(sample), size)

    return sample, degs

if __name__ == '__main__':
    pass
    # Do some tests
#    print "Uniformly random distribution of trees"
#    uniform_random_tree_sample(100, 50, 1)
#
#    print "Random binary tree..."
#    #G = random_binary_tree(15, p=0.8)
#    G = random_binary_tree_bottomup(100)
#    canon = canonize_tree(G)
#    
#    print "Encoding..."
#    s = encode_tree(G)
#
#    print "Compressing"
#
#    print "Length (string): ", len(s)
#    print "Length (bitstring):", ceil(len(s)/8.)
#    ss = bz2.compress(s)
#    sz = zlib.compress(s)
#    sl = pylzma.compress(s)
#    print "Compressed Length (bz2):", len(ss)
#    print "Compressed Length (zlib):", len(sz)
#    print "Compressed Length (lzma):", len(sl)
#
#    print "Canonical Length (string):", len(canon)
#    print "Canonical Length (bitstring):", ceil(len(canon)/8.)
#    cs = bz2.compress(canon)
#    cz = zlib.compress(canon)
#    cl = pylzma.compress(canon)
#    print "Canonical Compressed Length (bz2):", len(cs)
#    print "Canonical Compressed Length (zlib):", len(cz)
#    print "Canonical Compressed Length (lzma):", len(cl)
#    
#    print "Decoding..."
#    GG = decode_tree(s)
#    
#    gg_canon = canonize_tree(GG)
#
#    print "Isomorphic by canonical name: ", canon == gg_canon
#    #print "Isomorphic by VF2:", nx.is_isomorphic(G, GG)
#
#    # Large number of concatenated trees
#    s = ''
#    sc = ''
#    i = 0
#    for i in xrange(150):
#        try:
#            G = random_binary_tree(17, p=0.8)
#            sc += canonize_tree(G)
#            s += encode_tree(G)
#            i += 1
#        except:
#            pass
#
#    ss = bz2.compress(s)
#    scs = bz2.compress(sc)
#    
#    print i
#    print len(s)
#    print len(ss)
#
#    print len(sc)
#    print len(scs)
#    
#    sns.set_style('white')
#    def power_dist(fnc, args, kwargs, n):
#        ts = [fnc(*args, **kwargs) for i in xrange(n)]
#        ts = [t for t in ts if t.number_of_nodes() > 2**7]
#        print len(ts)
#
#        def get_power(t):
#            analyzer.mark_subtrees(t)
#
#            degs = [d['subtree-degree'] for n, d in t.nodes_iter(data=True)
#                    if d['subtree-degree'] > 1]
#
#            all_degs = array(sorted(list(set(degs))))
#
#            counts = []
#            for d in all_degs:
#                counts.append(where(degs == d)[0].shape[0])
#            
#            z = polyfit(log(all_degs)[all_degs < 2**6], 
#                    log(counts)[all_degs < 2**6], 1)
#            
#            return z[0]
#
#        print "Calculating power law..."
#        pws = [get_power(t) for t in ts]
#        print "Done."
#        return array(pws)
#    
#    # completely random
#    plt.figure()
#    ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
#    ix = [20000, 10000, 800, 200, 100, 100, 10]
#    means = []
#    stds = []
#    for p, i in zip(ps, ix):
#        pws = power_dist(random_binary_tree, (10,), {'p': p}, i)
#        #plt.hist(pws, normed=True, label='$p={}$'.format(p),
#        #        rwidth=0.9, linewidth=0, range=(-1.6, -0.8), bins=27)
#
#        means.append(pws.mean())
#        stds.append(pws.std(ddof=1))
#
#    #plt.legend()
#    #plt.xlabel('subtree degree distribution $\gamma$')
#    #plt.ylabel('probability density')
#    plt.errorbar(ps, means, yerr=stds)
#    plt.xlabel('bifurcation probability')
#    plt.ylabel('subtree degree distribution power')
#    plt.savefig('PaperPlots/rand_tree_subtree_deg_dist.svg', 
#            bbox_inches='tight')
#    # bottom up random
#    #plt.figure()
#    #for i in xrange(11, 14):
#    #    pws = power_dist(random_binary_tree_bottomup, (2**i,), {}, 100)
#    #    plt.hist(pws, normed=True, rwidth=0.9, linewidth=0,
#    #            label='deg = 2^{}'.format(i))
#    #plt.legend()
