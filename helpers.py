#!/usr/bin/env python

import networkx as nx

def sorted_connected_components(G):
    """ Return a list of connected component subgraphs of G sorted by
    size, largest first
    """
    return sorted(nx.connected_component_subgraphs(G), 
            reverse=True, key=len)

def sorted_connected_components_copy(G):
    """ Like sorted_connected_components, but don't return subgraphs
    but copy instead
    """
    return sorted(nx.connected_components(G), 
            reverse=True, key=len)

