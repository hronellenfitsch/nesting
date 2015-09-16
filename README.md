# nesting

This is a set of python packages to analyze leaf venation networks
generated using NET, the network extraction toolkit 
(see https://github.com/JanaLasser/network_extraction).
It was created for the publication

H. Ronellenfitsch, J. Lasser, D. Daly, E. Katifori; 
Topological phenotypes constitute a new dimension in the phenotypic space of leaf venation;
PLOS Computational Biology (under review)

If you use this software in your work, please cite the above publication!

## prerequisites

The package requires 
* python 2.7 (http://www.python.org)

additional python packages:
* numpy
* scipy
* matplotlib
* NetworkX
* blist
* cvxopt

All of the additional required packages are part of the python package index
and can be installed from http://www.pypi.python.org.

# using the decomposer and nesting tree analyzer

The package is split into two python scripts that handle hierarchical decompositon
and analysis of the nesting tree.
We also provide three test graphs to play around with in the folder /test_graphs/.

## Example
We want to construct the hierarchical decomposition tree for the leaf network
test_graphs/Dalbergia_miscolobium.gpickle.
We run

    > python decomposer.py -w test_graphs/Dalbergia_miscolobium.gpickle -s out/Dalbergia_miscolobium.pkl.bz2

This constructs the nesting tree and saves the result in the file out/Dalbergia_miscolobium.pkl.bz2
(Make sure that the folder exists!). The additional -w flag tells the decomposer to apply a workaround
for damaged leaf networks. In the case of this network, it does nothing because the network is not
damaged.

The software has additional command line flags that control its behavior and provide some
plotting of results. They are explained by running

    > python decomposer.py --help

Next, we want to analyze the nesting tree, calculate the nesting number and more.
We run

    > python analyzer.py out/Dalbergia_miscolobium.pkl.bz2 -s analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2

This performs the analysis, prints out some of the leaf traits, and saves
the results to analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2 (Again, make sure that
the folder exists!).
The set of partition asymmetries (1 - q_j as defined in the paper) 
and corresponding subtree degrees for each subtree of the
nesting tree is stored in

    analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2_data/degrees_asymmetries.txt

for analysis with other tools.

In order to do further geometric analysis, there is a graphical interface to inspect the output
of the analyzer. It is run by 

    > python tree_edit.py analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2

It allows to visually inspect the leaf network, the associated nesting tree,
visualize which parts of the nesting tree belong to which loops in the network,
and calculate and save geometric leaf traits.
The interface is designed to be largely self-explanatory.

Upon calculating leaf network traits, new folders are created, e.g.

    analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2_stats
    analyzed/Dalbergia_miscolobium_analyzed.pkl.bz2_data

These folders contain the results as text files that can be processed further.
tree_edit.py can be used in batch mode without the graphical interface to
extract data from many leaf networks. This is explained by running

    > python tree_edit.py --help

# further questions

The author of this software is Henrik Ronellenfitsch.
If you have questions about usage or find bugs, you can contact me
at

henrik [.] ronellenfitsch (at) gmail [dot] com

Unfortunately I cannot guarantee that I will be able to help but I will do my best.
