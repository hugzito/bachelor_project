# general imports
import networkx as nx
import pandas as pd

# michele imports
import sys
sys.path.append('../')

from michele_measures.ge_polarization.modules.ps import _ge_Q, ge
from michele_measures.network_distance import _resistance, correlation

### michele polarization method

def clean(G):
    G = G.to_undirected()

    # removing nodes missing polarity attribute
    ball = G.copy()
    for node in ball.nodes(data=True):
        if len(node[1]) == 1:
            G.remove_node(node[0])

    # include only biggest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    return G

def do_ge(G):
    G = clean(G)

    node_dict = nx.get_node_attributes(G, 'polarity')
    
    a = _ge_Q(G)

    b = ge(node_dict, {}, G, a)

    return b

### michele correlation between ideological distance and comment sentiment

# grabbing the data
def grab_network(subreddit, path):
    '''Given the name of a subreddit and the path to the folder where networks are stored, reads and returns the network'''

    filepath = path + subreddit + '.gexf'
    return nx.read_gexf(filepath)

# Transforming G to its undirected line graph equivalent G_line
# In the line graph, the new node ids will be the edges of G.
# So, if in G node 1 connected with node 2, in G_line there will be a node with id (1,2) (or (2,1))
def line_graph(G):
    '''Given a network, transforms it to its undirected line graph equivalent and returns it.'''

    G = G.to_undirected()

    G_line = nx.line_graph(G)

    # possibly need to copy edge values over

    return G_line

# Getting node values for G_line
def get_node_vals(G, G_line):
    '''Given networks G and G_line, gets the edge values from G for use as node values for G_line
       Does not assign these values in G_line, uses G_line to ensure correct formatting and returns a dictionary'''
    
    connection_formats = G_line.nodes

    node_val_dict = dict()
    for edge in list(G.edges(data=True)):
        sentiment = edge[2]['sentiment']

        # ensures correct formatting
        from_to = (edge[0], edge[1])
        to_from = (edge[1], edge[0])
        if from_to in connection_formats:
            node_val_dict[from_to] = sentiment
        elif to_from in connection_formats:
            node_val_dict[to_from] = sentiment
        
    return node_val_dict

# Getting title embedding differences
def get_ideology_distance(G, G_line):
    '''Given network G, for every pair of connected nodes calculates the difference between their title embeddigns.
       Uses G_line to ensure correct formatting of dictionary keys.
       Returns a dictionary of these differences.'''
    
    connection_formats = G_line.nodes

    ideology_distance_dict = dict()
    for edge in list(G.edges(data=True)):
        from_embed = G.nodes[edge[0]]['polarity']
        to_embed = G.nodes[edge[1]]['polarity']

        diff = abs(from_embed-to_embed)

        # ensuring correct formatting
        from_to = (edge[0], edge[1])
        to_from = (edge[1], edge[0])
        if from_to in connection_formats:
            ideology_distance_dict[from_to] = diff
        elif to_from in connection_formats:
            ideology_distance_dict[to_from] = diff

    return ideology_distance_dict

# driver function
def driver(subreddit, path):
    '''Given a subreddit name and a path to where networks are stored, uses above function'''

    G = grab_network(subreddit, path)
    G_line = line_graph(G)
    node_val_dict = get_node_vals(G, G_line)
    ideology_distance = get_ideology_distance(G, G_line)
    Q_line = _resistance(G_line)
    polarization = correlation(node_val_dict, ideology_distance, G_line, Q_line)

    return polarization


def driver_g(G):
    '''Given a subreddit name and a path to where networks are stored, uses above function'''
    G_line = line_graph(G)
    node_val_dict = get_node_vals(G, G_line)
    ideology_distance = get_ideology_distance(G, G_line)
    Q_line = _resistance(G_line)
    polarization = correlation(node_val_dict, ideology_distance, G_line, Q_line)

    return polarization