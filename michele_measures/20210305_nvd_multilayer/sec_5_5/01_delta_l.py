import sys, glob
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
sys.path.append("..")
from implementation import network_distance as nd

def pairwise_distances(G_local, countries, vectors, couplings, layer_jump_weight, label):
   vectors = nd._ge_ml_collapse_vectors(vectors, G_local, couplings = couplings, layer_jump_weight = layer_jump_weight)
   G_local = nd._ge_ml_preprocess(G_local, couplings = couplings, layer_jump_weight = layer_jump_weight)
   Q = nd._ge_ml_Q(G_local, layer_jump_weight = layer_jump_weight)
   dists = []
   for i in range(len(countries)):
      for j in range(len(countries)):
         if i != j:
            dists.append((countries[i], countries[j], label, nd.ge(vectors[i], vectors[j], G_local, Q = Q, multilayer = True), len(G_local.edges), len(G_local.nodes)))
   return dists

layer = sys.argv[1]
G = nx.read_edgelist("../data/euair_edges", delimiter = "\t", data = [("layer", int),])

countries = []
vectors = []
with open("euair_vectors", 'r') as f:
   for line in f:
      fields = line.strip().split('\t')
      countries.append(fields[0])
      vectors.append({n: 1 for n in fields[1].strip().split(',') if n.split('-')[1] != layer})

with open("../data/euair_couplings", 'r') as f:
   couplings = set([tuple([x for x in line.strip().split('\t') if x.split('-')[1] != layer]) for line in f])

layer_jump_weight = defaultdict(lambda : defaultdict(lambda : np.inf))

sys.stderr.write("%s...\n" % layer)
G_crippled = G.copy()
if layer != "orig":
   G_crippled.remove_edges_from([(e[0], e[1]) for e in G.edges(data = True) if e[2]["layer"] == int(layer)])
   G_crippled.remove_nodes_from(list(nx.isolates(G_crippled)))

dists = pd.DataFrame(
   columns = ("c1", "c2", "layer", "dist", "edge_count", "node_count"),
   data = pairwise_distances(G_crippled, countries, vectors.copy(), couplings, layer_jump_weight, layer)
)

dists.to_csv(f"{layer}.csv", sep = "\t", index = False)
