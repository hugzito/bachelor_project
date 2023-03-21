import sys
import numpy as np
import networkx as nx
from collections import defaultdict
sys.path.append("..")
from implementation import network_distance as nd

G_flat = nx.path_graph(2)
src_flat = {0: 1}
trg_flat = {1: 1}

Gs = nx.Graph()
for l in range(2):
   G = nx.path_graph(2)
   nx.set_edge_attributes(G, l, "layer")
   Gs = nx.disjoint_union(Gs, G)

couplings = set([tuple(l for l in range(0, 2 * 2, 2)), tuple(l for l in range(1, 2 * 2, 2))])
src = {0: 1}
trg = {(2 * 2) - 1: 1}

with open("couplingweight_distances.csv", 'w') as f:
   for coupling_weight in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, np.inf):
      layer_jump_weight = defaultdict(lambda : defaultdict(lambda : coupling_weight))
      src, trg = nd._ge_ml_collapse_vectors([src, trg], Gs, couplings = couplings, coupling_style = "chain", layer_jump_weight = layer_jump_weight)
      mlge = nd.ge(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain", layer_jump_weight = layer_jump_weight)
      mlemd = nd.emd(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain", layer_jump_weight = layer_jump_weight)
      mlgft = nd.gft(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain", layer_jump_weight = layer_jump_weight)
      ge = nd.ge(src_flat, trg_flat, G_flat)
      emd = nd.emd(src_flat, trg_flat, G_flat)
      gft = nd.gft(src_flat, trg_flat, G_flat)
      f.write(f"{coupling_weight}\t{mlge}\t{mlemd}\t{mlgft}\t{ge}\t{emd}\t{gft}\n")
