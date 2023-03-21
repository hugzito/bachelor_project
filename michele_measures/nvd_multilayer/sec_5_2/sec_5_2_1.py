import sys
import networkx as nx
sys.path.append("..")
from implementation import network_distance as nd

G_flat = nx.path_graph(2)
src_flat = {0: 1}
trg_flat = {1: 1}

with open("layercount_distances.csv", 'w') as f:
   for layers in range(2, 41):
      Gs = nx.Graph()
      for l in range(layers):
         G = nx.path_graph(2)
         nx.set_edge_attributes(G, l, "layer")
         Gs = nx.disjoint_union(Gs, G)
      couplings = set([tuple(l for l in range(0, layers * 2, 2)), tuple(l for l in range(1, layers * 2, 2))])
      src = {0: 1}
      trg = {(layers * 2) - 1: 1}
      mlge = nd.ge(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain")
      mlemd = nd.emd(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain")
      mlgft = nd.gft(src, trg, Gs, multilayer = True, couplings = couplings, coupling_style = "chain")
      ge = nd.ge(src_flat, trg_flat, G_flat)
      emd = nd.emd(src_flat, trg_flat, G_flat)
      gft = nd.gft(src_flat, trg_flat, G_flat)
      f.write(f"{layers}\t{mlge}\t{mlemd}\t{mlgft}\t{ge}\t{emd}\t{gft}\n")
