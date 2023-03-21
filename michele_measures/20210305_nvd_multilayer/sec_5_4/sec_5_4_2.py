import sys
sys.stderr.write("Setting up...\n")
import random, timeit
import numpy as np
import networkx as nx
from collections import defaultdict
sys.path.append("..")
from implementation import network_distance as nd

with open(f"runtimes_layers.csv", 'w') as f:
   f.write("layers\tge_avg\tge_std\temd_avg\temd_std\tgft_avg\tgft_std\n")
   for layers in (2, 3, 4, 6, 8, 12, 24):
      sys.stderr.write(f"{layers}\n")
      groups = int(round(24 / layers))
      p = np.full((groups, groups), 0.006)
      np.fill_diagonal(p, 0.3)
      continue_attempt = True
      while continue_attempt:
         continue_attempt = False
         sys.stderr.write("Attempt...\n")
         G = nx.Graph()
         Gs = [nx.stochastic_block_model([36] * groups, p) for _ in range(layers)]
         for i in range(len(Gs)):
            nx.set_edge_attributes(Gs[i], i, name = "layer")
            Gs[i].remove_nodes_from(list(nx.isolates(Gs[i])))
            G = nx.compose(G, nx.relabel_nodes(Gs[i], {n: f"{n}-{i}" for n in Gs[i].nodes}))
         couplings = set([tuple([f"{n}-{l}" for l in range(layers) if f"{n}-{l}" in set(G.nodes)]) for n in range(len(G.nodes))])
         try:
            _ = nd._ge_ml_preprocess(G, couplings = couplings)
         except ValueError:
            continue_attempt = True
      sys.stderr.write(f"{len(G.nodes)} {len(G.edges)}\n")
      runtimes = defaultdict(list)
      for run in range(10):
         sys.stderr.write(f"{run}\n")
         src, trg = {n: 1 for n in random.sample(G.nodes, 500)}, {n: 1 for n in random.sample(G.nodes, 500)}
         runtimes["ge"].append(timeit.timeit("nd.ge(src, trg, G, multilayer = True, couplings = couplings)", number = 1, globals = globals()))
         runtimes["emd"].append(timeit.timeit("nd.emd(src, trg, G, multilayer = True, couplings = couplings)", number = 1, globals = globals()))
         runtimes["gft"].append(timeit.timeit("nd.gft(src, trg, G, multilayer = True, couplings = couplings)", number = 1, globals = globals()))
      line = []
      line.append(layers)
      line.append(np.mean(runtimes["ge"]))
      line.append(np.std(runtimes["ge"]))
      line.append(np.mean(runtimes["emd"]))
      line.append(np.std(runtimes["emd"]))
      line.append(np.mean(runtimes["gft"]))
      line.append(np.std(runtimes["gft"]))
      f.write("%s\n" % '\t'.join(str(x) for x in line))
      f.flush()
     


