import sys
sys.stderr.write("Setting up...\n")
import random, timeit, time
import numpy as np
import networkx as nx
from collections import defaultdict
sys.path.append("..")
from implementation import network_distance as nd

p = np.full((7, 7), 0.006)
np.fill_diagonal(p, 0.3)
continue_attempt = True
while continue_attempt:
   continue_attempt = False
   sys.stderr.write("Attempt...\n")
   G = nx.Graph()
   Gs = [nx.stochastic_block_model([50] * 7, p) for _ in range(3)]
   for i in range(len(Gs)):
      nx.set_edge_attributes(Gs[i], i, name = "layer")
      Gs[i].remove_nodes_from(list(nx.isolates(Gs[i])))
      G = nx.compose(G, nx.relabel_nodes(Gs[i], {n: f"{n}-{i}" for n in Gs[i].nodes}))
   coups = set([tuple([f"{n}-{l}" for l in range(3) if f"{n}-{l}" in set(G.nodes)]) for n in range(1000)])
   try:
      _ = nd._ge_ml_preprocess(G, couplings = coups)
   except ValueError:
      continue_attempt = True

sys.stderr.write(f"{len(G.nodes)} {len(G.edges)}\n")

layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))


start = time.time()
G_emd = nd._spl_ml_preprocess(G, couplings = coups, layer_jump_weight = layer_jump_weight)
emd1 = time.time() - start

start = time.time()
G = nd._ge_ml_preprocess(G, couplings = coups, layer_jump_weight = layer_jump_weight)
ge1 = time.time() - start
gft1 = ge1

sys.stderr.write("Caching structures...\n")

start = time.time()
Q = nd._ge_ml_Q(G, layer_jump_weight = layer_jump_weight)
ge2 = time.time() - start

start = time.time()
v = nd._gft_ml_v(G, layer_jump_weight = layer_jump_weight)
gft2 = time.time() - start

start = time.time()
shortest_path_lengths = nd.calculate_spl(G_emd, set(G_emd.nodes), 16, return_as_dict = True)
emd2 = time.time() - start

runtimes_ge = timeit.repeat("nd.ge({n: 1 for n in random.sample(G.nodes, 800)}, {n: 1 for n in random.sample(G.nodes, 800)}, G, Q = Q, multilayer = True, couplings = coups)", repeat = 100, number = 1, globals = globals())
runtimes_emd = timeit.repeat("nd.emd({n: 1 for n in random.sample(G.nodes, 800)}, {n: 1 for n in random.sample(G.nodes, 800)}, G, multilayer = True, shortest_path_lengths = shortest_path_lengths, couplings = coups)", repeat = 100, number = 1, globals = globals())
runtimes_gft = timeit.repeat("nd.gft({n: 1 for n in random.sample(G.nodes, 800)}, {n: 1 for n in random.sample(G.nodes, 800)}, G, multilayer = True, v = v, couplings = coups)", repeat = 100, number = 1, globals = globals())

with open("runtimes_breakdown.csv", 'w') as f:
   f.write(f"MLGE1\t{ge1+ge2+runtimes_ge[0]}\n")
   f.write(f"MLGE1k\t{ge1+ge2+sum(runtimes_ge)}\n")
   f.write(f"MLEMD1\t{emd1+emd2+runtimes_emd[0]}\n")
   f.write(f"MLEMDk\t{emd1+emd2+sum(runtimes_emd)}\n")
   f.write(f"MLGFT1\t{gft1+gft2+runtimes_gft[0]}\n")
   f.write(f"MLGFT1k\t{gft1+gft2+sum(runtimes_gft)}\n")


