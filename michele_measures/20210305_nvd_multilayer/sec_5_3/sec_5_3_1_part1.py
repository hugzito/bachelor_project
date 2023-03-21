import sys
sys.stderr.write("Setting up...\n")
import random, copy
import numpy as np
import networkx as nx
from collections import defaultdict
sys.path.append("..")
from implementation import network_distance as nd

def si_step(G, seeds, contagion_parameter):
   new_seeds = set()
   infected = set(seeds.keys())
   for n in G.nodes:
      neighbors = set(G.neighbors(n))
      if (len(neighbors & infected) / len(neighbors)) > contagion_parameter:
         new_seeds.add(n)
   return {_: 1 for _ in new_seeds}

def generate_change(G, infection):
   src = {_: 1 for _ in random.sample(G.nodes, 20)}
   trg = copy.deepcopy(src)
   for _ in range(5):
      trg = si_step(G, trg, infection)
   return src, trg

def transform_into_monolayer(src, trg):
   src_mono = {k.split('-')[0]: v for k, v in src.items()}
   trg_mono = {k.split('-')[0]: v for k, v in trg.items()}
   return src_mono, trg_mono

def flatten(G):
   G_flat = nx.Graph((e[0].split('-')[0], e[1].split('-')[0]) for e in G.edges)
   return G_flat

network = sys.argv[1]

G = nx.read_edgelist(f"../data/{network}_edges", delimiter = "\t", data = [("layer", int),])
G_flat = flatten(G)
layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))

with open(f"../data/{network}_couplings", 'r') as f:
   couplings = set([tuple(line.strip().split('\t')) for line in f])

G_emd = nd._spl_ml_preprocess(G, couplings = couplings, layer_jump_weight = layer_jump_weight)
G = nd._ge_ml_preprocess(G, couplings = couplings, layer_jump_weight = layer_jump_weight)

if network == "euair":
   infections = np.random.normal(loc = 0.15, scale = 0.05, size = 10000)
elif network == "egosm":
   infections = np.random.normal(loc = 0.1, scale = 0.05, size = 10000)
elif network == "copenhagen":
   infections = np.random.normal(loc = 0.1, scale = 0.05, size = 10000)
elif network == "aarhus":
   infections = np.random.normal(loc = 0.19, scale = 0.019, size = 10000)
elif network == "physics":
   infections = np.random.normal(loc = 0.17, scale = 0.05, size = 10000)
elif network == "ira":
   infections = np.random.normal(loc = 0.18, scale = 0.06, size = 10000)

if infections.min() < 0:
   infections -= infections.min()

i = 0

sys.stderr.write("Caching structures...\n")
Q = nd._ge_ml_Q(G, layer_jump_weight = layer_jump_weight)
v = nd._gft_ml_v(G, layer_jump_weight = layer_jump_weight)
Q_flat = nd._ge_Q(G_flat)
v_flat = nd._gft_v(G_flat)
shortest_path_lengths = nd.calculate_spl(G_emd, set(G_emd.nodes), 16, return_as_dict = True)
shortest_path_lengths_flat = nd.calculate_spl(G_flat, set(G_flat.nodes), 16, return_as_dict = True)

with open(f"cascade_{network}.csv", 'w') as f:
   f.write("beta\tmlge\tge\tmlemd\temd\tmlgft\tgft\tmlcount\tcount\n")
   for infection in infections:
      src, trg = generate_change(G, infection)
      src_mono, trg_mono = transform_into_monolayer(src, trg)
      line = []
      line.append(infection)
      line.append(nd.ge(src, trg, G, Q = Q, multilayer = True))
      line.append(nd.ge(src_mono, trg_mono, G_flat, Q = Q_flat))
      line.append(nd.emd(src, trg, G_emd, shortest_path_lengths = shortest_path_lengths, multilayer = True))
      line.append(nd.emd(src_mono, trg_mono, G_flat, shortest_path_lengths = shortest_path_lengths_flat))
      line.append(nd.gft(src, trg, G, v = v, multilayer = True))
      line.append(nd.gft(src_mono, trg_mono, G_flat, v = v_flat))
      line.append(len(trg))
      line.append(len(trg_mono))
      f.write("%s\n" % '\t'.join(str(x) for x in line))
      f.flush()
      i += 1

