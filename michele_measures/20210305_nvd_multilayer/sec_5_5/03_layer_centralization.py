import math
import pandas as pd
import networkx as nx

def centralization(centrality, c_type):
   n_val = len(centrality)
   if c_type == "degree":
      c_denominator = (n_val - 1) * (n_val - 2)
   elif c_type == "close":
      c_top = (n_val - 1) * (n_val - 2)
      c_bottom = (2 * n_val) - 3   
      c_denominator = c_top / c_bottom
   elif c_type == "between":
      c_denominator = 2 * (n_val - 1)
   elif c_type == "eigen":
      c_denominator = math.sqrt(2) / 2 * (n_val - 2)
   c_node_max = max(centrality.values())
   c_sorted = sorted(centrality.values(), reverse = True)
   c_numerator = 0
   for value in c_sorted:
      if c_type == "degree":
         c_numerator += (c_node_max * (n_val - 1) - value * (n_val - 1))
      else:
         c_numerator += (c_node_max - value)
   network_centrality = c_numerator / c_denominator
   if c_type == "between":
      network_centrality = network_centrality * 2
   return network_centrality

G = nx.read_edgelist("../data/euair_edges", delimiter = "\t", data = [("layer", int),])

centralizations = []
for layer in range(1, 38):
   G_layer = nx.Graph([(e[0], e[1]) for e in G.edges(data = True) if e[2]["layer"] == layer])
   degrees = dict(nx.degree_centrality(G_layer))
   close = dict(nx.closeness_centrality(G_layer))
   between = dict(nx.betweenness_centrality(G_layer))
   centralizations.append((layer, centralization(degrees, "degree"), centralization(close, "close"), centralization(between, "between")))

centralizations = pd.DataFrame(data = centralizations, columns = ("layer", "degcentr", "closecentr", "betcentr"))

df = pd.read_csv("layer_distance.csv", sep = "\t")
df = df[df["layer"] != "orig"]
df["layer"] = df["layer"].astype(int)
df = df.merge(centralizations, on = "layer")
df.to_csv("layer_centralization.csv", sep = "\t", index = False)
