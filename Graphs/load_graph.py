import pickle

with open("review_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Graph loaded successfully")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


