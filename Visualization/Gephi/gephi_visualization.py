# Run this after you have the model trained
import networkx as nx
import numpy as np

# 1. Simulate High-Resolution Development
# We generate 50 frames from Birth (t=0) to Adult (t=3.5)
t_steps = 50
t_span = torch.linspace(0, 3.5, t_steps).to(device)

# Switch model to evaluation mode
model.eval()

with torch.no_grad():
    # NOTE: Using the signature from your uploaded notebook (4 arguments)
    # If using the improved model I gave earlier, remove 'edge_weight'
    logits = model(x_init, edge_index, edge_weight, t_span)
    probs = torch.sigmoid(logits).cpu().numpy()

# 2. Build the Graph
G = nx.DiGraph(mode="dynamic")

# Add Nodes (Neurons)
for i in range(NUM_NODES):
    # We use the neuron name as the label
    # id_to_name comes from the 'neuron_map' in your notebook
    name = list(neuron_map.keys())[list(neuron_map.values()).index(i)]
    G.add_node(i, label=name)

# 3. Add Dynamic Edges
# We scan for edges that cross the probability threshold
THRESHOLD = 0.6
count = 0

# Optimization: Only check edges that exist in the final Adult stage
# (This skips checking 50,000 empty pairs)
final_adj = probs[-1]
rows, cols = np.where(final_adj > THRESHOLD)

print(f"Analyzing {len(rows)} potential connections...")

for u, v in zip(rows, cols):
    if u == v: continue # Skip self-loops

    # Get the probability history for this specific wire
    prob_history = probs[:, u, v]

    # Find the EXACT moment it crosses the threshold
    # argmax returns the index of the first 'True' value
    is_active = prob_history > THRESHOLD

    if is_active.any():
        first_idx = np.argmax(is_active)
        start_time = t_span[first_idx].item()

        # Add the edge with a Time Interval
        # 'start': When the synapse forms
        # 'end': We set it to 4.0 (Connectome stays stable)
        # 'weight': We add the final strength so you can size edges in Gephi
        w = float(final_adj[u, v])
        G.add_edge(u, v, start=start_time, end=4.0, weight=w)
        count += 1

print(f"Exported {count} dynamic edges to 'connectome_dynamic.gexf'")

# 4. Save
nx.write_gexf(G, "connectome_dynamic.gexf")

from google.colab import files
files.download("connectome_dynamic.gexf")