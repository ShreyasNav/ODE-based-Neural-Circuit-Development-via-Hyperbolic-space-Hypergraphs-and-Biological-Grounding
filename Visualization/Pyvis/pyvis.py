# Run this after you have the model trained

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from pyvis.network import Network
from IPython.display import display, HTML
from networkx.algorithms import community

# Define the timeline you asked for
time_steps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def get_color_map(G):
    """
    Detects communities (clusters) in the graph and assigns a color to each.
    This helps visually separate 'Sensory' from 'Motor' groups automatically.
    """
    # Use Greedy Modularity to find clusters
    communities = community.greedy_modularity_communities(G)

    # Generate colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = {}

    for i, comm in enumerate(communities):
        c = colors[i % len(colors)] # Cycle through colors
        for node in comm:
            node_colors[node] = c

    return node_colors

def visualize_evolution(time_steps, threshold=0.85):
    """
    Generates interactive plots for a list of time steps.
    High threshold (0.85) ensures we only see the 'Skeleton' of the brain.
    """
    filenames = []

    model.eval()
    with torch.no_grad():
        # Predict for ALL time steps at once for efficiency
        t_tensor = torch.tensor(time_steps).float().to(device)
        predictions = model(x_init, edge_index, edge_weight, t_tensor)

    for i, t in enumerate(time_steps):
        print(f"Rendering Time {t:.1f}...", end=" ")

        # 1. Process Probability Matrix
        adj_prob = torch.sigmoid(predictions[i]).cpu().numpy()

        # 2. Build Graph
        G = nx.DiGraph()

        # Add Edges FIRST (to determine active nodes)
        rows, cols = np.where(adj_prob > threshold)
        for r, c in zip(rows, cols):
            if r != c:
                weight = float(adj_prob[r, c])
                G.add_edge(int(r), int(c), weight=weight)

        # Add Nodes (Only those with connections to reduce clutter)
        active_nodes = list(G.nodes())

        # 3. Apply Clustering (Community Detection) for Colors
        if len(active_nodes) > 0:
            node_colors = get_color_map(G)
        else:
            node_colors = {}

        for n_id in active_nodes:
            name = id_to_name[n_id]
            color = node_colors.get(n_id, '#97C2FC')

            # Size node by its importance (Degree Centrality)
            deg = G.degree[n_id]
            size = 10 + (deg * 1.5)

            G.add_node(n_id, label=name, title=f"{name}\nDegree: {deg}", color=color, size=size)

        # 4. PyVis Visualization
        net = Network(height="600px", width="100%", notebook=True, cdn_resources='remote', directed=True, bgcolor="#222222", font_color="white")

        # Physics: specialized for "exploding" hairballs into structures
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.1,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          }
        }
        """)

        net.from_nx(G)
        fname = f"connectome_t_{t:.1f}.html"
        net.show(fname)
        filenames.append(fname)
        print(f"-> Saved {fname}")

    return filenames

# Run the Generator
generated_files = visualize_evolution(time_steps, threshold=0.85)

# Display them
print("\nLINKS TO INTERACTIVE PLOTS (Click to open in Colab):")
for f in generated_files:
    display(HTML(f"<h3><a href='{f}' target='_blank'>View Graph at Time {f.split('_')[-1][:-5]}</a></h3>"))
    # Also display the first, middle, and last inline for quick check
    if "0.0" in f or "1.5" in f or "3.0" in f:
        print(f"\n--- Preview: {f} ---")
        display(HTML(f))