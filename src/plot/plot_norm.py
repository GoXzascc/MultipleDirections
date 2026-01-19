
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def plot_concept_norms():
    # Define models to compare
    models = ["Qwen3-1.7B", "gemma-2-2b"]
    base_dir = "assets/concept_vectors"
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)

    # Find common concepts
    concepts = set()
    for model in models:
        norm_files = glob.glob(os.path.join(base_dir, model, "*_norm.json"))
        model_concepts = {os.path.basename(f).replace("_norm.json", "") for f in norm_files}
        if not concepts:
            concepts = model_concepts
        else:
            concepts &= model_concepts
    
    print(f"Found concepts: {concepts}")

    # Set style
    plt.style.use('seaborn-v0_8-paper')
    # Increase font sizes globally
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    # Define markers and colors
    markers = {"Qwen3-1.7B": "s", "gemma-2-2b": "*", "EleutherAI/pythia-160m": "^"}
    # Use a colormap for distinct colors for concepts
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    concept_colors = {concept: colors[i % len(colors)] for i, concept in enumerate(concepts)}

    plt.figure(figsize=(12, 8))
    
    # Keep track of labels for legend to avoid duplicates
    handles = []
    labels = []
    
    for concept in concepts:
        for model in models:
            file_path = os.path.join(base_dir, model, f"{concept}_norm.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    norms = json.load(f)
                
                # Create percentage-based x-axis
                num_layers = len(norms)
                x_axis = np.linspace(0, 100, num_layers)
                
                label = f"{model} - {concept}"
                line, = plt.plot(
                    x_axis, 
                    norms, 
                    marker=markers.get(model, 'o'), 
                    color=concept_colors[concept], 
                    linewidth=2, 
                    markersize=8,
                    label=label
                )
    
    plt.title("Concept Direction Norm Comparison")
    plt.xlabel("Layer Depth (%)")
    plt.ylabel("L2 Norm of Concept Vector")
    
    # Create custom legends
    # 1. Legend for Concepts (Colors)
    from matplotlib.lines import Line2D
    concept_handles = [Line2D([0], [0], color=concept_colors[c], lw=2, label=c) for c in concepts]
    first_legend = plt.legend(handles=concept_handles, title="Concepts", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.gca().add_artist(first_legend)
    
    # 2. Legend for Models (Markers)
    model_handles = [Line2D([0], [0], color='gray', marker=markers.get(m, 'o'), linestyle='None', markersize=8, label=m) for m in models]
    plt.legend(handles=model_handles, title="Models", loc='upper left', bbox_to_anchor=(1.05, 0.7))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')   
    plt.tight_layout()
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "norm_comparison_all.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_concept_norms()
