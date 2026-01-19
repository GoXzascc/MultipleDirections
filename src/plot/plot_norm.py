
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
        'font.size': 24,
        'axes.titlesize': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
    })

    # Define markers and colors
    markers = {"Qwen3-1.7B": "s", "gemma-2-2b": "*", "EleutherAI/pythia-160m": "^"}
    # Use a colormap for distinct colors for concepts
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    concept_colors = {concept: colors[i % len(colors)] for i, concept in enumerate(concepts)}

    concept_renames = {
        "language_en_fr_paired": "translation",
    }

    plt.figure(figsize=(12, 8))
    
    for concept in concepts:
        for model in models:
            file_path = os.path.join(base_dir, model, f"{concept}_norm.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    norms = json.load(f)
                
                # Create percentage-based x-axis
                num_layers = len(norms)
                x_axis = np.linspace(0, 100, num_layers)
                
                label = f"{model} - {concept_renames.get(concept, concept)}"
                line, = plt.plot(
                    x_axis, 
                    norms, 
                    marker=markers.get(model, 'o'), 
                    color=concept_colors[concept], 
                    linewidth=2, 
                    markersize=8,
                    label=label
                )
    
    # plt.title("Concept Direction Norm Comparison")
    plt.xlabel("Layer Depth (%)", fontweight='bold')
    plt.ylabel("L2 Norm", fontweight='bold')
    
    # Create combined custom legend
    from matplotlib.lines import Line2D
    
    # Concept handles (Colors)
    # Add a title/header manually if needed or just list them. 
    # Let's just list them clearly.
    custom_handles = []
    
    # Add header for Concepts
    # custom_handles.append(Line2D([0], [0], color='w', label=r'$\bf{Concepts:}$'))
    for c in sorted(concepts):
        display_c = concept_renames.get(c, c)
        custom_handles.append(Line2D([0], [0], color=concept_colors[c], lw=3, label=display_c))
    
    # Add spacing or header for Models
    # custom_handles.append(Line2D([0], [0], color='w', label=r'$\bf{Models:}$'))
    for m in models:
        custom_handles.append(Line2D([0], [0], color='gray', marker=markers.get(m, 'o'), linestyle='None', markersize=10, label=m))
    
    leg = plt.legend(
        handles=custom_handles,
        loc='lower right',
        frameon=True,
        framealpha=0.9,
    )
    plt.setp(leg.get_texts(), fontweight='bold')

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
