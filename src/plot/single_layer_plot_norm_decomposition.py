import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_model_name_for_path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_norm_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir="assets/norm_decomposition"):
    """Find all models that have norm decomposition files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("norm_decomposition_") and f.endswith(".pt")]
            if files:
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_norm_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item  # Map model name to directory name
                except:
                    models[item] = item
    return models


def find_norm_files_by_model(model_name: str, base_dir="assets/norm_decomposition", vector_type: str = None):
    """Find all norm_decomposition_*.pt files for a specific model."""
    model_dir = os.path.join(base_dir, model_name)
    norm_files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                norm_files.append(os.path.join(model_dir, file))
    
    if not norm_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    norm_files.append(os.path.join(model_dir, file))
    
    if not norm_files:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                        if vector_type is not None and f"_{vector_type}" not in file:
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            obj = load_norm_pt(file_path)
                            file_model = obj.get("model", "")
                            if file_model == model_name or file_model.split('/')[-1] == model_name.split('/')[-1]:
                                norm_files.append(file_path)
                        except:
                            pass
    return sorted(norm_files)


def plot_single_layer_comparison(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outdir: str
):
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_norm_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("norm_decomposition_") and filename.endswith(".pt"):
                name_part = filename[len("norm_decomposition_"):-3]
                for suffix in ["_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_concept[concept_name] = obj["results"]
    
    # Load random vector files
    concept_data_random = {}
    for file_path in concept_files_random:
        obj = load_norm_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("norm_decomposition_") and filename.endswith(".pt"):
                name_part = filename[len("norm_decomposition_"):-3]
                for suffix in ["_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_random[concept_name] = obj["results"]
    
    all_concepts = set(concept_data_concept.keys()) | set(concept_data_random.keys())
    if not all_concepts:
        raise ValueError("No concept data found")
    
    # Find middle layer
    first_concept = list(all_concepts)[0]
    if first_concept in concept_data_concept and concept_data_concept[first_concept]:
        all_layers = sorted(concept_data_concept[first_concept].keys())
    elif first_concept in concept_data_random and concept_data_random[first_concept]:
        all_layers = sorted(concept_data_random[first_concept].keys())
    else:
        raise ValueError("No layer data found")
    
    if len(all_layers) == 0:
        raise ValueError("No layers found")
    
    middle_layer_idx = len(all_layers) // 2
    middle_layer = all_layers[middle_layer_idx]
    print(f"Plotting for middle layer: {middle_layer} (index {middle_layer_idx})")

    # Colors
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    concept_colors = {}
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])

    # Decompositions to plot
    decompositions = [
        ("A", ['parallel_norm_A', 'ortho_norm_A'], r"Decomposition A: $h(\alpha) - \alpha v$"),
        ("B", ['parallel_norm_B', 'ortho_norm_B'], r"Decomposition B: $(h(\alpha)-h_0) - \alpha v$"),
    ]

    for decomp_name, norm_types, decomp_label in decompositions:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        
        # Plot Logic
        # ax[0] = Concept, ax[1] = Random
        
        data_sources = [
            (axes[0], concept_data_concept, "Concept Directions"),
            (axes[1], concept_data_random, "Random Directions")
        ]
        
        for ax, data_source, title in data_sources:
            for concept_name in sorted(all_concepts):
                concept_color = concept_colors[concept_name]
                if concept_name in data_source and middle_layer in data_source[concept_name]:
                    results = data_source[concept_name][middle_layer]
                    alpha = _to_np(results["alpha"])
                    
                    for norm_type in norm_types:
                        if norm_type in results:
                            norm_val = _to_np(results[norm_type])
                            mask = np.isfinite(alpha) & np.isfinite(norm_val)
                            
                            # Calculate markevery to avoid clutter
                            n_points = len(alpha[mask])
                            mark_freq = max(1, n_points // 10)  # ~10 markers per line

                            if 'parallel' in norm_type:
                                linestyle = '-'
                                marker = 'o'
                                linew = 2.0
                                alp = 0.9
                            else:
                                linestyle = '-.'  # Changed to dash-dot
                                marker = '^'      # Changed to triangle
                                linew = 1.8
                                alp = 0.8
                            
                            if np.any(mask):
                                # No per-line label, using custom legend
                                ax.plot(alpha[mask], norm_val[mask], 
                                       color=concept_color,
                                       linewidth=linew, 
                                       linestyle=linestyle,
                                       marker=marker,
                                       markersize=6,
                                       markevery=mark_freq,
                                       alpha=alp)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Alpha", fontweight='bold')
            ax.set_ylabel("Norm", fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
            
            # Remove top/right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Common Legend
        legend_handles = []
        legend_labels = []
        
        # Concept Colors
        for concept_name in sorted(all_concepts):
            legend_handles.append(Line2D([0], [0], color=concept_colors[concept_name], linewidth=2, linestyle='-'))
            legend_labels.append(concept_name)
        
        # Styles
        legend_handles.append(Line2D([0], [0], color='gray', linewidth=2.0, linestyle='-', marker='o', markersize=6))
        legend_labels.append('Parallel Norm')
        legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.8, linestyle='-.', marker='^', markersize=6))
        legend_labels.append('Ortho Norm')
        
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.05), ncol=min(len(legend_handles), 6), 
                  frameon=True, fancybox=True, shadow=False, framealpha=0.95, edgecolor='black')

        fig.suptitle(f"{model_name} | Layer {middle_layer} | {decomp_label}", fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save
        model_short = get_model_name_for_path(model_name)
        out_filename = f"single_layer_norm_decomposition_{model_short}_{decomp_name}.pdf"
        out_path = os.path.join(outdir, out_filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot single layer norm decomposition")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--model", type=str, default=None, 
                   help="model name (e.g., 'EleutherAI/pythia-70m').")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine which models to plot
    if args.model:
        models_to_plot = {args.model: get_model_name_for_path(args.model)}
    else:
        models_to_plot = find_all_models()
        # Filter Default
        targets = ['gemma-2-2b', 'Qwen3-1.7B']
        filtered_models = {}
        for m_name, m_dir in models_to_plot.items():
            if any(t in m_name for t in targets) or any(t in m_dir for t in targets):
                filtered_models[m_name] = m_dir
        if filtered_models:
            models_to_plot = filtered_models

    if not models_to_plot:
        print("No matching models found.")
        return

    for model_name, model_dir in models_to_plot.items():
        print(f"\nProcessing {model_name}...")
        concept_files = find_norm_files_by_model(model_name, vector_type="concept")
        random_files = find_norm_files_by_model(model_name, vector_type="random")
        
        if concept_files or random_files:
            plot_single_layer_comparison(
                model_name, 
                concept_files if concept_files else [], 
                random_files if random_files else [], 
                args.outdir
            )

if __name__ == "__main__":
    main()
