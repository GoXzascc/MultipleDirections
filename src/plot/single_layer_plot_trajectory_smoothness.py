# single_layer_plot_trajectory_smoothness.py
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


def load_trajectory_smoothness_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir="assets/trajectory_smoothness"):
    """Find all models that have trajectory smoothness files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("trajectory_smoothness_") and f.endswith(".pt")]
            if files:
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_trajectory_smoothness_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item
                except:
                    models[item] = item
    return models


def find_trajectory_smoothness_files_by_model(model_name: str, base_dir="assets/trajectory_smoothness", vector_type: str = None):
    """Find all trajectory_smoothness_*.pt files for a specific model."""
    model_dir = os.path.join(base_dir, model_name)
    files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("trajectory_smoothness_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                files.append(os.path.join(model_dir, file))
    
    if not files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("trajectory_smoothness_") and file.endswith(".pt"):
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    files.append(os.path.join(model_dir, file))
    
    if not files:
        if os.path.exists(base_dir):
            for root, dirs, file_list in os.walk(base_dir):
                for file in file_list:
                    if file.startswith("trajectory_smoothness_") and file.endswith(".pt"):
                        if vector_type is not None and f"_{vector_type}" not in file:
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            obj = load_trajectory_smoothness_pt(file_path)
                            file_model = obj.get("model", "")
                            if file_model == model_name or file_model.split('/')[-1] == model_name.split('/')[-1]:
                                files.append(file_path)
                        except:
                            pass
    return sorted(files)


def plot_single_layer_comparison(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outdir: str
):
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_trajectory_smoothness_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("trajectory_smoothness_") and filename.endswith(".pt"):
                name_part = filename[len("trajectory_smoothness_"):-3]
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
        obj = load_trajectory_smoothness_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("trajectory_smoothness_") and filename.endswith(".pt"):
                name_part = filename[len("trajectory_smoothness_"):-3]
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

    # Check if we have new format data (with _raw and _removed) or old format (just cos_velocity)
    # Check first concept's first layer
    first_concept = list(all_concepts)[0]
    sample_data_source = concept_data_concept if first_concept in concept_data_concept else concept_data_random
    sample_results = sample_data_source[first_concept][middle_layer]
    has_new_format = 'cos_velocity_raw' in sample_results and 'cos_velocity_removed' in sample_results
    
    if has_new_format:
        # New format: show raw vs removed for concept vs random (1 row x 4 cols)
        fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=300)
        
        plot_configs = [
            (0, concept_data_concept, "cos_velocity_raw", "Concept Directions\n(Without Removal)"),
            (1, concept_data_concept, "cos_velocity_removed", "Concept Directions\n(Steering Removed)"),
            (2, concept_data_random, "cos_velocity_raw", "Random Directions\n(Without Removal)"),
            (3, concept_data_random, "cos_velocity_removed", "Random Directions\n(Steering Removed)"),
        ]
    else:
        # Old format: just show concept vs random (1 row x 2 cols)
        fig, axes_array = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        axes = [axes_array[0], axes_array[1], None, None]  # Pad to match loop structure
        
        plot_configs = [
            (0, concept_data_concept, "cos_velocity", "Concept Directions"),
            (1, concept_data_random, "cos_velocity", "Random Directions"),
        ]
    
    for ax_idx, data_source, velocity_key, title_text in plot_configs:
        if axes[ax_idx] is None:
            continue
            
        ax = axes[ax_idx]
        
        for concept_name in sorted(all_concepts):
            concept_color = concept_colors[concept_name]
            if concept_name in data_source and middle_layer in data_source[concept_name]:
                results = data_source[concept_name][middle_layer]
                
                # Check if the velocity key exists
                if velocity_key in results:
                    alpha = _to_np(results["alpha"])
                    mean_val = _to_np(results[velocity_key])
                    
                    # Check for std
                    std_key = f"{velocity_key}_std"
                    has_std = std_key in results
                    
                    if has_std:
                        std_val = _to_np(results[std_key])
                        mask = np.isfinite(alpha) & np.isfinite(mean_val) & np.isfinite(std_val)
                    else:
                        mask = np.isfinite(alpha) & np.isfinite(mean_val)
                    
                    if np.any(mask):
                        # Determine line style based on velocity type
                        if "removed" in velocity_key:
                            linestyle = '--'
                            linewidth = 2.5
                        else:
                            linestyle = '-'
                            linewidth = 3.0
                        
                        # Plot mean line
                        ax.plot(alpha[mask], mean_val[mask],
                               color=concept_color,
                               linewidth=linewidth,
                               linestyle=linestyle,
                               alpha=0.9)
                        
                        # Plot std as shaded area if available
                        if has_std:
                            ax.fill_between(alpha[mask],
                                          mean_val[mask] - std_val[mask],
                                          mean_val[mask] + std_val[mask],
                                          color=concept_color,
                                          alpha=0.15)
        
        # Style the axis
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$", fontweight='bold', fontsize=20)
        
        # Only show y-label on the first subplot
        if ax_idx == 0:
            ax.set_ylabel(r"$\cos((\phi(\alpha) - \phi(\alpha - \epsilon)), (\phi(\alpha) - \phi(\alpha + \epsilon))$", fontweight='bold', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(title_text, fontweight='bold', fontsize=16)
        ax.set_ylim(-1.05, 1.05)
        
        # Reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Common Legend
    legend_handles = []
    
    # Concept Colors
    for concept_name in sorted(all_concepts):
        legend_handles.append(Line2D([0], [0], color=concept_colors[concept_name], linewidth=3, linestyle='-'))
    
    # Styles
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=3.0, linestyle='-'))
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--'))
    
    legend_labels = sorted(all_concepts)
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    
    fig.legend(legend_handles, legend_labels, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_handles), 4), 
              frameon=True, fancybox=True, shadow=False, framealpha=0.95, edgecolor='black',
              fontsize=14)

    # Save
    model_short = get_model_name_for_path(model_name)
    out_filename = f"single_layer_trajectory_smoothness_{model_short}_merged.pdf"
    out_path = os.path.join(outdir, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot single layer trajectory smoothness")
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
        concept_files = find_trajectory_smoothness_files_by_model(model_name, vector_type="concept")
        random_files = find_trajectory_smoothness_files_by_model(model_name, vector_type="random")
        
        if concept_files or random_files:
            plot_single_layer_comparison(
                model_name, 
                concept_files if concept_files else [], 
                random_files if random_files else [], 
                args.outdir
            )

if __name__ == "__main__":
    main()
