# plot_direction_change.py
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def load_direction_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir="assets/directional_change"):
    """Find all models that have directional change files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("directional_change_") and f.endswith(".pt")]
            if files:
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_direction_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item
                except:
                    models[item] = item
    
    return models


def find_direction_files_by_model(model_name: str, base_dir="assets/directional_change", vector_type: str = None):
    """Find all directional_change_*.pt files for a specific model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory to search
        vector_type: If provided, filter files by vector type ('concept' or 'random')
    """
    # Try the full model name first
    model_dir = os.path.join(base_dir, model_name)
    direction_files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("directional_change_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                direction_files.append(os.path.join(model_dir, file))
    
    # If not found and model_name contains '/', try just the last part
    if not direction_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("directional_change_") and file.endswith(".pt"):
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    direction_files.append(os.path.join(model_dir, file))
    
    return sorted(direction_files)


def plot_all_layers_all_concepts(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None
):
    """
    Plot directional change vs alpha for all layers and all concepts in one figure.
    Each layer gets a subplot with 2 panels (cos_delta_v, cos_delta_h0).
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector direction file paths
        concept_files_random: List of random vector direction file paths
        outpath: Output path for the plot
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("directional_change_") and filename.endswith(".pt"):
                name_part = filename[len("directional_change_"):-3]
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
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("directional_change_") and filename.endswith(".pt"):
                name_part = filename[len("directional_change_"):-3]
                for suffix in ["_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_random[concept_name] = obj["results"]
    
    # Get all concepts (union of both)
    all_concepts = set(concept_data_concept.keys()) | set(concept_data_random.keys())
    if not all_concepts:
        raise ValueError("No concept data found")
    
    # Get all layers from the first available concept
    first_concept = list(all_concepts)[0]
    if first_concept in concept_data_concept and concept_data_concept[first_concept]:
        all_layers = sorted(concept_data_concept[first_concept].keys())
    elif first_concept in concept_data_random and concept_data_random[first_concept]:
        all_layers = sorted(concept_data_random[first_concept].keys())
    else:
        raise ValueError("No layer data found")
    num_layers = len(all_layers)
    
    # Professional color palette
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    
    # Create a color for each concept
    concept_colors = {}
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    # Direction metrics to plot
    direction_types = ['cos_delta_v', 'cos_delta_h0']
    num_cols = len(direction_types)
    
    direction_labels = {
        'cos_delta_v': r'$\cos(h(\alpha) - \alpha v, v)$',
        'cos_delta_h0': r'$\cos(h(\alpha) - \alpha v, h_0)$',
    }
    
    direction_titles = {
        'cos_delta_v': 'Residual vs Steering',
        'cos_delta_h0': 'Residual vs Original',
    }
    
    fig, axes = plt.subplots(num_layers, num_cols, figsize=(num_cols * 5, num_layers * 3.5), dpi=300)
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each layer
    for layer_idx, layer_num in enumerate(all_layers):
        for col_idx, dir_type in enumerate(direction_types):
            ax = axes[layer_idx, col_idx]
            
            # Plot all concepts for this layer
            for concept_name in sorted(all_concepts):
                # Plot concept vector results
                if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                    results = concept_data_concept[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        
                        if np.any(mask):
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=concept_colors[concept_name],
                                   label=f"{concept_name} (Concept)" if layer_idx == 0 and col_idx == 0 else "",
                                   linewidth=2.0, marker='o', markersize=2,
                                   markevery=max(1, len(alpha[mask]) // 20), alpha=0.8, linestyle='-')
                
                # Plot random vector results
                if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                    results = concept_data_random[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        
                        if np.any(mask):
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=concept_colors[concept_name],
                                   label=f"{concept_name} (Random)" if layer_idx == 0 and col_idx == 0 else "",
                                   linewidth=2.0, marker='s', markersize=2,
                                   markevery=max(1, len(alpha[mask]) // 20), alpha=0.5, linestyle='--')
            
            ax.set_xscale("log")
            ax.set_xlabel("Alpha", fontweight='bold', fontsize=10)
            ax.set_ylabel(direction_labels[dir_type], fontweight='bold', fontsize=10)
            
            # Add horizontal lines at -1, 0, 1 for reference
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.3)
            ax.axhline(y=-1, color='gray', linestyle=':', linewidth=1, alpha=0.3)
            
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"L{layer_num} | {direction_titles[dir_type]}", fontweight='bold', pad=8, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='both')
            ax.set_axisbelow(True)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('#333333')
    
    # Add legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=min(5, len(handles)), frameon=True, fancybox=True, 
                  shadow=False, framealpha=0.95, edgecolor='black', fontsize=10)
    
    # Add overall title
    fig.suptitle(f"{model_name} | Directional Change vs Alpha (Concept vs Random Vectors)", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.06)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def plot_by_concept(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None
):
    """
    Plot directional change in subplots organized by concept.
    Layout: one row per concept, columns are layers.
    Each subplot shows both cos_delta_v and cos_delta_h0.
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector direction file paths
        concept_files_random: List of random vector direction file paths
        outpath: Output path for the plot
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("directional_change_") and filename.endswith(".pt"):
                name_part = filename[len("directional_change_"):-3]
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
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("directional_change_") and filename.endswith(".pt"):
                name_part = filename[len("directional_change_"):-3]
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
    
    first_concept = list(all_concepts)[0]
    if first_concept in concept_data_concept and concept_data_concept[first_concept]:
        all_layers = sorted(concept_data_concept[first_concept].keys())
    elif first_concept in concept_data_random and concept_data_random[first_concept]:
        all_layers = sorted(concept_data_random[first_concept].keys())
    else:
        raise ValueError("No layer data found")
    
    num_layers = len(all_layers)
    num_concepts = len(all_concepts)
    sorted_concepts = sorted(all_concepts)
    
    # Colors for direction types
    dir_colors = {
        'cos_delta_v': '#2E86AB',
        'cos_delta_h0': '#A23B72', 
    }
    
    # Create figure: rows = concepts, cols = layers
    fig, axes = plt.subplots(num_concepts, num_layers, 
                             figsize=(num_layers * 4, num_concepts * 3.5), dpi=300)
    if num_concepts == 1:
        axes = axes.reshape(1, -1)
    if num_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for concept_idx, concept_name in enumerate(sorted_concepts):
        for layer_idx, layer_num in enumerate(all_layers):
            ax = axes[concept_idx, layer_idx]
            
            # Plot concept vector directions
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, color in dir_colors.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            short_name = dir_type.replace('cos_delta_', '')
                            label = f"{short_name} (C)" if concept_idx == 0 and layer_idx == 0 else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=color, label=label,
                                   linewidth=2.0, alpha=0.8, linestyle='-')
            
            # Plot random vector directions
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, color in dir_colors.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            short_name = dir_type.replace('cos_delta_', '')
                            label = f"{short_name} (R)" if concept_idx == 0 and layer_idx == 0 else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=color, label=label,
                                   linewidth=2.0, alpha=0.5, linestyle='--')
            
            ax.set_xscale("log")
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            
            if concept_idx == num_concepts - 1:
                ax.set_xlabel("Alpha", fontweight='bold', fontsize=10)
            if layer_idx == 0:
                ax.set_ylabel(f"{concept_name}\nCosine", fontweight='bold', fontsize=10)
            
            ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=8, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=4, frameon=True, fancybox=True, framealpha=0.95, 
                  edgecolor='black', fontsize=9)
    
    fig.suptitle(f"{model_name} | Directional Change by Concept", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot directional change vs alpha for all layers and concepts")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, 
                   help="model name (e.g., 'EleutherAI/pythia-70m'). If not provided, will plot all detected models")
    ap.add_argument("--by-concept", action="store_true", 
                   help="use concept-based layout (concepts as rows, layers as columns)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    # Determine which models to plot
    if args.model:
        models_to_plot = {args.model: args.model.split("/")[-1]}
    else:
        models_to_plot = find_all_models()
        if not models_to_plot:
            print("Error: No models found in assets/directional_change/")
            print("Please specify --model <model_name> or ensure directional change files exist")
            return
        print(f"Found {len(models_to_plot)} models to plot:")
        for model_name, model_dir in models_to_plot.items():
            print(f"  - {model_name} (dir: {model_dir})")
    
    # Plot each model
    for model_name, model_dir in models_to_plot.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Find direction files for concept and random vectors
        concept_files_concept = find_direction_files_by_model(model_name, vector_type="concept")
        concept_files_random = find_direction_files_by_model(model_name, vector_type="random")
        
        if not concept_files_concept and not concept_files_random:
            print(f"  Warning: No directional change files found for {model_name}")
            continue
        
        if concept_files_concept:
            print(f"  Found {len(concept_files_concept)} concept vector files:")
            for f in concept_files_concept[:3]:
                print(f"    - {os.path.basename(f)}")
            if len(concept_files_concept) > 3:
                print(f"    ... and {len(concept_files_concept) - 3} more")
        
        if concept_files_random:
            print(f"  Found {len(concept_files_random)} random vector files:")
            for f in concept_files_random[:3]:
                print(f"    - {os.path.basename(f)}")
            if len(concept_files_random) > 3:
                print(f"    ... and {len(concept_files_random) - 3} more")
        
        if concept_files_concept or concept_files_random:
            model_short = model_name.split("/")[-1]
            
            if args.by_concept:
                output_filename = f"direction_change_by_concept_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_by_concept(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                )
            else:
                output_filename = f"direction_change_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_all_layers_all_concepts(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                )
            
            if save:
                print(f"  ✓ Saved: {output_filename}")
    
    if save:
        print(f"\n{'='*60}")
        print(f"All plots saved to: {args.outdir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
