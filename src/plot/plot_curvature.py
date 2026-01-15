# plot_curvature.py
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
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


def load_curvature_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_curvature_files(search_dir="assets/linear"):
    """Find all curvature_*.pt files in the search directory."""
    curvature_files = []
    if os.path.exists(search_dir):
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.startswith("curvature_") and file.endswith(".pt"):
                    curvature_files.append(os.path.join(root, file))
    return sorted(curvature_files)


def find_all_models(base_dir="assets/linear"):
    """Find all models that have curvature files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("curvature_") and f.endswith(".pt")]
            if files:
                # Try to get model name from first file
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_curvature_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item  # Map model name to directory name
                except:
                    # If loading fails, use directory name as model name
                    models[item] = item
    
    return models


def find_curvature_files_by_model(model_name: str, base_dir="assets/linear", vector_type: str = None, remove_type: str = None):
    """Find all curvature_*.pt files for a specific model.
    
    Handles cases where model_name might be "EleutherAI/pythia-70m" but directory is "pythia-70m".
    
    Args:
        model_name: Name of the model
        base_dir: Base directory to search
        vector_type: If provided, filter files by vector type ('concept' or 'random')
        remove_type: If provided, filter files by remove type ('wo_remove' or 'w_remove')
    """
    # Try the full model name first
    model_dir = os.path.join(base_dir, model_name)
    curvature_files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("curvature_") and file.endswith(".pt"):
                # Filter by vector_type if specified
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                # Filter by remove_type if specified
                if remove_type is not None and f"_{remove_type}" not in file:
                    continue
                curvature_files.append(os.path.join(model_dir, file))
    
    # If not found and model_name contains '/', try just the last part
    if not curvature_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("curvature_") and file.endswith(".pt"):
                    # Filter by vector_type if specified
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    # Filter by remove_type if specified
                    if remove_type is not None and f"_{remove_type}" not in file:
                        continue
                    curvature_files.append(os.path.join(model_dir, file))
    
    # If still not found, search all subdirectories and match by model name in files
    if not curvature_files:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.startswith("curvature_") and file.endswith(".pt"):
                        # Filter by vector_type if specified
                        if vector_type is not None and f"_{vector_type}" not in file:
                            continue
                        # Filter by remove_type if specified
                        if remove_type is not None and f"_{remove_type}" not in file:
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            obj = load_curvature_pt(file_path)
                            file_model = obj.get("model", "")
                            # Match if model name matches (full or short)
                            if file_model == model_name or file_model.split('/')[-1] == model_name.split('/')[-1]:
                                curvature_files.append(file_path)
                        except:
                            pass
    
    return sorted(curvature_files)


def plot_all_layers_all_concepts(model_name: str, concept_files_concept: list[str], concept_files_random: list[str], outpath: str | None, remove_suffix: str = ""):
    """
    Plot kappa vs alpha for all layers and all concepts in one figure.
    Each layer gets a subplot, and each subplot shows all concepts with both concept and random vectors.
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector curvature file paths
        concept_files_random: List of random vector curvature file paths
        outpath: Output path for the plot
        remove_suffix: Suffix to add to title (e.g., "wo_remove" or "w_remove")
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_curvature_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            # Extract from filename: curvature_<concept>_concept_<remove_type>.pt
            filename = os.path.basename(file_path)
            if filename.startswith("curvature_") and filename.endswith(".pt"):
                name_part = filename[10:-3]  # Remove "curvature_" prefix and ".pt" suffix
                # Remove suffixes: _concept, _random, _wo_remove, _w_remove
                for suffix in ["_w_remove", "_wo_remove", "_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_concept[concept_name] = obj["results"]
    
    # Load random vector files
    concept_data_random = {}
    for file_path in concept_files_random:
        obj = load_curvature_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            # Extract from filename: curvature_<concept>_random_<remove_type>.pt
            filename = os.path.basename(file_path)
            if filename.startswith("curvature_") and filename.endswith(".pt"):
                name_part = filename[10:-3]  # Remove "curvature_" prefix and ".pt" suffix
                # Remove suffixes: _concept, _random, _wo_remove, _w_remove
                for suffix in ["_w_remove", "_wo_remove", "_concept", "_random"]:
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
    
    # Professional color palette (matching the project style)
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    
    # Create a color for each concept (use predefined or generate)
    concept_colors = {}
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    # Determine grid layout
    cols = min(4, num_layers)  # Max 4 columns
    rows = (num_layers + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=300)
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each layer
    for layer_idx, layer_num in enumerate(all_layers):
        ax = axes[layer_idx]
        
        # Plot all concepts for this layer (both concept and random vectors)
        for concept_name in sorted(all_concepts):
            # Plot concept vector results (solid, thick, colored)
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name]
                alpha = _to_np(results[layer_num]["alpha"])
                kappa = _to_np(results[layer_num]["kappa"])
                mask = np.isfinite(alpha) & np.isfinite(kappa)
                
                if np.any(mask):
                    ax.plot(alpha[mask], kappa[mask], 
                           color=concept_colors[concept_name],
                           label=f"{concept_name.replace('_', ' ').title()} (Concept)",
                           linewidth=3.0, marker='o', markersize=5,
                           markevery=max(1, len(alpha[mask]) // 20), alpha=0.95, linestyle='-')
            
            # Plot random vector results (dashed, thin, gray-ish)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name]
                alpha = _to_np(results[layer_num]["alpha"])
                kappa = _to_np(results[layer_num]["kappa"])
                mask = np.isfinite(alpha) & np.isfinite(kappa)
                
                if np.any(mask):
                    ax.plot(alpha[mask], kappa[mask], 
                           color='#888888',  # Use gray for all random vectors
                           label=f"{concept_name.replace('_', ' ').title()} (Random)",
                           linewidth=1.5, marker='x', markersize=4,
                           markevery=max(1, len(alpha[mask]) // 20), alpha=0.7, linestyle=':')
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Alpha", fontweight='bold', fontsize=12)
        ax.set_ylabel("Kappa (κ)", fontweight='bold', fontsize=12)
        ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=10, fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='both')
        ax.set_axisbelow(True)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    title_suffix = f" ({remove_suffix})" if remove_suffix else ""
    fig.suptitle(f"{model_name} | Kappa vs Alpha Across Layers and Concepts (Concept vs Random Vectors){title_suffix}", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)  # Make room for suptitle and legend
    
    # Collect legend handles and labels from the first subplot (avoiding duplicates)
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend at the bottom of the figure, outside plot area
    fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 5),
               frameon=True, fancybox=True, shadow=False, framealpha=0.95,
               edgecolor='black', fontsize=9, bbox_to_anchor=(0.5, 0.01))
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot kappa vs alpha for all layers and all concepts")
    ap.add_argument("--pt", type=str, default=None, help="path to curvature_*.pt (optional, will auto-discover if not provided)")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs (optional)")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, 
                   help="model name (e.g., 'EleutherAI/pythia-70m'). If not provided, will plot all detected models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    # Determine which models to plot
    if args.model:
        # Plot specific model
        models_to_plot = {args.model: get_model_name_for_path(args.model)}
    else:
        # Plot all detected models
        models_to_plot = find_all_models()
        if not models_to_plot:
            print("Error: No models found in assets/linear/")
            print("Please specify --model <model_name> or ensure curvature files exist")
            return
        print(f"Found {len(models_to_plot)} models to plot:")
        for model_name, model_dir in models_to_plot.items():
            print(f"  - {model_name} (dir: {model_dir})")
    
    # Plot each model
    for model_name, model_dir in models_to_plot.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # For each remove type (wo_remove and w_remove), create separate plots
        for remove_type in ["wo_remove", "w_remove"]:
            print(f"\n--- Processing {remove_type} ---")
            
            # Find curvature files for concept and random vectors with specific remove type
            concept_files_concept = find_curvature_files_by_model(
                model_name, vector_type="concept", remove_type=remove_type
            )
            concept_files_random = find_curvature_files_by_model(
                model_name, vector_type="random", remove_type=remove_type
            )
            
            if not concept_files_concept and not concept_files_random:
                print(f"  Warning: No {remove_type} files found for {model_name}")
                continue
            
            if concept_files_concept:
                print(f"  Found {len(concept_files_concept)} concept vector files ({remove_type}):")
                for f in concept_files_concept[:3]:  # Show first 3
                    print(f"    - {os.path.basename(f)}")
                if len(concept_files_concept) > 3:
                    print(f"    ... and {len(concept_files_concept) - 3} more")
            else:
                print(f"  Warning: No concept vector files ({remove_type}) found")
            
            if concept_files_random:
                print(f"  Found {len(concept_files_random)} random vector files ({remove_type}):")
                for f in concept_files_random[:3]:  # Show first 3
                    print(f"    - {os.path.basename(f)}")
                if len(concept_files_random) > 3:
                    print(f"    ... and {len(concept_files_random) - 3} more")
            else:
                print(f"  Warning: No random vector files ({remove_type}) found")
            
            if concept_files_concept or concept_files_random:
                # Create output filename
                model_short = get_model_name_for_path(model_name)
                output_filename = f"all_layers_all_concepts_{model_short}_{remove_type}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                
                plot_all_layers_all_concepts(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                    remove_suffix=remove_type,
                )
                
                if save:
                    print(f"  ✓ Saved: {output_filename}")
    
    if save:
        print(f"\n{'='*60}")
        print(f"All plots saved to: {args.outdir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
