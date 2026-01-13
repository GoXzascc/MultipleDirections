# plot_velocity_direction.py
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


def load_velocity_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir="assets/velocity_direction"):
    """Find all models that have velocity direction files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("velocity_direction_") and f.endswith(".pt")]
            if files:
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_velocity_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item
                except:
                    models[item] = item
    
    return models


def find_velocity_files_by_model(model_name: str, base_dir="assets/velocity_direction", vector_type: str = None):
    """Find all velocity_direction_*.pt files for a specific model."""
    model_dir = os.path.join(base_dir, model_name)
    files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("velocity_direction_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                files.append(os.path.join(model_dir, file))
    
    # Try short name if not found
    if not files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("velocity_direction_") and file.endswith(".pt"):
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    files.append(os.path.join(model_dir, file))
    
    return sorted(files)


def plot_velocity_direction(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None
):
    """
    Plot cos(v(α), v(α-ε)) vs alpha for all layers and concepts.
    
    Layout: A rows on top (one subplot per layer), each subplot shows all concepts.
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_velocity_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("velocity_direction_") and filename.endswith(".pt"):
                name_part = filename[len("velocity_direction_"):-3]
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
        obj = load_velocity_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("velocity_direction_") and filename.endswith(".pt"):
                name_part = filename[len("velocity_direction_"):-3]
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
    
    # Get all layers
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
    concept_colors = {}
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    # Layout: max 6 columns per row
    max_cols = 6
    num_cols = min(num_layers, max_cols)
    num_rows = (num_layers + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5), dpi=300)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx, layer_num in enumerate(all_layers):
        row_idx = layer_idx // max_cols
        col_idx = layer_idx % max_cols
        ax = axes[row_idx, col_idx]
        
        for concept_name in sorted(all_concepts):
            concept_color = concept_colors[concept_name]
            
            # Plot concept vector results
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                cos_vel = _to_np(results["cos_velocity"])
                mask = np.isfinite(alpha) & np.isfinite(cos_vel)
                
                if np.any(mask):
                    show_label = (layer_idx == 0)
                    label = f"{concept_name}" if show_label else ""
                    ax.plot(alpha[mask], cos_vel[mask], 
                           color=concept_color,
                           label=label,
                           linewidth=1.5, 
                           linestyle='-',
                           alpha=0.85)
            
            # Plot random vector results (gray)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                cos_vel = _to_np(results["cos_velocity"])
                mask = np.isfinite(alpha) & np.isfinite(cos_vel)
                
                if np.any(mask):
                    ax.plot(alpha[mask], cos_vel[mask], 
                           color='#AAAAAA',
                           linewidth=0.8, 
                           linestyle='-',
                           alpha=0.5)
        
        ax.set_xscale("log")
        ax.set_xlabel("Alpha", fontweight='bold', fontsize=9)
        ax.set_ylabel(r"$\cos(\phi(\alpha), \phi(\alpha-\epsilon))$", fontweight='bold', fontsize=9)
        ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=6, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, axis='both')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#333333')
    
    # Hide unused subplots
    for layer_idx in range(num_layers, num_rows * max_cols):
        row_idx = layer_idx // max_cols
        col_idx = layer_idx % max_cols
        if row_idx < num_rows and col_idx < num_cols:
            axes[row_idx, col_idx].axis('off')
    
    fig.suptitle(f"{model_name} | Velocity Direction Consistency", 
                 fontweight='bold', fontsize=14, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    
    for concept_name in sorted(all_concepts):
        legend_handles.append(Line2D([0], [0], color=concept_colors[concept_name], linewidth=2, linestyle='-'))
        legend_labels.append(concept_name)
    
    legend_handles.append(Line2D([0], [0], color='#AAAAAA', linewidth=0.8, linestyle='-', alpha=0.5))
    legend_labels.append('Random (gray)')
    
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                  ncol=min(len(legend_handles), 8), frameon=True, fancybox=True, 
                  shadow=False, framealpha=0.95, edgecolor='black', fontsize=9)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot velocity direction consistency")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, help="model name")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    if args.model:
        models_to_plot = {args.model: args.model.split("/")[-1]}
    else:
        models_to_plot = find_all_models()
        if not models_to_plot:
            print("Error: No models found in assets/velocity_direction/")
            return
        print(f"Found {len(models_to_plot)} models to plot:")
        for model_name, model_dir in models_to_plot.items():
            print(f"  - {model_name} (dir: {model_dir})")
    
    for model_name, model_dir in models_to_plot.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        concept_files_concept = find_velocity_files_by_model(model_name, vector_type="concept")
        concept_files_random = find_velocity_files_by_model(model_name, vector_type="random")
        
        if not concept_files_concept and not concept_files_random:
            print(f"  Warning: No velocity direction files found for {model_name}")
            continue
        
        if concept_files_concept:
            print(f"  Found {len(concept_files_concept)} concept vector files")
        if concept_files_random:
            print(f"  Found {len(concept_files_random)} random vector files")
        
        model_short = model_name.split("/")[-1]
        output_filename = f"velocity_direction_{model_short}.pdf"
        output_path = None if args.show else os.path.join(args.outdir, output_filename)
        
        plot_velocity_direction(
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
