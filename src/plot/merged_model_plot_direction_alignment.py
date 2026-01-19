# merged_model_plot_direction_alignment.py
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


def load_direction_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_direction_files_by_model(model_name: str, base_dir="assets/direction_alignment", vector_type: str = None):
    """Find all direction_alignment_*.pt files for a specific model."""
    model_dir = os.path.join(base_dir, model_name)
    files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("direction_alignment_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                files.append(os.path.join(model_dir, file))
    
    if not files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("direction_alignment_") and file.endswith(".pt"):
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    files.append(os.path.join(model_dir, file))
    
    return sorted(files)


def get_data_for_model(model_name: str):
    """Load concept and random data for a model."""
    concept_files = find_direction_files_by_model(model_name, vector_type="concept")
    random_files = find_direction_files_by_model(model_name, vector_type="random")
    
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files:
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("direction_alignment_") and filename.endswith(".pt"):
                name_part = filename[len("direction_alignment_"):-3]
                for suffix in ["_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_concept[concept_name] = obj["results"]
    
    # Load random vector files
    concept_data_random = {}
    for file_path in random_files:
        obj = load_direction_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            if filename.startswith("direction_alignment_") and filename.endswith(".pt"):
                name_part = filename[len("direction_alignment_"):-3]
                for suffix in ["_concept", "_random"]:
                    if name_part.endswith(suffix):
                        name_part = name_part[:-len(suffix)]
                concept_name = name_part
            else:
                concept_name = filename
        concept_data_random[concept_name] = obj["results"]
        
    return concept_data_concept, concept_data_random


def get_middle_layer(concept_data):
    """Find the middle layer index from data."""
    if not concept_data:
        return None
    
    first_concept = list(concept_data.keys())[0]
    all_layers = sorted(concept_data[first_concept].keys())
    
    if len(all_layers) == 0:
        return None
    
    middle_layer_idx = len(all_layers) // 2
    middle_layer = all_layers[middle_layer_idx]
    return middle_layer


def plot_merged_direction_alignment(outdir: str):
    # Models to plot
    models = [
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
        ("google/gemma-2-2b", "Gemma-2-2b")
    ]
    
    # Colors
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    
    # Create figure (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=300)
    
    # Metrics to plot
    metrics = [
        ('cos_delta_v', r'$\cos(v_{\mathrm{fwd}}, v)$', '$\phi$ vs Steering'),
        ('cos_delta_h0', r'$\cos(v_{\mathrm{fwd}}, h_0)$', '$\phi$ vs Original')
    ]
    
    all_concept_names = set()
    
    for model_idx, (model_path, model_short_name) in enumerate(models):
        print(f"Processing {model_short_name}...")
        concept_data, random_data = get_data_for_model(model_path)
        
        if not concept_data and not random_data:
            print(f"Warning: No data found for {model_short_name}")
            continue
            
        # Find middle layer
        middle_layer = get_middle_layer(concept_data) if concept_data else get_middle_layer(random_data)
        if middle_layer is None:
            print(f"Warning: No layers found for {model_short_name}")
            continue
            
        print(f"  Middle layer: {middle_layer}")
        
        # Collect all concepts for legend
        curr_concepts = set(concept_data.keys()) | set(random_data.keys())
        all_concept_names.update(curr_concepts)
        
        # Determine colors for this model's concepts
        concept_colors = {}
        for i, concept in enumerate(sorted(curr_concepts)):
            concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
            
        # Plot corresponding subplots
        # Qwen takes axes 0, 1
        # Gemma takes axes 2, 3
        start_ax_idx = model_idx * 2
        
        for i, (metric_key, ylabel, title_suffix) in enumerate(metrics):
            ax = axes[start_ax_idx + i]
            
            # Plot Concept Data
            for concept_name in sorted(curr_concepts):
                base_color = concept_colors[concept_name]
                
                if concept_name in concept_data and middle_layer in concept_data[concept_name]:
                    results = concept_data[concept_name][middle_layer]
                    if metric_key in results:
                        alpha = _to_np(results["alpha"])
                        val = _to_np(results[metric_key])
                        mask = np.isfinite(alpha) & np.isfinite(val)
                        
                        if np.any(mask):
                            ax.plot(alpha[mask], val[mask],
                                   color=base_color,
                                   linewidth=3.0,
                                   linestyle='-',
                                   alpha=0.9)
            
            # Plot Random Data
            for concept_name in sorted(curr_concepts):
                if concept_name in random_data and middle_layer in random_data[concept_name]:
                    results = random_data[concept_name][middle_layer]
                    if metric_key in results:
                        alpha = _to_np(results["alpha"])
                        val = _to_np(results[metric_key])
                        mask = np.isfinite(alpha) & np.isfinite(val)
                        
                        if np.any(mask):
                            ax.plot(alpha[mask], val[mask],
                                   color='#888888',
                                   linewidth=1.5,
                                   linestyle='--',
                                   alpha=0.6)

            # Styling
            ax.set_xscale("log")
            
            # Only set Y label for the first plot of each model? 
            # Or only for the very first plot? 
            # User usually prefers Y label on leftmost or when metric changes.
            # Here we have 2 metrics.
            # Col 0: Metric 1 (Qwen) -> Needs Y label
            # Col 1: Metric 2 (Qwen) -> Needs Y label (different unit/meaning?) Both are Cosine.
            # Col 2: Metric 1 (Gemma) -> Y label redundant if same row? But convenient.
            # Let's put Y label on Col 0 and Col 2 (start of each model? No, start of each metric type?)
            # Actually, standard is Y label on Col 0. But since Metric 1 and 2 are different quantities, 
            # visual clarity might benefit from labels. 
            # However, typically "Cosine Similarity" is the shared Y unit.
            # Let's add Y label for Col 0 and Col 1 (since they are different metrics), 
            # and omit for Col 2 and Col 3 (since they repeat metrics).
            
            if start_ax_idx + i == 0:
                ax.set_ylabel(metrics[0][1], fontweight='bold', fontsize=20)
            elif start_ax_idx + i == 1:
                ax.set_ylabel(metrics[1][1], fontweight='bold', fontsize=20)
            
            ax.set_xlabel(r"$\alpha$", fontweight='bold', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # Title: Model Name + Metric Description
            full_title = f"{model_short_name}\n{title_suffix}"
            ax.set_title(full_title, fontweight='bold', fontsize=16)
            
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
            ax.axhline(y=-1, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
            ax.set_axisbelow(True)

    # Legend
    legend_handles = []
    # Assign global colors for legend
    global_concept_colors = {}
    for i, concept in enumerate(sorted(all_concept_names)):
        global_concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
        legend_handles.append(Line2D([0], [0], color=global_concept_colors[concept], linewidth=3, linestyle='-'))
    
    legend_handles.append(Line2D([0], [0], color='#888888', linewidth=1.5, linestyle='--'))
    legend_labels = sorted(all_concept_names) + ['Random']
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    fig.legend(legend_handles, legend_labels, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_handles), 6), 
              frameon=True, fancybox=True, shadow=False, framealpha=0.95, edgecolor='black',
              fontsize=14)

    # Save
    out_filename = "merged_model_direction_alignment.pdf"
    out_path = os.path.join(outdir, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    plot_merged_direction_alignment(args.outdir)


if __name__ == "__main__":
    main()
