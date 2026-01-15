# plot_direction_alignment.py
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def find_all_models(base_dir="assets/direction_alignment"):
    """Find all models that have direction alignment files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("direction_alignment_") and f.endswith(".pt")]
            if files:
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_direction_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item
                except:
                    models[item] = item
    
    return models


def find_direction_files_by_model(model_name: str, base_dir="assets/direction_alignment", vector_type: str = None):
    """Find all direction_alignment_*.pt files for a specific model.
    
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
            if file.startswith("direction_alignment_") and file.endswith(".pt"):
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                direction_files.append(os.path.join(model_dir, file))
    
    # If not found and model_name contains '/', try just the last part
    if not direction_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("direction_alignment_") and file.endswith(".pt"):
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
    Plot direction alignment vs alpha for all layers and all concepts in one figure.
    Each layer gets a subplot with 2 panels (cos_delta_v, cos_delta_h0) with error bars.
    
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
                # Plot concept vector results (solid, thick, colored)
                if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                    results = concept_data_concept[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        dir_std = _to_np(results[f"{dir_type}_std"]) if f"{dir_type}_std" in results else None
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        
                        if np.any(mask):
                            # Plot mean line
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=concept_colors[concept_name],
                                   label=f"{concept_name} (Concept)" if layer_idx == 0 and col_idx == 0 else "",
                                   linewidth=3.0, marker='o', markersize=4,
                                   markevery=max(1, len(alpha[mask]) // 20), alpha=0.95, linestyle='-')
                            
                            # Add shaded region for std
                            if dir_std is not None:
                                mask_std = mask & np.isfinite(dir_std)
                                if np.any(mask_std):
                                    ax.fill_between(alpha[mask_std], 
                                                   dir_val[mask_std] - dir_std[mask_std],
                                                   dir_val[mask_std] + dir_std[mask_std],
                                                   color=concept_colors[concept_name], alpha=0.15)
                
                # Plot random vector results (dotted, thin, gray)
                if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                    results = concept_data_random[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        dir_std = _to_np(results[f"{dir_type}_std"]) if f"{dir_type}_std" in results else None
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        
                        if np.any(mask):
                            # Plot mean line
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color='#888888',  # Gray for random vectors
                                   label=f"{concept_name} (Random)" if layer_idx == 0 and col_idx == 0 else "",
                                   linewidth=1.5, marker='x', markersize=3,
                                   markevery=max(1, len(alpha[mask]) // 20), alpha=0.7, linestyle=':')
                            
                            # Add shaded region for std
                            if dir_std is not None:
                                mask_std = mask & np.isfinite(dir_std)
                                if np.any(mask_std):
                                    ax.fill_between(alpha[mask_std], 
                                                   dir_val[mask_std] - dir_std[mask_std],
                                                   dir_val[mask_std] + dir_std[mask_std],
                                                   color='#888888', alpha=0.1)
            
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
    
    # Add overall title
    fig.suptitle(f"{model_name} | Direction Alignment vs Alpha (Concept vs Random Vectors)", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    # Collect legend handles first to determine space needed
    handles, labels = axes[0, 0].get_legend_handles_labels()
    num_legend_rows = (len(handles) + 4) // 5 if handles else 0
    legend_height = 0.03 + num_legend_rows * 0.02
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=legend_height + 0.06)
    
    # Add legend at the bottom (outside plot area)
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_height),
                  ncol=min(5, len(handles)), frameon=True, fancybox=True, 
                  shadow=False, framealpha=0.95, edgecolor='black', fontsize=10)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def plot_compact_grid(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None,
    cols_per_row: int = 6
):
    """
    Plot direction alignment in a compact grid layout with multiple layers per row.
    Each subplot shows both cos_delta_v and cos_delta_h0 in one panel with error bars.
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector direction file paths
        concept_files_random: List of random vector direction file paths
        outpath: Output path for the plot
        cols_per_row: Number of layers per row
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
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
    for file_path in concept_files_random:
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
    num_rows = (num_layers + cols_per_row - 1) // cols_per_row
    
    # Colors for concepts
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    concept_colors = {}
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    # Line styles: solid for cos_delta_v (steering), dashed for cos_delta_h0 (original)
    direction_styles = {
        'cos_delta_v': {'linestyle': '-', 'marker': 'o', 'markersize': 3},
        'cos_delta_h0': {'linestyle': '--', 'marker': 's', 'markersize': 3},
    }
    direction_labels = {
        'cos_delta_v': 'vs Steering',
        'cos_delta_h0': 'vs Original',
    }
    
    fig, axes = plt.subplots(num_rows, cols_per_row, 
                             figsize=(cols_per_row * 3.2, num_rows * 2.8), dpi=300,
                             squeeze=False)
    
    for layer_idx, layer_num in enumerate(all_layers):
        row = layer_idx // cols_per_row
        col = layer_idx % cols_per_row
        ax = axes[row, col]
        
        for concept_name in sorted(all_concepts):
            base_color = concept_colors[concept_name]
            
            # Plot concept vector results
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, style in direction_styles.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        dir_std = _to_np(results[f"{dir_type}_std"]) if f"{dir_type}_std" in results else None
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            label = f"{concept_name} {direction_labels[dir_type]}" if layer_idx == 0 else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=base_color, label=label,
                                   linewidth=2.0, alpha=0.9,
                                   linestyle=style['linestyle'],
                                   marker=style['marker'], markersize=style['markersize'],
                                   markevery=max(1, len(alpha[mask]) // 10))
                            
                            # Add shaded region for std (only for concept vectors to avoid clutter)
                            if dir_std is not None and dir_type == 'cos_delta_v':
                                mask_std = mask & np.isfinite(dir_std)
                                if np.any(mask_std):
                                    ax.fill_between(alpha[mask_std], 
                                                   dir_val[mask_std] - dir_std[mask_std],
                                                   dir_val[mask_std] + dir_std[mask_std],
                                                   color=base_color, alpha=0.12)
            
            # Plot random vector results (gray, thinner)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, style in direction_styles.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            label = f"Rand {direction_labels[dir_type]}" if layer_idx == 0 and concept_name == sorted(all_concepts)[0] else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color='#888888', label=label,
                                   linewidth=1.0, alpha=0.5,
                                   linestyle=':',
                                   marker='x', markersize=2,
                                   markevery=max(1, len(alpha[mask]) // 10))
        
        ax.set_xscale("log")
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"L{layer_num}", fontweight='bold', fontsize=9, pad=4)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=8)
        
        # Only show x/y labels on edge subplots
        if row == num_rows - 1:
            ax.set_xlabel("α", fontsize=9)
        if col == 0:
            ax.set_ylabel("Cosine", fontsize=9)
    
    # Hide unused subplots
    for layer_idx in range(num_layers, num_rows * cols_per_row):
        row = layer_idx // cols_per_row
        col = layer_idx % cols_per_row
        axes[row, col].set_visible(False)
    
    fig.suptitle(f"{model_name} | Direction Alignment (—: vs Steering, --: vs Original)", 
                 fontweight='bold', fontsize=12, y=0.995)
    
    # Collect legend handles first to determine space needed
    handles, labels = axes[0, 0].get_legend_handles_labels()
    num_legend_rows = (len(handles) + 5) // 6 if handles else 0
    legend_height = 0.04 + num_legend_rows * 0.025  # Dynamic height based on legend rows
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=legend_height + 0.08, hspace=0.35, wspace=0.25)
    
    # Add legend at bottom with proper spacing
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_height + 0.02),
                  ncol=min(6, len(handles)), frameon=True, fancybox=True, 
                  framealpha=0.95, edgecolor='black', fontsize=8)
    
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
    Plot direction alignment in subplots organized by concept.
    Layout: one row per concept, columns are layers.
    Both cos_delta_v and cos_delta_h0 are plotted in the same panel for each layer with error bars.
    
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
    for file_path in concept_files_random:
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
    
    # Line styles: solid for cos_delta_v, dashed for cos_delta_h0
    dir_styles = {
        'cos_delta_v': {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'markersize': 3},
        'cos_delta_h0': {'color': '#A23B72', 'linestyle': '--', 'marker': 's', 'markersize': 3},
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
            
            # Plot concept vector directions (solid, thick, colored)
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, style in dir_styles.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        dir_std = _to_np(results[f"{dir_type}_std"]) if f"{dir_type}_std" in results else None
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            short_name = dir_type.replace('cos_delta_', '')
                            label = f"{short_name} (C)" if concept_idx == 0 and layer_idx == 0 else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color=style['color'], label=label,
                                   linewidth=3.0, alpha=0.95,
                                   linestyle=style['linestyle'],
                                   marker=style['marker'], markersize=style['markersize'],
                                   markevery=max(1, len(alpha[mask]) // 20))
                            
                            # Add shaded region for std
                            if dir_std is not None:
                                mask_std = mask & np.isfinite(dir_std)
                                if np.any(mask_std):
                                    ax.fill_between(alpha[mask_std], 
                                                   dir_val[mask_std] - dir_std[mask_std],
                                                   dir_val[mask_std] + dir_std[mask_std],
                                                   color=style['color'], alpha=0.15)
            
            # Plot random vector directions (dotted, thin, gray)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for dir_type, style in dir_styles.items():
                    if dir_type in results:
                        dir_val = _to_np(results[dir_type])
                        dir_std = _to_np(results[f"{dir_type}_std"]) if f"{dir_type}_std" in results else None
                        mask = np.isfinite(alpha) & np.isfinite(dir_val)
                        if np.any(mask):
                            short_name = dir_type.replace('cos_delta_', '')
                            label = f"{short_name} (R)" if concept_idx == 0 and layer_idx == 0 else ""
                            ax.plot(alpha[mask], dir_val[mask], 
                                   color='#888888', label=label,
                                   linewidth=1.5, alpha=0.7, linestyle=':', marker='x', markersize=2,
                                   markevery=max(1, len(alpha[mask]) // 20))
                            
                            # Add shaded region for std
                            if dir_std is not None:
                                mask_std = mask & np.isfinite(dir_std)
                                if np.any(mask_std):
                                    ax.fill_between(alpha[mask_std], 
                                                   dir_val[mask_std] - dir_std[mask_std],
                                                   dir_val[mask_std] + dir_std[mask_std],
                                                   color='#888888', alpha=0.1)
            
            ax.set_xscale("log")
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            
            if concept_idx == num_concepts - 1:
                ax.set_xlabel("Alpha", fontweight='bold', fontsize=10)
            if layer_idx == 0:
                ax.set_ylabel(f"{concept_name}\nCosine", fontweight='bold', fontsize=10)
            
            ax.set_title(f"L{layer_num}", fontweight='bold', pad=8, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
    
    fig.suptitle(f"{model_name} | Direction Alignment by Concept", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    # Collect legend handles first
    handles, labels = axes[0, 0].get_legend_handles_labels()
    num_legend_rows = (len(handles) + 3) // 4 if handles else 0
    legend_height = 0.03 + num_legend_rows * 0.025
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=legend_height + 0.06)
    
    # Add legend at bottom (outside plot area)
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_height),
                  ncol=4, frameon=True, fancybox=True, framealpha=0.95, 
                  edgecolor='black', fontsize=9)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def plot_inter_token_stats(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None,
    cols_per_row: int = 6
):
    """
    Plot inter-token delta statistics in a compact grid layout.
    Shows how similar the residual directions are across different token positions.
    Visualizes: mean (solid line), mean±std (darker shaded region), and min-max range (lighter shaded region).
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector direction file paths
        concept_files_random: List of random vector direction file paths
        outpath: Output path for the plot
        cols_per_row: Number of layers per row
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
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
    for file_path in concept_files_random:
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
    num_rows = (num_layers + cols_per_row - 1) // cols_per_row
    
    # Colors for concepts
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
    }
    color_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0']
    concept_colors = {}
    for i, concept in enumerate(sorted(all_concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    fig, axes = plt.subplots(num_rows, cols_per_row, 
                             figsize=(cols_per_row * 3.2, num_rows * 2.8), dpi=300,
                             squeeze=False)
    
    for layer_idx, layer_num in enumerate(all_layers):
        row = layer_idx // cols_per_row
        col = layer_idx % cols_per_row
        ax = axes[row, col]
        
        for concept_name in sorted(all_concepts):
            base_color = concept_colors[concept_name]
            
            # Plot concept vector results
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                # Plot mean inter-token cosine similarity
                if "delta_inter_cos_mean" in results:
                    inter_mean = _to_np(results["delta_inter_cos_mean"])
                    inter_std = _to_np(results["delta_inter_cos_std"]) if "delta_inter_cos_std" in results else None
                    inter_max = _to_np(results["delta_inter_cos_max"]) if "delta_inter_cos_max" in results else None
                    inter_min = _to_np(results["delta_inter_cos_min"]) if "delta_inter_cos_min" in results else None
                    mask = np.isfinite(alpha) & np.isfinite(inter_mean)
                    
                    if np.any(mask):
                        # Add shaded region for min-max range (lightest, outermost)
                        if inter_max is not None and inter_min is not None:
                            mask_minmax = mask & np.isfinite(inter_max) & np.isfinite(inter_min)
                            if np.any(mask_minmax):
                                ax.fill_between(alpha[mask_minmax], 
                                               inter_min[mask_minmax],
                                               inter_max[mask_minmax],
                                               color=base_color, alpha=0.08, linewidth=0)
                        
                        # Add shaded region for mean±std (darker, middle layer)
                        if inter_std is not None:
                            mask_std = mask & np.isfinite(inter_std)
                            if np.any(mask_std):
                                ax.fill_between(alpha[mask_std], 
                                               inter_mean[mask_std] - inter_std[mask_std],
                                               inter_mean[mask_std] + inter_std[mask_std],
                                               color=base_color, alpha=0.2, linewidth=0)
                        
                        # Plot mean line (most prominent)
                        label = f"{concept_name} (Concept)" if layer_idx == 0 else ""
                        ax.plot(alpha[mask], inter_mean[mask], 
                               color=base_color, label=label,
                               linewidth=2.5, alpha=0.9, marker='o', markersize=3,
                               markevery=max(1, len(alpha[mask]) // 10))
            
            # Plot random vector results (gray, thinner)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                if "delta_inter_cos_mean" in results:
                    inter_mean = _to_np(results["delta_inter_cos_mean"])
                    inter_max = _to_np(results["delta_inter_cos_max"]) if "delta_inter_cos_max" in results else None
                    inter_min = _to_np(results["delta_inter_cos_min"]) if "delta_inter_cos_min" in results else None
                    mask = np.isfinite(alpha) & np.isfinite(inter_mean)
                    
                    if np.any(mask):
                        # Add shaded region for min-max range (light gray)
                        if inter_max is not None and inter_min is not None:
                            mask_minmax = mask & np.isfinite(inter_max) & np.isfinite(inter_min)
                            if np.any(mask_minmax):
                                ax.fill_between(alpha[mask_minmax], 
                                               inter_min[mask_minmax],
                                               inter_max[mask_minmax],
                                               color='#888888', alpha=0.1, linewidth=0)
                        
                        # Plot mean line
                        label = f"Random" if layer_idx == 0 and concept_name == sorted(all_concepts)[0] else ""
                        ax.plot(alpha[mask], inter_mean[mask], 
                               color='#888888', label=label,
                               linewidth=1.0, alpha=0.5, linestyle=':', marker='x', markersize=2,
                               markevery=max(1, len(alpha[mask]) // 10))
        
        ax.set_xscale("log")
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"L{layer_num}", fontweight='bold', fontsize=9, pad=4)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=8)
        
        # Only show x/y labels on edge subplots
        if row == num_rows - 1:
            ax.set_xlabel("α", fontsize=9)
        if col == 0:
            ax.set_ylabel("Inter-token cos", fontsize=9)
    
    # Hide unused subplots
    for layer_idx in range(num_layers, num_rows * cols_per_row):
        row = layer_idx // cols_per_row
        col = layer_idx % cols_per_row
        axes[row, col].set_visible(False)
    
    fig.suptitle(f"{model_name} | Inter-Token Residual Similarity (light: min-max, dark: mean±std)", 
                 fontweight='bold', fontsize=11, y=0.995)
    
    # Collect legend handles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    num_legend_rows = (len(handles) + 5) // 6 if handles else 0
    legend_height = 0.04 + num_legend_rows * 0.025
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=legend_height + 0.08, hspace=0.35, wspace=0.25)
    
    # Add legend at bottom
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_height + 0.02),
                  ncol=min(6, len(handles)), frameon=True, fancybox=True, 
                  framealpha=0.95, edgecolor='black', fontsize=8)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot direction alignment vs alpha for all layers and concepts")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, 
                   help="model name (e.g., 'EleutherAI/pythia-70m'). If not provided, will plot all detected models")
    ap.add_argument("--by-concept", action="store_true", 
                   help="use concept-based layout (concepts as rows, layers as columns)")
    ap.add_argument("--compact", action="store_true",
                   help="use compact grid layout (multiple layers per row, both metrics in each subplot)")
    ap.add_argument("--inter-token", action="store_true",
                   help="plot inter-token residual similarity statistics")
    ap.add_argument("--cols", type=int, default=6,
                   help="number of columns (layers per row) for compact layout (default: 6)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    # Determine which models to plot
    if args.model:
        models_to_plot = {args.model: get_model_name_for_path(args.model)}
    else:
        models_to_plot = find_all_models()
        if not models_to_plot:
            print("Error: No models found in assets/direction_alignment/")
            print("Please specify --model <model_name> or ensure direction alignment files exist")
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
            print(f"  Warning: No direction alignment files found for {model_name}")
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
            model_short = get_model_name_for_path(model_name)
            
            if args.inter_token:
                output_filename = f"direction_alignment_inter_token_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_inter_token_stats(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                    cols_per_row=args.cols,
                )
            elif args.compact:
                output_filename = f"direction_alignment_compact_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_compact_grid(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                    cols_per_row=args.cols,
                )
            elif args.by_concept:
                output_filename = f"direction_alignment_by_concept_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_by_concept(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                )
            else:
                output_filename = f"direction_alignment_{model_short}.pdf"
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
