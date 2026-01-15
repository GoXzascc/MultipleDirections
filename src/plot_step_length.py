# plot_step_length.py
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


def load_norm_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    if "results" not in obj:
        raise ValueError(f"Expected key 'results' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir="assets/step_length"):
    """Find all models that have norm decomposition files."""
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.startswith("norm_decomposition_") and f.endswith(".pt")]
            if files:
                # Try to get model name from first file
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_norm_pt(first_file)
                    file_model = obj.get("model", item)
                    models[file_model] = item  # Map model name to directory name
                except:
                    # If loading fails, use directory name as model name
                    models[item] = item
    
    return models


def find_norm_files_by_model(model_name: str, base_dir="assets/step_length", vector_type: str = None):
    """Find all norm_decomposition_*.pt files for a specific model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory to search
        vector_type: If provided, filter files by vector type ('concept' or 'random')
    """
    # Try the full model name first
    model_dir = os.path.join(base_dir, model_name)
    norm_files = []
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                # Filter by vector_type if specified
                if vector_type is not None and f"_{vector_type}" not in file:
                    continue
                norm_files.append(os.path.join(model_dir, file))
    
    # If not found and model_name contains '/', try just the last part
    if not norm_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                    # Filter by vector_type if specified
                    if vector_type is not None and f"_{vector_type}" not in file:
                        continue
                    norm_files.append(os.path.join(model_dir, file))
    
    # If still not found, search all subdirectories and match by model name in files
    if not norm_files:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.startswith("norm_decomposition_") and file.endswith(".pt"):
                        # Filter by vector_type if specified
                        if vector_type is not None and f"_{vector_type}" not in file:
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            obj = load_norm_pt(file_path)
                            file_model = obj.get("model", "")
                            # Match if model name matches (full or short)
                            if file_model == model_name or file_model.split('/')[-1] == model_name.split('/')[-1]:
                                norm_files.append(file_path)
                        except:
                            pass
    
    return sorted(norm_files)


def plot_all_layers_all_concepts(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None
):
    """
    Plot norm decomposition vs alpha for all layers and all concepts in one figure.
    Each layer gets a subplot with 3 panels (total_norm, parallel_norm, ortho_norm).
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector norm file paths
        concept_files_random: List of random vector norm file paths
        outpath: Output path for the plot
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_norm_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            # Extract from filename: norm_decomposition_<concept>_concept.pt
            filename = os.path.basename(file_path)
            if filename.startswith("norm_decomposition_") and filename.endswith(".pt"):
                name_part = filename[len("norm_decomposition_"):-3]
                # Remove suffixes: _concept, _random
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
            # Extract from filename: norm_decomposition_<concept>_random.pt
            filename = os.path.basename(file_path)
            if filename.startswith("norm_decomposition_") and filename.endswith(".pt"):
                name_part = filename[len("norm_decomposition_"):-3]
                # Remove suffixes: _concept, _random
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
    
    # Create figure: each layer has 6 subplots (3 for A, 3 for B)
    # Layout: rows = layers, cols = 6 norm types (A: total, parallel, ortho; B: total, parallel, ortho)
    norm_types_A = ['total_norm_A', 'parallel_norm_A', 'ortho_norm_A']
    norm_types_B = ['total_norm_B', 'parallel_norm_B', 'ortho_norm_B']
    # Fallback for old format (without A/B suffix)
    norm_types_old = ['total_norm', 'parallel_norm', 'ortho_norm']
    
    norm_labels = {
        'total_norm_A': r'$\|h(\alpha) - \alpha v\|$',
        'parallel_norm_A': r'$\|(\cdot)_{\parallel}\|$',
        'ortho_norm_A': r'$\|(\cdot)_{\perp}\|$',
        'total_norm_B': r'$\|(h(\alpha)-h_0) - \alpha v\|$',
        'parallel_norm_B': r'$\|(\cdot)_{\parallel}\|$',
        'ortho_norm_B': r'$\|(\cdot)_{\perp}\|$',
        # Old format fallback
        'total_norm': r'$\|h(\alpha) - \alpha v\|$',
        'parallel_norm': r'$\|(\cdot)_{\parallel}\|$',
        'ortho_norm': r'$\|(\cdot)_{\perp}\|$',
    }
    
    # Check if data uses new format (A/B) or old format
    first_concept = list(all_concepts)[0]
    first_layer = all_layers[0]
    sample_results = None
    if first_concept in concept_data_concept and first_layer in concept_data_concept[first_concept]:
        sample_results = concept_data_concept[first_concept][first_layer]
    elif first_concept in concept_data_random and first_layer in concept_data_random[first_concept]:
        sample_results = concept_data_random[first_concept][first_layer]
    
    use_new_format = sample_results is not None and 'total_norm_A' in sample_results
    
    # Line styles for norm types (solid, dashed, dotted)
    norm_type_linestyles = {
        'total': '-',
        'parallel': '--', 
        'ortho': ':',
    }
    
    if use_new_format:
        # New layout: A rows on top, B rows on bottom
        # Max columns per row (e.g., 6), wrap to multiple rows if needed
        max_cols = 6
        num_cols = min(num_layers, max_cols)
        rows_per_group = (num_layers + max_cols - 1) // max_cols  # Ceiling division
        num_rows = rows_per_group * 2  # 2 groups: A and B
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5), dpi=300)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Helper to get (row, col) for a given layer index within a group
        def get_ax_position(layer_idx, group_offset):
            row_in_group = layer_idx // max_cols
            col = layer_idx % max_cols
            return group_offset + row_in_group, col
        
        row_configs = [
            (0, norm_types_A, "A: h - αv"),  # A group starts at row 0
            (rows_per_group, norm_types_B, "B: Δh - αv"),  # B group starts after A rows
        ]
        
        for group_offset, norm_types, row_label in row_configs:
            for layer_idx, layer_num in enumerate(all_layers):
                row_idx, col_idx = get_ax_position(layer_idx, group_offset)
                ax = axes[row_idx, col_idx]
                
                # Plot all concepts and all norm types for this layer
                for concept_idx, concept_name in enumerate(sorted(all_concepts)):
                    concept_color = concept_colors[concept_name]
                    
                    # Plot concept vector results (solid lines, colored by concept)
                    if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                        results = concept_data_concept[concept_name][layer_num]
                        alpha = _to_np(results["alpha"])
                        
                        for norm_type in norm_types:
                            if norm_type in results:
                                norm_val = _to_np(results[norm_type])
                                mask = np.isfinite(alpha) & np.isfinite(norm_val)
                                
                                # Determine norm type for linestyle
                                if 'total' in norm_type:
                                    norm_key = 'total'
                                elif 'parallel' in norm_type:
                                    norm_key = 'parallel'
                                else:
                                    norm_key = 'ortho'
                                
                                if np.any(mask):
                                    # Only add label once (first subplot)
                                    show_label = (group_offset == 0 and layer_idx == 0)
                                    label = f"{concept_name} {norm_key}" if show_label else ""
                                    ax.plot(alpha[mask], norm_val[mask], 
                                           color=concept_color,
                                           label=label,
                                           linewidth=1.5, 
                                           linestyle=norm_type_linestyles[norm_key],
                                           alpha=0.85)
                    
                    # Plot random vector results (gray, thinner)
                    if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                        results = concept_data_random[concept_name][layer_num]
                        alpha = _to_np(results["alpha"])
                        
                        for norm_type in norm_types:
                            if norm_type in results:
                                norm_val = _to_np(results[norm_type])
                                mask = np.isfinite(alpha) & np.isfinite(norm_val)
                                
                                if 'total' in norm_type:
                                    norm_key = 'total'
                                elif 'parallel' in norm_type:
                                    norm_key = 'parallel'
                                else:
                                    norm_key = 'ortho'
                                
                                if np.any(mask):
                                    ax.plot(alpha[mask], norm_val[mask], 
                                           color='#AAAAAA',
                                           linewidth=0.8, 
                                           linestyle=norm_type_linestyles[norm_key],
                                           alpha=0.5)
                
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Alpha", fontweight='bold', fontsize=9)
                ax.set_ylabel("Norm", fontweight='bold', fontsize=9)
                ax.set_title(f"Layer {layer_num} | {row_label}", fontweight='bold', pad=6, fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, axis='both')
                ax.set_axisbelow(True)
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('#333333')
        
        # Hide unused subplots (when layers don't fill the grid)
        for group_offset in [0, rows_per_group]:
            for layer_idx in range(num_layers, rows_per_group * max_cols):
                row_idx, col_idx = get_ax_position(layer_idx, group_offset)
                if row_idx < num_rows and col_idx < num_cols:
                    axes[row_idx, col_idx].axis('off')
    else:
        # Old format: rows = layers, columns = norm types (total, parallel, ortho)
        # Apply same max_cols logic
        max_cols = 6
        num_cols = min(num_layers, max_cols)
        num_rows = (num_layers + max_cols - 1) // max_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5), dpi=300)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each layer (all norm types in same subplot)
        for layer_idx, layer_num in enumerate(all_layers):
            row_idx = layer_idx // max_cols
            col_idx = layer_idx % max_cols
            ax = axes[row_idx, col_idx]
            
            # Plot all concepts for this layer
            for concept_name in sorted(all_concepts):
                concept_color = concept_colors[concept_name]
                
                # Plot concept vector results (colored by concept, linestyle by norm type)
                if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                    results = concept_data_concept[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    
                    for norm_type in norm_types_old:
                        if norm_type in results:
                            norm_val = _to_np(results[norm_type])
                            mask = np.isfinite(alpha) & np.isfinite(norm_val)
                            
                            if 'total' in norm_type:
                                norm_key = 'total'
                            elif 'parallel' in norm_type:
                                norm_key = 'parallel'
                            else:
                                norm_key = 'ortho'
                            
                            if np.any(mask):
                                show_label = (layer_idx == 0)
                                label = f"{concept_name} {norm_key}" if show_label else ""
                                ax.plot(alpha[mask], norm_val[mask], 
                                       color=concept_color,
                                       label=label,
                                       linewidth=1.5, 
                                       linestyle=norm_type_linestyles[norm_key],
                                       alpha=0.85)
                
                # Plot random vector results (gray, thinner)
                if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                    results = concept_data_random[concept_name][layer_num]
                    alpha = _to_np(results["alpha"])
                    
                    for norm_type in norm_types_old:
                        if norm_type in results:
                            norm_val = _to_np(results[norm_type])
                            mask = np.isfinite(alpha) & np.isfinite(norm_val)
                            
                            if 'total' in norm_type:
                                norm_key = 'total'
                            elif 'parallel' in norm_type:
                                norm_key = 'parallel'
                            else:
                                norm_key = 'ortho'
                            
                            if np.any(mask):
                                ax.plot(alpha[mask], norm_val[mask], 
                                       color='#AAAAAA',
                                       linewidth=0.8, 
                                       linestyle=norm_type_linestyles[norm_key],
                                       alpha=0.5)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Alpha", fontweight='bold', fontsize=9)
            ax.set_ylabel("Norm", fontweight='bold', fontsize=9)
            ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=6, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, axis='both')
            ax.set_axisbelow(True)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('#333333')
        
        # Hide unused subplots
        for layer_idx in range(num_layers, num_rows * max_cols):
            row_idx = layer_idx // max_cols
            col_idx = layer_idx % max_cols
            if row_idx < num_rows and col_idx < num_cols:
                axes[row_idx, col_idx].axis('off')
    
    # Add overall title
    fig.suptitle(f"{model_name} | Norm Decomposition vs Alpha", 
                 fontweight='bold', fontsize=14, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    # Create custom legend with concept colors and linestyle explanations
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    
    # Add concept color entries
    for concept_name in sorted(all_concepts):
        legend_handles.append(Line2D([0], [0], color=concept_colors[concept_name], linewidth=2, linestyle='-'))
        legend_labels.append(concept_name)
    
    # Add linestyle entries for norm types
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-'))
    legend_labels.append('Total')
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--'))
    legend_labels.append('Parallel')
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':'))
    legend_labels.append('Ortho')
    
    # Add gray entry for random vectors
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


def plot_norm_comparison(
    model_name: str, 
    concept_files_concept: list[str], 
    concept_files_random: list[str], 
    outpath: str | None
):
    """
    Plot all norm types in subplots for each layer.
    Layout: one row per concept, columns are layers.
    Each subplot shows total, parallel, ortho norms together.
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector norm file paths
        concept_files_random: List of random vector norm file paths
        outpath: Output path for the plot
    """
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
    
    # Colors for norm types
    norm_colors = {
        'total_norm': '#2E86AB',
        'parallel_norm': '#A23B72', 
        'ortho_norm': '#06A77D',
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
            
            # Plot concept vector norms (solid, thick, colored)
            if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
                results = concept_data_concept[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for norm_type, color in norm_colors.items():
                    norm_val = _to_np(results[norm_type])
                    mask = np.isfinite(alpha) & np.isfinite(norm_val)
                    if np.any(mask):
                        label = f"{norm_type.replace('_', ' ').title()} (C)" if concept_idx == 0 and layer_idx == 0 else ""
                        ax.plot(alpha[mask], norm_val[mask], 
                               color=color, label=label,
                               linewidth=3.0, alpha=0.95, linestyle='-', marker='o', markersize=3,
                               markevery=max(1, len(alpha[mask]) // 20))
            
            # Plot random vector norms (dotted, thin, gray)
            if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
                results = concept_data_random[concept_name][layer_num]
                alpha = _to_np(results["alpha"])
                
                for norm_type, color in norm_colors.items():
                    norm_val = _to_np(results[norm_type])
                    mask = np.isfinite(alpha) & np.isfinite(norm_val)
                    if np.any(mask):
                        label = f"{norm_type.replace('_', ' ').title()} (R)" if concept_idx == 0 and layer_idx == 0 else ""
                        ax.plot(alpha[mask], norm_val[mask], 
                               color='#888888', label=label,  # Gray for random
                               linewidth=1.5, alpha=0.7, linestyle=':', marker='x', markersize=2,
                               markevery=max(1, len(alpha[mask]) // 20))
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            if concept_idx == num_concepts - 1:
                ax.set_xlabel("Alpha", fontweight='bold', fontsize=10)
            if layer_idx == 0:
                ax.set_ylabel(f"{concept_name}\nNorm", fontweight='bold', fontsize=10)
            
            ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=8, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
    
    fig.suptitle(f"{model_name} | Norm Decomposition Comparison", 
                 fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)
    
    # Add legend at bottom (outside plot area)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                  ncol=6, frameon=True, fancybox=True, framealpha=0.95, 
                  edgecolor='black', fontsize=9)
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot norm decomposition vs alpha for all layers and concepts")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, 
                   help="model name (e.g., 'EleutherAI/pythia-70m'). If not provided, will plot all detected models")
    ap.add_argument("--comparison", action="store_true", 
                   help="use comparison layout (concepts as rows, layers as columns)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    # Determine which models to plot
    if args.model:
        models_to_plot = {args.model: get_model_name_for_path(args.model)}
    else:
        models_to_plot = find_all_models()
        if not models_to_plot:
            print("Error: No models found in assets/step_length/")
            print("Please specify --model <model_name> or ensure norm decomposition files exist")
            return
        print(f"Found {len(models_to_plot)} models to plot:")
        for model_name, model_dir in models_to_plot.items():
            print(f"  - {model_name} (dir: {model_dir})")
    
    # Plot each model
    for model_name, model_dir in models_to_plot.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Find norm files for concept and random vectors
        concept_files_concept = find_norm_files_by_model(model_name, vector_type="concept")
        concept_files_random = find_norm_files_by_model(model_name, vector_type="random")
        
        if not concept_files_concept and not concept_files_random:
            print(f"  Warning: No norm decomposition files found for {model_name}")
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
            
            if args.comparison:
                output_filename = f"step_length_comparison_{model_short}.pdf"
                output_path = None if args.show else os.path.join(args.outdir, output_filename)
                plot_norm_comparison(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    output_path,
                )
            else:
                output_filename = f"step_length_{model_short}.pdf"
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
