# plot_trajectory_smoothness.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    setup_publication_style,
    to_np,
    assign_colors_to_concepts,
    load_pt_file,
    find_all_models,
    find_files_by_model,
    extract_concept_name_from_filename,
    get_model_name_for_path
)

# Set publication-quality style
setup_publication_style()


def load_trajectory_smoothness_pt(path: str):
    """Load a trajectory_smoothness .pt file."""
    return load_pt_file(path, expected_key="results")


def find_trajectory_smoothness_files_by_model(model_name: str, base_dir="assets/trajectory_smoothness", 
                                               vector_type: str = None):
    """Find all trajectory_smoothness_*.pt files for a specific model."""
    return find_files_by_model(
        model_name=model_name,
        base_dir=base_dir,
        file_prefix="trajectory_smoothness_",
        vector_type=vector_type,
        suffix_filter=None
    )


def plot_single_layer_on_axis(ax, layer_num: int, concept_data_concept: dict, concept_data_random: dict,
                              all_concepts: set, concept_colors: dict, show_legend: bool = False):
    """
    Plot cos_velocity metrics for a single layer.
    Shows two versions: raw (without steering removed) and removed (with steering removed).
    Shows mean as line and std as shaded area.
    
    Args:
        ax: Matplotlib axis object
        layer_num: Layer number to plot
        concept_data_concept: Dictionary of concept data
        concept_data_random: Dictionary of random data
        all_concepts: Set of all concept names
        concept_colors: Dictionary mapping concept names to colors
        show_legend: Whether to show legend
    """
    for concept_name in sorted(all_concepts):
        concept_color = concept_colors[concept_name]
        
        # Plot concept vector results
        if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
            results = concept_data_concept[concept_name][layer_num]
            
            # Plot raw velocity (without removing steering) - solid line
            if "cos_velocity_raw" in results and "cos_velocity_raw_std" in results:
                alpha = to_np(results["alpha"])
                mean_val = to_np(results["cos_velocity_raw"])
                std_val = to_np(results["cos_velocity_raw_std"])
                mask = np.isfinite(alpha) & np.isfinite(mean_val) & np.isfinite(std_val)
                
                if np.any(mask):
                    # Plot mean line
                    ax.plot(alpha[mask], mean_val[mask],
                           color=concept_color,
                           linewidth=2.0,
                           linestyle='-',
                           alpha=0.9,
                           label=f"{concept_name} (raw)" if show_legend else "")
                    
                    # Plot std as shaded area
                    ax.fill_between(alpha[mask],
                                   mean_val[mask] - std_val[mask],
                                   mean_val[mask] + std_val[mask],
                                   color=concept_color,
                                   alpha=0.15)
            
            # Plot velocity with steering removed - dashed line
            if "cos_velocity_removed" in results and "cos_velocity_removed_std" in results:
                alpha = to_np(results["alpha"])
                mean_val = to_np(results["cos_velocity_removed"])
                std_val = to_np(results["cos_velocity_removed_std"])
                mask = np.isfinite(alpha) & np.isfinite(mean_val) & np.isfinite(std_val)
                
                if np.any(mask):
                    # Plot mean line
                    ax.plot(alpha[mask], mean_val[mask],
                           color=concept_color,
                           linewidth=2.0,
                           linestyle='--',
                           alpha=0.9,
                           label=f"{concept_name} (removed)" if show_legend else "")
                    
                    # Plot std as shaded area
                    ax.fill_between(alpha[mask],
                                   mean_val[mask] - std_val[mask],
                                   mean_val[mask] + std_val[mask],
                                   color=concept_color,
                                   alpha=0.15)
            
            # Backward compatibility: if only cos_velocity exists (old format)
            elif "cos_velocity" in results and "cos_velocity_std" in results:
                alpha = to_np(results["alpha"])
                mean_val = to_np(results["cos_velocity"])
                std_val = to_np(results["cos_velocity_std"])
                mask = np.isfinite(alpha) & np.isfinite(mean_val) & np.isfinite(std_val)
                
                if np.any(mask):
                    ax.plot(alpha[mask], mean_val[mask],
                           color=concept_color,
                           linewidth=2.0,
                           linestyle='-',
                           alpha=0.9)
                    ax.fill_between(alpha[mask],
                                   mean_val[mask] - std_val[mask],
                                   mean_val[mask] + std_val[mask],
                                   color=concept_color,
                                   alpha=0.15)
        
        # Plot random vector results (gray) - only raw version
        if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
            results = concept_data_random[concept_name][layer_num]
            
            # Try new format first
            if "cos_velocity_raw" in results:
                alpha = to_np(results["alpha"])
                cos_vel = to_np(results["cos_velocity_raw"])
                mask = np.isfinite(alpha) & np.isfinite(cos_vel)
                
                if np.any(mask):
                    ax.plot(alpha[mask], cos_vel[mask],
                           color='#AAAAAA',
                           linewidth=1.0,
                           linestyle='-',
                           alpha=0.4)
            # Backward compatibility
            elif "cos_velocity" in results:
                alpha = to_np(results["alpha"])
                cos_vel = to_np(results["cos_velocity"])
                mask = np.isfinite(alpha) & np.isfinite(cos_vel)
                
                if np.any(mask):
                    ax.plot(alpha[mask], cos_vel[mask],
                           color='#AAAAAA',
                           linewidth=1.0,
                           linestyle='-',
                           alpha=0.4)
    
    # Style the axis
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (α)", fontweight='bold', fontsize=10)
    ax.set_ylabel(r"$\cos(\phi, \phi_{\epsilon})$", fontweight='bold', fontsize=10)
    ax.set_title(f"Layer {layer_num}", fontweight='bold', pad=8, fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    
    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, axis='both')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    # Add legend only if requested (typically for first subplot)
    if show_legend:
        from matplotlib.lines import Line2D
        legend_handles = []
        legend_labels = []
        
        # Add concept colors and line styles
        for cn in sorted(all_concepts):
            # Raw version (solid line)
            legend_handles.append(Line2D([0], [0], color=concept_colors[cn], linewidth=2, linestyle='-'))
            legend_labels.append(f"{cn.replace('_', ' ').title()} (Raw)")
            # Removed version (dashed line)
            legend_handles.append(Line2D([0], [0], color=concept_colors[cn], linewidth=2, linestyle='--'))
            legend_labels.append(f"{cn.replace('_', ' ').title()} (Removed)")
        
        # Add random vector indicator
        legend_handles.append(Line2D([0], [0], color='#AAAAAA', linewidth=1.0, alpha=0.4))
        legend_labels.append('Random')
        
        ax.legend(legend_handles, legend_labels, loc='best', ncol=1, frameon=True,
                 fancybox=True, shadow=False, framealpha=0.95,
                 edgecolor='black', fontsize=7)


def plot_trajectory_smoothness(
    model_name: str,
    concept_files_concept: list[str],
    concept_files_random: list[str],
    outpath: str | None
):
    """
    Plot trajectory smoothness metrics for all layers in a single figure.
    Creates a 2-row layout: first row for concept vectors, second row for random vectors.
    Each row has 3 columns (one per layer).
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector files
        concept_files_random: List of random vector files
        outpath: Output path for the plot
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_trajectory_smoothness_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            concept_name = extract_concept_name_from_filename(filename, prefix="trajectory_smoothness_")
        concept_data_concept[concept_name] = obj["results"]
    
    # Load random vector files
    concept_data_random = {}
    for file_path in concept_files_random:
        obj = load_trajectory_smoothness_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            concept_name = extract_concept_name_from_filename(filename, prefix="trajectory_smoothness_")
        concept_data_random[concept_name] = obj["results"]
    
    # Get all concepts
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
    
    # Assign colors to concepts
    concept_colors = assign_colors_to_concepts(all_concepts)
    
    # Calculate grid dimensions (3 columns per row, 2 rows for concept and random)
    n_layers = len(all_layers)
    n_cols = min(n_layers, 3)  # Max 3 columns
    n_layers_per_row = (n_layers + n_cols - 1) // n_cols  # How many rows needed per type
    n_rows = n_layers_per_row * 2  # 2 types: concept and random
    
    # Create figure with subplots
    fig_width = n_cols * 7
    fig_height = n_rows * 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    
    # Ensure axes is 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot concept vectors in first row(s)
    for idx, layer_num in enumerate(all_layers):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Show legend only on the first subplot
        show_legend = (idx == 0)
        plot_single_layer_on_axis(
            ax=ax,
            layer_num=layer_num,
            concept_data_concept=concept_data_concept,
            concept_data_random={},  # Don't plot random in first row
            all_concepts=all_concepts,
            concept_colors=concept_colors,
            show_legend=show_legend
        )
    
    # Hide unused subplots in concept rows
    for idx in range(n_layers, n_layers_per_row * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if row_idx < n_layers_per_row:
            axes[row_idx, col_idx].set_visible(False)
    
    # Plot random vectors in second row(s)
    for idx, layer_num in enumerate(all_layers):
        row_idx = n_layers_per_row + idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Show legend only on the first random subplot
        show_legend = (idx == 0)
        plot_single_layer_on_axis(
            ax=ax,
            layer_num=layer_num,
            concept_data_concept={},  # Don't plot concept in second row
            concept_data_random=concept_data_random,
            all_concepts=all_concepts,
            concept_colors=concept_colors,
            show_legend=show_legend
        )
    
    # Hide unused subplots in random rows
    for idx in range(n_layers, n_layers_per_row * n_cols):
        row_idx = n_layers_per_row + idx // n_cols
        col_idx = idx % n_cols
        if row_idx < n_rows:
            axes[row_idx, col_idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f"{model_name} | Trajectory Smoothness Analysis\nConcept Vectors (Top) vs Random Vectors (Bottom)",
                 fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save or show
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='pdf')
    else:
        plt.show()
    
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot trajectory smoothness analysis")
    ap.add_argument("--outdir", type=str, default="plots", help="where to save PDFs")
    ap.add_argument("--show", action="store_true", help="show interactively instead of saving")
    ap.add_argument("--model", type=str, default=None, help="model name")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save = (not args.show)

    if args.model:
        models_to_plot = {args.model: get_model_name_for_path(args.model)}
    else:
        models_to_plot = find_all_models(base_dir="assets/trajectory_smoothness", 
                                         file_prefix="trajectory_smoothness_")
        if not models_to_plot:
            print("Error: No models found in assets/trajectory_smoothness/")
            return
        print(f"Found {len(models_to_plot)} models to plot:")
        for model_name, model_dir in models_to_plot.items():
            print(f"  - {model_name} (dir: {model_dir})")
    
    for model_name, model_dir in models_to_plot.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        concept_files_concept = find_trajectory_smoothness_files_by_model(model_name, vector_type="concept")
        concept_files_random = find_trajectory_smoothness_files_by_model(model_name, vector_type="random")
        
        if not concept_files_concept and not concept_files_random:
            print(f"  Warning: No trajectory smoothness files found for {model_name}")
            continue
        
        if concept_files_concept:
            print(f"  Found {len(concept_files_concept)} concept vector files")
        if concept_files_random:
            print(f"  Found {len(concept_files_random)} random vector files")
        
        model_short = get_model_name_for_path(model_name)
        output_filename = f"trajectory_smoothness_{model_short}.pdf"
        output_path = None if args.show else os.path.join(args.outdir, output_filename)
        
        plot_trajectory_smoothness(
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
