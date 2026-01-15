# plot_curvature.py
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
    style_axis,
    get_model_name_for_path
)

# Set publication-quality style
setup_publication_style()


def load_curvature_pt(path: str):
    """Load a curvature .pt file."""
    return load_pt_file(path, expected_key="results")


def find_curvature_files_by_model(model_name: str, base_dir="assets/curvature", 
                                   vector_type: str = None, remove_type: str = None):
    """Find all curvature_*.pt files for a specific model."""
    return find_files_by_model(
        model_name=model_name,
        base_dir=base_dir,
        file_prefix="curvature_",
        vector_type=vector_type,
        suffix_filter=remove_type
    )


def plot_single_layer_on_axis(ax, layer_num: int, concept_data_concept: dict, concept_data_random: dict, 
                              all_concepts: set, concept_colors: dict, show_legend: bool = False):
    """
    Plot kappa vs alpha for a single layer on a given axis.
    
    Args:
        ax: Matplotlib axis object to plot on
        layer_num: Layer number to plot
        concept_data_concept: Dictionary of concept data {concept_name: results}
        concept_data_random: Dictionary of random data {concept_name: results}
        all_concepts: Set of all concept names
        concept_colors: Dictionary mapping concept names to colors
        show_legend: Whether to show legend on this subplot
    """
    # Plot all concepts for this layer (both concept and random vectors)
    for concept_name in sorted(all_concepts):
        # Plot concept vector results (solid, thick, colored)
        if concept_name in concept_data_concept and layer_num in concept_data_concept[concept_name]:
            results = concept_data_concept[concept_name]
            alpha = to_np(results[layer_num]["alpha"])
            kappa = to_np(results[layer_num]["kappa"])
            mask = np.isfinite(alpha) & np.isfinite(kappa)
            
            if np.any(mask):
                ax.plot(alpha[mask], kappa[mask], 
                       color=concept_colors[concept_name],
                       label=f"{concept_name.replace('_', ' ').title()} (Concept)",
                       linewidth=2.5, marker='o', markersize=4,
                       markevery=max(1, len(alpha[mask]) // 15), alpha=0.95, linestyle='-')
        
        # Plot random vector results (dashed, thin, gray-ish)
        if concept_name in concept_data_random and layer_num in concept_data_random[concept_name]:
            results = concept_data_random[concept_name]
            alpha = to_np(results[layer_num]["alpha"])
            kappa = to_np(results[layer_num]["kappa"])
            mask = np.isfinite(alpha) & np.isfinite(kappa)
            
            if np.any(mask):
                ax.plot(alpha[mask], kappa[mask], 
                       color='#888888',  # Use gray for all random vectors
                       label=f"{concept_name.replace('_', ' ').title()} (Random)",
                       linewidth=1.5, marker='x', markersize=3,
                       markevery=max(1, len(alpha[mask]) // 15), alpha=0.7, linestyle=':')
    
    # Style the axis
    style_axis(
        ax,
        xlabel="Alpha (α)",
        ylabel="Curvature (κ)",
        title=f"Layer {layer_num}",
        use_log_scale=True,
        grid=True
    )
    
    # Add legend only if requested (typically for first subplot)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', ncol=1, frameon=True, 
                 fancybox=True, shadow=False, framealpha=0.95,
                 edgecolor='black', fontsize=8)


def plot_all_layers_all_concepts(model_name: str, concept_files_concept: list[str], concept_files_random: list[str], outdir: str | None, remove_suffix: str = ""):
    """
    Plot kappa vs alpha for all layers in a single figure.
    Creates a grid layout with 3 columns, where each subplot shows one layer.
    
    Args:
        model_name: Name of the model
        concept_files_concept: List of concept vector curvature file paths
        concept_files_random: List of random vector curvature file paths
        outdir: Output directory for the plots
        remove_suffix: Suffix to add to title (e.g., "wo_remove" or "w_remove")
    """
    # Load concept vector files
    concept_data_concept = {}
    for file_path in concept_files_concept:
        obj = load_curvature_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            concept_name = extract_concept_name_from_filename(filename, prefix="curvature_")
        concept_data_concept[concept_name] = obj["results"]
    
    # Load random vector files
    concept_data_random = {}
    for file_path in concept_files_random:
        obj = load_curvature_pt(file_path)
        concept_name = obj.get("concept_category", None)
        if concept_name is None:
            filename = os.path.basename(file_path)
            concept_name = extract_concept_name_from_filename(filename, prefix="curvature_")
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
    
    # Assign colors to concepts
    concept_colors = assign_colors_to_concepts(all_concepts)
    
    # Get short model name for filenames
    model_short = get_model_name_for_path(model_name)
    
    # Calculate grid dimensions (3 columns per row)
    n_layers = len(all_layers)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig_width = n_cols * 7
    fig_height = n_rows * 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    
    # Flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each layer
    for idx, layer_num in enumerate(all_layers):
        ax = axes[idx]
        # Show legend only on the first subplot
        show_legend = (idx == 0)
        plot_single_layer_on_axis(
            ax=ax,
            layer_num=layer_num,
            concept_data_concept=concept_data_concept,
            concept_data_random=concept_data_random,
            all_concepts=all_concepts,
            concept_colors=concept_colors,
            show_legend=show_legend
        )
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    title_suffix = f" ({remove_suffix})" if remove_suffix else ""
    fig.suptitle(f"{model_name} | Curvature vs Alpha - All Layers{title_suffix}", 
                 fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save or show
    if outdir:
        output_filename = f"all_layers_{model_short}_{remove_suffix}.pdf"
        output_path = os.path.join(outdir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"    ✓ Saved: {output_filename}")
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
        models_to_plot = find_all_models(base_dir="assets/curvature", file_prefix="curvature_")
        if not models_to_plot:
            print("Error: No models found in assets/curvature/")
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
                plot_all_layers_all_concepts(
                    model_name,
                    concept_files_concept if concept_files_concept else [],
                    concept_files_random if concept_files_random else [],
                    None if args.show else args.outdir,
                    remove_suffix=remove_type,
                )
    
    if save:
        print(f"\n{'='*60}")
        print(f"All plots saved to: {args.outdir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
