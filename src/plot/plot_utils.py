# plot_utils.py
# Common utilities for plotting scripts

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_model_name_for_path


def setup_publication_style():
    """Set up publication-quality matplotlib style."""
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


def to_np(x):
    """Convert tensor or array-like to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_concept_colors():
    """Get professional color palette for concepts."""
    colors = {
        'safety': '#A23B72',
        'language_en_fr': '#2E86AB',
        'language_en_fr_paired': '#2E86AB',
        'sycophantic': '#F18F01',
        'evil': '#C73E1D',
        'optimistic': '#06A77D',
        'refusal': '#9C27B0',
    }
    return colors


def get_color_list():
    """Get list of colors for cycling through concepts."""
    return ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#FFA726', '#26C6A0', '#9C27B0']


def assign_colors_to_concepts(concepts):
    """Assign colors to a set of concepts.
    
    Args:
        concepts: Set or list of concept names
    
    Returns:
        Dictionary mapping concept names to colors
    """
    colors = get_concept_colors()
    color_list = get_color_list()
    
    concept_colors = {}
    for i, concept in enumerate(sorted(concepts)):
        concept_colors[concept] = colors.get(concept, color_list[i % len(color_list)])
    
    return concept_colors


def load_pt_file(path: str, expected_key: str = "results"):
    """Load a .pt file and validate it has expected key.
    
    Args:
        path: Path to .pt file
        expected_key: Expected key in the loaded dictionary
    
    Returns:
        Loaded dictionary
    
    Raises:
        ValueError: If expected key is not found
    """
    obj = torch.load(path, map_location="cpu")
    if expected_key not in obj:
        raise ValueError(f"Expected key '{expected_key}' in {path}, got keys: {list(obj.keys())}")
    return obj


def find_all_models(base_dir: str, file_prefix: str = None):
    """Find all models that have data files in base_dir.
    
    Args:
        base_dir: Base directory to search (e.g., "assets/curvature")
        file_prefix: Optional prefix for files to look for (e.g., "curvature_")
    
    Returns:
        Dictionary mapping model names to directory names
    """
    models = {}
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if directory has .pt files
            files = [f for f in os.listdir(item_path) if f.endswith(".pt")]
            if file_prefix:
                files = [f for f in files if f.startswith(file_prefix)]
            
            if files:
                # Try to get model name from first file
                try:
                    first_file = os.path.join(item_path, files[0])
                    obj = load_pt_file(first_file, expected_key="results")
                    file_model = obj.get("model", item)
                    models[file_model] = item  # Map model name to directory name
                except:
                    # If loading fails, use directory name as model name
                    models[item] = item
    
    return models


def find_files_by_model(model_name: str, base_dir: str, file_prefix: str = None, 
                        vector_type: str = None, suffix_filter: str = None):
    """Find all data files for a specific model.
    
    Handles cases where model_name might be "EleutherAI/pythia-70m" but directory is "pythia-70m".
    
    Args:
        model_name: Name of the model
        base_dir: Base directory to search (e.g., "assets/curvature")
        file_prefix: Prefix for files to look for (e.g., "curvature_")
        vector_type: If provided, filter files by vector type ('concept' or 'random')
        suffix_filter: If provided, filter files by suffix (e.g., 'wo_remove', 'w_remove')
    
    Returns:
        Sorted list of file paths
    """
    # Try the full model name first
    model_dir = os.path.join(base_dir, model_name)
    data_files = []
    
    def filter_file(file):
        """Check if file matches all filters."""
        if not file.endswith(".pt"):
            return False
        if file_prefix and not file.startswith(file_prefix):
            return False
        if vector_type is not None and f"_{vector_type}" not in file:
            return False
        if suffix_filter is not None and f"_{suffix_filter}" not in file:
            return False
        return True
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if filter_file(file):
                data_files.append(os.path.join(model_dir, file))
    
    # If not found and model_name contains '/', try just the last part
    if not data_files and '/' in model_name:
        short_name = model_name.split('/')[-1]
        model_dir = os.path.join(base_dir, short_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if filter_file(file):
                    data_files.append(os.path.join(model_dir, file))
    
    # If still not found, search all subdirectories and match by model name in files
    if not data_files:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if filter_file(file):
                        file_path = os.path.join(root, file)
                        try:
                            obj = load_pt_file(file_path, expected_key="results")
                            file_model = obj.get("model", "")
                            # Match if model name matches (full or short)
                            if file_model == model_name or file_model.split('/')[-1] == model_name.split('/')[-1]:
                                data_files.append(file_path)
                        except:
                            pass
    
    return sorted(data_files)


def extract_concept_name_from_filename(filename: str, prefix: str = None):
    """Extract concept name from a data file filename.
    
    Args:
        filename: Base filename (e.g., "curvature_safety_concept_wo_remove.pt")
        prefix: Optional prefix to remove (e.g., "curvature_")
    
    Returns:
        Concept name (e.g., "safety")
    """
    if filename.endswith(".pt"):
        name_part = filename[:-3]  # Remove ".pt" suffix
    else:
        name_part = filename
    
    if prefix and name_part.startswith(prefix):
        name_part = name_part[len(prefix):]
    
    # Remove common suffixes
    for suffix in ["_w_remove", "_wo_remove", "_concept", "_random"]:
        if name_part.endswith(suffix):
            name_part = name_part[:-len(suffix)]
    
    return name_part


def style_axis(ax, xlabel: str = None, ylabel: str = None, title: str = None, 
               use_log_scale: bool = True, grid: bool = True):
    """Apply consistent styling to an axis.
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        use_log_scale: Whether to use log scale for both axes
        grid: Whether to show grid
    """
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
    if title:
        ax.set_title(title, fontweight='bold', pad=15, fontsize=16)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='both')
        ax.set_axisbelow(True)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
