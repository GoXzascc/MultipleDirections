import argparse
import torch
import transformers
import os
import torch.nn.functional as F
from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    set_seed,
    run_model_with_steering,
    hidden_to_flat,
    get_model_name_for_path,
    parse_layers_to_run,
)
from extract_concepts import load_concept_datasets
from loguru import logger
from tqdm import tqdm

def compute_linearity_score(trajectory_data):
    """
    Compute the linearity score based on PCA variance explained.
    
    Args:
        trajectory_data (torch.Tensor): [num_steps, num_samples, hidden_dim]
            The collected hidden states for each step and sample.
            
    Returns:
        tuple: (mean_score, std_score)
            - mean_score: Average linearity score (PC1 var / Total var) across samples
            - std_score: Standard deviation of linearity score across samples
    """
    # Permute to [num_samples, num_steps, hidden_dim] to treat each sample's trajectory independently
    # We want to measure if EACH trajectory is a line.
    X = trajectory_data.permute(1, 0, 2).float() # [N, T, D]
    
    # Center each trajectory independently
    X_mean = X.mean(dim=1, keepdim=True)
    X_centered = X - X_mean
    
    # Compute SVD for each sample
    # torch.linalg.svdvals is efficient and batched
    # S has shape [N, min(T, D)]
    S = torch.linalg.svdvals(X_centered)
    
    # Variance is proportional to squared singular values
    eigenvalues = S ** 2
    
    total_variance = eigenvalues.sum(dim=-1)
    pc1_variance = eigenvalues[:, 0]
    
    # Avoid division by zero
    epsilon = 1e-12
    valid_mask = total_variance > epsilon
    
    scores = torch.ones_like(total_variance) # Default to 1.0 if no variance
    scores[valid_mask] = pc1_variance[valid_mask] / total_variance[valid_mask]
    
    return scores.mean().item(), scores.std().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to process. If not specified, process all models. Available: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to use from the dataset",
    )
    # Trajectory sweep parameters
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=20) # Fewer points than smoothness
    
    parser.add_argument(
        "--layers",
        type=str,
        default="25,50,75",
        help="Comma-separated percentages or layer indices.",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/linear/{model_name}", exist_ok=True)
        
        logger.info(f"Loading model: {model_full_name}")
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_full_name, device_map=device, dtype=dtype, trust_remote_code=True
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_full_name, use_fast=True, device=device, dtype=dtype
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load model {model_full_name}: {e}")
            continue

        layers_to_run = parse_layers_to_run(args.layers, max_layers)
        
        for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
            concept_vectors_path = f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
            if not os.path.exists(concept_vectors_path):
                logger.warning(f"Concept vectors not found for {concept_category_name} in {model_name}. Skipping.")
                continue
                
            concept_vectors = torch.load(concept_vectors_path)
            
            # Load dataset and select prompts
            positive_dataset, _, dataset_key = load_concept_datasets(
                concept_category_name, concept_category_config
            )
            
            selected_prompts = []
            total_tokens = 0
            for i in range(min(args.test_size, len(positive_dataset))):
                prompt = positive_dataset[i][dataset_key]
                tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
                prompt_length = tokens.input_ids.shape[1]
                if total_tokens + prompt_length > args.max_tokens and len(selected_prompts) > 0:
                    break
                selected_prompts.append(prompt)
                total_tokens += prompt_length
                if total_tokens >= args.max_tokens:
                    break
            
            input_ids = tokenizer(
                selected_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_tokens,
            ).to(device).input_ids

            # Generate random vector for comparison
            random_vector_dir = f"assets/linear/{model_name}/random_vectors"
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"
            
            vector_dim = concept_vectors.shape[1]
            if os.path.exists(random_vector_path):
                random_vector_data = torch.load(random_vector_path)
                # handle both formats if old exists
                if isinstance(random_vector_data, dict):
                    random_vector = random_vector_data["random_vector"]
                else:
                    random_vector = random_vector_data
            else:
                random_vector = torch.randn(vector_dim, dtype=torch.float32)
                random_vector = random_vector / random_vector.norm()
                torch.save({"random_vector": random_vector}, random_vector_path)
            
            # Run analysis
            for vector_type, vector_source in [("concept", concept_vectors), ("random", random_vector)]:
                results = {}
                
                for layer_idx in tqdm(layers_to_run, desc=f"Linearity ({concept_category_name}, {vector_type})"):
                    if vector_type == "concept":
                        steering_vector = concept_vectors[layer_idx, :]
                    else:
                        steering_vector = random_vector
                    
                    # Sweep alphas and collect trajectories
                    outputs = [] # List of [batch*seq, d]
                    
                    # Create alphas range
                    alphas = torch.logspace(
                         float(torch.log10(torch.tensor(args.alpha_min))),
                         float(torch.log10(torch.tensor(args.alpha_max))),
                         steps=args.alpha_points
                    ).tolist()
                    
                    # Ensure 0.0 is included
                    if 0.0 not in alphas:
                        alphas = [0.0] + alphas
                    
                    for alpha in alphas:
                        h = run_model_with_steering(
                            model=model,
                            input_ids=input_ids,
                            steering_vector=steering_vector,
                            layer_idx=layer_idx,
                            alpha_value=alpha,
                            device=device
                        )
                        # h is [batch, seq, d]
                        outputs.append(hidden_to_flat(h, target_dtype=torch.float32).cpu()) 
                    
                    # Stack: [num_alphas, total_tokens, d]
                    trajectory_data = torch.stack(outputs)
                    
                    # Compute Linearity
                    mean_score, std_score = compute_linearity_score(trajectory_data)
                    
                    results[layer_idx] = {
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "alphas": alphas
                    }
                    logger.info(f"Linearity for {concept_category_name} in {model_name} at layer {layer_idx}: {mean_score:.4f} +/- {std_score:.4f}")
                
                # Save results
                save_path = f"assets/linear/{model_name}/linearity_{concept_category_name}_{vector_type}.pt"
                torch.save(
                    {
                        "model": model_full_name,
                        "concept_category": concept_category_name,
                        "vector_type": vector_type,
                        "results": results,
                    },
                    save_path
                )
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
