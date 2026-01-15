import argparse
import torch
import transformers
import os
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


def directional_change():
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
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=200)
    parser.add_argument(
        "--layers",
        type=str,
        default="25,50,75",
        help="Comma-separated percentages (e.g., '25,50,75') or layer indices (e.g., '5,10,15'). Use 'all' to run all layers.",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/directional_change.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    # Determine which models to process
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    logger.info(f"Starting directional change measurement...")
    logger.info(f"Models to process: {[model_name for model_name, _ in models_to_process]}")
    device = "cuda"
    dtype = torch.bfloat16
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/directional_change/{model_name}", exist_ok=True)
        
        logger.info(f"Loading model: {model_full_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_full_name, device_map=device, dtype=dtype, trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_full_name, use_fast=True, device=device, dtype=dtype
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Parse layers argument
        layers_to_run = parse_layers_to_run(args.layers, max_layers)
        
        logger.info(f"Total layers in model: {max_layers}")
        logger.info(f"Layers to run: {layers_to_run}")

        for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
            concept_vectors = torch.load(
                f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
            )

            positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
                concept_category_name, concept_category_config
            )
            
            # Select prompts to meet max_tokens constraint
            selected_prompts = []
            total_tokens = 0
            for i in range(min(args.test_size, len(positive_dataset))):
                prompt = positive_dataset[i][dataset_key]
                # Tokenize to check length
                tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
                prompt_length = tokens.input_ids.shape[1]
                
                # Check if adding this prompt would exceed max_tokens
                if total_tokens + prompt_length > args.max_tokens and len(selected_prompts) > 0:
                    break
                
                selected_prompts.append(prompt)
                total_tokens += prompt_length
                
                # Stop if we've reached max_tokens
                if total_tokens >= args.max_tokens:
                    break
            
            logger.info(
                f"Selected {len(selected_prompts)} prompts with ~{total_tokens} tokens "
                f"for {concept_category_name}"
            )
            
            # Tokenize all selected prompts together
            input_ids = tokenizer(
                selected_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_tokens,
            ).to(device)
            input_ids = input_ids.input_ids

            vector_dim = concept_vectors.shape[1]

            # Generate and save a random vector for this concept (reuse if exists)
            random_vector_dir = f"assets/directional_change/{model_name}/random_vectors"
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"

            if os.path.exists(random_vector_path):
                random_vector_data = torch.load(random_vector_path)
                random_vector = random_vector_data["random_vector"]
                logger.info(f"Using saved random vector for {concept_category_name}")
            else:
                # Generate a random vector with the same dimension as concept vectors
                random_vector = torch.randn(vector_dim, dtype=torch.float32)
                # Normalize the random vector to have unit norm
                random_vector = random_vector / random_vector.norm()
                torch.save(
                    {
                        "random_vector": random_vector,
                        "concept_category": concept_category_name,
                    },
                    random_vector_path,
                )
                logger.info(
                    f"Generated and saved random vector for {concept_category_name}"
                )

            # Run directional change analysis with concept vector and random vector separately
            for vector_type, vector_source in [
                ("concept", concept_vectors),
                ("random", random_vector),
            ]:
                results: dict[int, dict[str, torch.Tensor]] = {}

                for layer_idx in tqdm(
                    layers_to_run,
                    desc=f"Measuring directional change ({concept_category_name}, {vector_type})",
                ):
                    if vector_type == "concept":
                        # Use the concept vector from the corresponding layer
                        steering_vector = concept_vectors[layer_idx, :]
                    else:
                        # Use the same random vector for all layers
                        steering_vector = random_vector

                    layer_results = compute_directional_change(
                        model=model,
                        input_ids=input_ids,
                        steering_vector=steering_vector,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        device=device,
                        alpha_min=args.alpha_min,
                        alpha_max=args.alpha_max,
                        alpha_points=args.alpha_points,
                    )

                    results[layer_idx] = layer_results

                # Save results
                save_path = f"assets/directional_change/{model_name}/directional_change_{concept_category_name}_{vector_type}.pt"
                torch.save(
                    {
                        "model": model_full_name,
                        "concept_category": concept_category_name,
                        "vector_type": vector_type,
                        "alpha_min": args.alpha_min,
                        "alpha_max": args.alpha_max,
                        "alpha_points": args.alpha_points,
                        "results": results,
                    },
                    save_path,
                )
                logger.info(f"Saved directional change ({vector_type}) to {save_path}")
        
        # Clean up model to free memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info(f"Finished processing model: {model_full_name}")


def compute_directional_change(
    model,
    input_ids,
    steering_vector: torch.Tensor,
    layer_idx: int,
    dtype,
    device,
    alpha_min: float = 1e-3,
    alpha_max: float = 1e7,
    alpha_points: int = 200,
) -> dict[str, torch.Tensor]:
    """
    Compute directional changes for different alpha values.
    
    For each alpha, we compute:
    - h(alpha): hidden state after steering with alpha * steering_vector
    - h(0): hidden state without steering (baseline)
    - delta = h(alpha) - alpha * v: residual after removing steering contribution
    
    We analyze directional changes:
    1. cos(delta, v): how the residual aligns with steering direction
    2. cos(delta, h(0)): how the residual relates to original direction
    
    Returns:
        Dictionary containing:
        - alpha: tensor of alpha values
        - cos_delta_v: cos(h(alpha) - alpha*v, v) - residual alignment with steering
        - cos_delta_h0: cos(h(alpha) - alpha*v, h(0)) - residual alignment with original
    """
    # Build alpha sweep in log-space
    alphas = torch.logspace(
        float(torch.log10(torch.tensor(alpha_min))),
        float(torch.log10(torch.tensor(alpha_max))),
        steps=int(alpha_points),
    ).tolist()

    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between vectors.
        
        Args:
            a: tensor of shape [N, d] or [d]
            b: tensor of shape [N, d] or [d]
        
        Returns:
            Cosine similarity (scalar if both inputs are 1D, or [N] if 2D)
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        
        a_norm = torch.nn.functional.normalize(a, dim=-1)
        b_norm = torch.nn.functional.normalize(b, dim=-1)
        
        # Handle broadcasting for vector-to-batch comparison
        if a_norm.shape[0] == 1 and b_norm.shape[0] > 1:
            a_norm = a_norm.expand(b_norm.shape[0], -1)
        elif b_norm.shape[0] == 1 and a_norm.shape[0] > 1:
            b_norm = b_norm.expand(a_norm.shape[0], -1)
        
        cos_sim = (a_norm * b_norm).sum(dim=-1)
        return cos_sim

    # Normalize steering vector
    steering_vec_bf16 = steering_vector.to(dtype=torch.bfloat16, device=device)
    steering_vec_normalized = steering_vec_bf16 / steering_vec_bf16.norm()

    # Get baseline hidden state (alpha = 0)
    h0 = hidden_to_flat(
        run_model_with_steering(
            model=model,
            input_ids=input_ids,
            steering_vector=steering_vector,
            layer_idx=layer_idx,
            alpha_value=0.0,
            device=device,
        )
    )  # [batch*seq, d]

    alpha_list = []
    # Cosine similarities - mean and std
    cos_delta_v_list = []     # cos(h(alpha) - alpha*v, v) - mean
    cos_delta_v_std_list = []  # cos(h(alpha) - alpha*v, v) - std
    cos_delta_h0_list = []    # cos(h(alpha) - alpha*v, h(0)) - mean
    cos_delta_h0_std_list = []  # cos(h(alpha) - alpha*v, h(0)) - std
    # Inter-token delta similarities - statistics of pairwise cosine similarities between deltas
    delta_inter_cos_max_list = []   # max pairwise cos sim between deltas
    delta_inter_cos_min_list = []   # min pairwise cos sim between deltas
    delta_inter_cos_mean_list = []  # mean pairwise cos sim between deltas
    delta_inter_cos_std_list = []   # std pairwise cos sim between deltas

    for alpha in tqdm(alphas, desc=f"  Layer {layer_idx}", leave=False):
        # Get hidden state with steering
        h_alpha = hidden_to_flat(
            run_model_with_steering(
                model=model,
                input_ids=input_ids,
                steering_vector=steering_vector,
                layer_idx=layer_idx,
                alpha_value=alpha,
                device=device,
            )
        )  # [batch*seq, d]
        
        # Compute steering contribution
        steering_contribution = alpha * steering_vec_bf16.unsqueeze(0)  # [1, d]
        # Compute delta (change from baseline)
        delta = h_alpha - steering_contribution  # [batch*seq, d]
        
        # ===== Cosine similarities =====
        
        # cos(delta, v): alignment of change with steering direction
        cos_delta_v = _cosine_similarity(delta, steering_vec_normalized)  # [batch*seq]
        cos_delta_v_mean = cos_delta_v.mean().item()
        cos_delta_v_std = cos_delta_v.std().item()
        
        # cos(delta, h(0)): alignment of change with original direction
        cos_delta_h0 = _cosine_similarity(delta, h0)  # [batch*seq]
        cos_delta_h0_mean = cos_delta_h0.mean().item()
        cos_delta_h0_std = cos_delta_h0.std().item()
        
        # ===== Inter-token delta similarities =====
        # Compute pairwise cosine similarities between delta vectors at different token positions
        delta_normalized = torch.nn.functional.normalize(delta, dim=-1)  # [batch*seq, d]
        # Compute cosine similarity matrix: cos_sim[i, j] = cos(delta_i, delta_j)
        delta_cos_sim_matrix = torch.mm(delta_normalized, delta_normalized.t())  # [batch*seq, batch*seq]
        
        # Extract upper triangular part (excluding diagonal) to avoid duplicate pairs and self-similarity
        n_tokens = delta_cos_sim_matrix.shape[0]
        triu_indices = torch.triu_indices(n_tokens, n_tokens, offset=1)
        pairwise_delta_cos_sims = delta_cos_sim_matrix[triu_indices[0], triu_indices[1]]
        
        # Compute statistics
        delta_inter_cos_max = pairwise_delta_cos_sims.max().item()
        delta_inter_cos_min = pairwise_delta_cos_sims.min().item()
        delta_inter_cos_mean = pairwise_delta_cos_sims.mean().item()
        delta_inter_cos_std = pairwise_delta_cos_sims.std().item()
        
        # Append to lists
        alpha_list.append(alpha)
        cos_delta_v_list.append(cos_delta_v_mean)
        cos_delta_v_std_list.append(cos_delta_v_std)
        cos_delta_h0_list.append(cos_delta_h0_mean)
        cos_delta_h0_std_list.append(cos_delta_h0_std)
        delta_inter_cos_max_list.append(delta_inter_cos_max)
        delta_inter_cos_min_list.append(delta_inter_cos_min)
        delta_inter_cos_mean_list.append(delta_inter_cos_mean)
        delta_inter_cos_std_list.append(delta_inter_cos_std)

    return {
        "alpha": torch.tensor(alpha_list, dtype=torch.float32),
        "cos_delta_v": torch.tensor(cos_delta_v_list, dtype=torch.float32),
        "cos_delta_v_std": torch.tensor(cos_delta_v_std_list, dtype=torch.float32),
        "cos_delta_h0": torch.tensor(cos_delta_h0_list, dtype=torch.float32),
        "cos_delta_h0_std": torch.tensor(cos_delta_h0_std_list, dtype=torch.float32),
        "delta_inter_cos_max": torch.tensor(delta_inter_cos_max_list, dtype=torch.float32),
        "delta_inter_cos_min": torch.tensor(delta_inter_cos_min_list, dtype=torch.float32),
        "delta_inter_cos_mean": torch.tensor(delta_inter_cos_mean_list, dtype=torch.float32),
        "delta_inter_cos_std": torch.tensor(delta_inter_cos_std_list, dtype=torch.float32),
        "layer_idx": layer_idx,
    }


if __name__ == "__main__":
    directional_change()
