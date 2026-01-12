import argparse
import torch
import transformers
import os
from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    set_seed,
    _get_layers_container,
    load_concept_datasets,
)
from loguru import logger
from tqdm import tqdm


def linear():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=200)
    parser.add_argument("--eps_rel", type=float, default=1e-2)
    parser.add_argument("--eps_min", type=float, default=1e-3)
    parser.add_argument("--eps_max", type=float, default=1e2)
    parser.add_argument("--denom_eps", type=float, default=1e-8)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    model_name = args.model.split("/")[-1]
    os.makedirs(f"assets/linear/{model_name}", exist_ok=True)
    logger.info(f"Starting linear measurement...")
    logger.info(f"Loading model: {args.model}")
    device = "cuda"
    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=True, device=device, dtype=dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_layers = MODEL_LAYERS[args.model]
    for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
        concept_vectors = torch.load(
            f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
        )
        positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
            concept_category_name, concept_category_config
        )
        actual_size = min(args.test_size, len(positive_dataset))
        input_prompts = positive_dataset[:actual_size][dataset_key]
        input_ids = tokenizer(
            input_prompts, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        input_ids = input_ids.input_ids

        # Curvature sweep per layer (discrete curvature as "linearity" score)
        results: dict[int, dict[str, torch.Tensor]] = {}
        num_layers = min(max_layers, concept_vectors.shape[0])
        for layer_idx in tqdm(range(num_layers), desc=f"Measuring curvature ({concept_category_name})"):
            concept_vector = concept_vectors[layer_idx, :]
            alpha_list: list[torch.Tensor] = []
            eps_list: list[torch.Tensor] = []
            kappa_list: list[torch.Tensor] = []

            for item in model_steering(
                model=model,
                input_ids=input_ids,
                concept_vector=concept_vector,
                layer_idx=layer_idx,
                dtype=dtype,
                device=device,
                alpha_factor_init=0,
                alpha_factor_num=0,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_points=args.alpha_points,
                eps_rel=args.eps_rel,
                eps_min=args.eps_min,
                eps_max=args.eps_max,
                denom_eps=args.denom_eps,
            ):
                alpha_list.append(item["alpha"])
                eps_list.append(item["eps"])
                kappa_list.append(item["kappa"])

            results[layer_idx] = {
                "alpha": torch.stack(alpha_list) if alpha_list else torch.empty((0,)),
                "eps": torch.stack(eps_list) if eps_list else torch.empty((0,)),
                "kappa": torch.stack(kappa_list) if kappa_list else torch.empty((0,)),
            }

        save_path = f"assets/linear/{model_name}/curvature_{concept_category_name}.pt"
        torch.save(
            {
                "model": args.model,
                "concept_category": concept_category_name,
                "alpha_min": args.alpha_min,
                "alpha_max": args.alpha_max,
                "alpha_points": args.alpha_points,
                "eps_rel": args.eps_rel,
                "eps_min": args.eps_min,
                "eps_max": args.eps_max,
                "denom_eps": args.denom_eps,
                "results": results,
            },
            save_path,
        )
        logger.info(f"Saved curvature sweep to {save_path}")


def model_steering(
    model,
    input_ids,
    concept_vector,
    layer_idx,
    dtype,
    device,
    alpha_factor_init,
    alpha_factor_num,
    alpha_min: float = 1e-3,
    alpha_max: float = 1e7,
    alpha_points: int = 200,
    eps_rel: float = 1e-2,
    eps_min: float = 1e-3,
    eps_max: float = 1e2,
    denom_eps: float = 1e-8,
):
    """
    Sweep alphas and compute discrete curvature with a backward epsilon:

      v(alpha) = (x(alpha) - x(alpha - eps)) / eps
      kappa(alpha) = ||v(alpha) - v(alpha - eps)|| / (||v(alpha)|| + ||v(alpha - eps)|| + denom_eps)

    Here eps is chosen per alpha as:
      eps = clamp(alpha * eps_rel, eps_min, eps_max)

    Notes:
    - x(alpha) is the mean-pooled last-block hidden state (pre final LN for most architectures).
    - We require alpha - 2*eps > 0 to form v(alpha - eps).
    """

    # Build alpha sweep in linear alpha-space, but spaced in log-space for coverage
    alphas = torch.logspace(
        float(torch.log10(torch.tensor(alpha_min))),
        float(torch.log10(torch.tensor(alpha_max))),
        steps=int(alpha_points),
    ).tolist()

    layers_container = _get_layers_container(model)
    target_layer_module = layers_container[layer_idx]
    last_layer_module = layers_container[len(layers_container) - 1]

    def _run_with_alpha(alpha_value: float) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def _last_layer_forward_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden.detach()
            return output

        def _steer_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                vec = concept_vector.to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden + (alpha_value * vec)
                return (hidden,) + output[1:]
            vec = concept_vector.to(device=output.device, dtype=output.dtype)
            return output + (alpha_value * vec)

        last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
        steer_handle = target_layer_module.register_forward_hook(_steer_hook)
        _ = model(input_ids, output_hidden_states=True)
        steer_handle.remove()
        last_handle.remove()

        h = captured.get("h", None)
        if h is None:
            raise RuntimeError("Failed to capture hidden states for curvature computation")
        return h

    def _hidden_to_x(h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq, d] -> x: [d]
        return h.mean(dim=(0, 1)).to(torch.float32)

    for alpha in alphas:
        eps = float(max(eps_min, min(eps_max, abs(alpha) * eps_rel)))
        if alpha - 2.0 * eps <= 0.0:
            continue

        # Backward-eps velocities:
        # v(alpha) depends on x(alpha) and x(alpha-eps)
        # v(alpha-eps) depends on x(alpha-eps) and x(alpha-2eps)
        x0 = _hidden_to_x(_run_with_alpha(alpha))
        x1 = _hidden_to_x(_run_with_alpha(alpha - eps))
        x2 = _hidden_to_x(_run_with_alpha(alpha - 2.0 * eps))

        v_alpha = (x0 - x1) / eps
        v_prev = (x1 - x2) / eps  # v(alpha - eps)

        num = torch.norm(v_alpha - v_prev, p=2)
        den = torch.norm(v_alpha, p=2) + torch.norm(v_prev, p=2) + float(denom_eps)
        kappa = num / den

        yield {
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "eps": torch.tensor(eps, dtype=torch.float32),
            "kappa": kappa.to(torch.float32),
        }


if __name__ == "__main__":
    linear()
