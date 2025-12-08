import argparse
import contextlib
import torch
import transformers
import torch.nn.functional as F
import os
import datasets
import gc
from loguru import logger
from utils import set_seed, _get_layers_container
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_LAYERS = {
    "google/gemma-3-270m-it": 18,
    "google/gemma-3-4b-it": 34,  # TODO: replace with Gemma2, and add Mistral3
    "google/gemma-3-12b-it": 48,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8b": 36,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-410m": 24,
    "EleutherAI/pythia-160m": 12,
}

CONCEPT_CATEGORIES = {
    "safety": ["assets/harmbench", "instruction"],
    # "language_en_fr": ["assets/language_translation", "text"],
    # "random": ["assets/harmbench", "instruction"],
    # "random1": ["assets/harmbench", "instruction"],
    # "random2": ["assets/language_translation", "text"],
    # "random3": ["assets/language_translation", "text"],
}


def config() -> argparse.Namespace:
    """
    Config for approximation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=MODEL_LAYERS.keys(),
    )
    parser.add_argument(
        "--concept_category",
        type=str,
        default="safety",
        choices=CONCEPT_CATEGORIES.keys(),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approx_dataset_size", type=int, default=30)
    parser.add_argument("--alpha_factor", type=int, default=1000)
    parser.add_argument("--alpha_min", type=float, default=1)
    return parser.parse_args()


def _align_vec_to_hidden(vec: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Broadcast concept vector to match hidden shape."""
    if vec.dim() < hidden.dim():
        vec = vec.view(*([1] * (hidden.dim() - vec.dim())), *vec.shape)
    vec = vec.to(device=hidden.device, dtype=hidden.dtype)
    return vec.expand_as(hidden)


def approx() -> None:
    """
    Test whether the approximation is correct
    """
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/approx.log")
    logger.info("Starting Approximation...")
    args = config()
    # Disable flash attention kernels when computing higher-order derivatives.
    # Flash attention backward currently lacks a second derivative, which we need
    # for autograd.functional.jvp below. Falling back to math kernels avoids
    # RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_backward
    if args.device.startswith("cuda") and torch.cuda.is_available():
        # Prefer the consolidated API when available (PyTorch >= 2.1).
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        else:
            # Fallback for older PyTorch versions.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    set_seed(args.seed)
    model_name = args.model.split("/")[-1]
    dtype = getattr(torch, args.dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=True, dtype=dtype
    )
    max_layers = MODEL_LAYERS[args.model]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"args: {args}")

    for concept_category_name, concept_category_metadata in CONCEPT_CATEGORIES.items():
        save_path = f"assets/linearity/{model_name}/{concept_category_name}.pt"
        concept_vectors = torch.load(save_path)
        if concept_category_name in ["language_en_fr", "random2", "random3"]:
            dataset_path = os.path.join(concept_category_metadata[0], "en.jsonl")
            dataset = datasets.load_dataset(
                "json", data_files=dataset_path, split="train"
            )
            dataset_key = concept_category_metadata[1]
        elif concept_category_name in ["random1", "safety", "random"]:
            dataset_path = os.path.join(
                concept_category_metadata[0], "harmful_data.jsonl"
            )
            dataset = datasets.load_dataset(
                "json", data_files=dataset_path, split="train"
            )
            dataset_key = concept_category_metadata[1]
        else:
            raise ValueError(f"Invalid concept category name: {concept_category_name}")
        dataset = dataset.shuffle().select(range(args.approx_dataset_size))
        input_prompts = dataset[: args.approx_dataset_size][dataset_key]
        input_ids = tokenizer(
            input_prompts, return_tensors="pt", truncation=True, padding=True
        ).to(args.device)
        input_ids = input_ids.input_ids
        mses = []
        cosine_sims = []
        rel_errs = []
        for layer_idx in tqdm(range(max_layers), desc="Approximation"):
            cosine_sims.append([])
            mses.append([])
            rel_errs.append([])
            concept_vector = concept_vectors[layer_idx, :]
            logger.info(f"Concept vector: {concept_vector.shape}")
            layers_container = _get_layers_container(model)
            target_layer_module = layers_container[layer_idx]
            original_output = model(input_ids)
            logit_wo_steering = original_output.logits

            # Capture the baseline hidden states at the steering layer once
            captured = {}

            def _capture_hidden(module, inputs, output):
                captured["hidden"] = output[0] if isinstance(output, tuple) else output
                return output

            capture_handle = target_layer_module.register_forward_hook(_capture_hidden)
            with torch.no_grad():
                _ = model(input_ids)
            capture_handle.remove()

            base_hidden = captured["hidden"].detach().requires_grad_(True)

            def _logits_from_hidden(new_hidden: torch.Tensor) -> torch.Tensor:
                """Forward pass where the steering layer output is replaced by new_hidden."""

                def _replace(_, __, output):
                    if isinstance(output, tuple):
                        return (new_hidden,) + output[1:]
                    return new_hidden

                handle = target_layer_module.register_forward_hook(_replace)
                out = model(input_ids)
                handle.remove()
                return out.logits

            # Jacobian-vector product from steering layer output to logits
            aligned_vec = _align_vec_to_hidden(concept_vector, base_hidden)
            if base_hidden.is_cuda:
                if hasattr(torch.backends.cuda, "sdp_kernel"):
                    sdp_ctx = torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_math=True, enable_mem_efficient=False
                    )
                else:
                    sdp_ctx = contextlib.nullcontext()
            else:
                sdp_ctx = contextlib.nullcontext()
            with sdp_ctx:
                phi_x, jvp = torch.autograd.functional.jvp(
                    _logits_from_hidden,
                    (base_hidden,),
                    (aligned_vec,),
                    create_graph=False,
                    strict=False,
                )
            logger.info(f"phi_x: {phi_x.shape}")
            logger.info(f"jvp (layer->{layer_idx}->logits): {jvp.shape}")
            for idx in tqdm(range(args.alpha_factor)):
                cur_approx_alpha = args.alpha_min * (idx + 1)

                def _forward_hook(module, inputs, output):
                    # Ensure concept vector matches device/dtype
                    if isinstance(output, tuple):
                        hidden = output[0]
                        vec = concept_vector.to(
                            device=hidden.device, dtype=hidden.dtype
                        )
                        hidden = hidden + cur_approx_alpha * vec
                        return (hidden,) + output[1:]
                    else:
                        vec = concept_vector.to(
                            device=output.device, dtype=output.dtype
                        )
                        return output + cur_approx_alpha * vec

                hook_handle = target_layer_module.register_forward_hook(_forward_hook)
                steered_output = model(input_ids)
                steered_logits = steered_output.logits
                hook_handle.remove()
                # Linear prediction: logits at x plus alpha * J v
                delta_true = steered_logits - phi_x
                delta_predicted = jvp * cur_approx_alpha
                # Compare real steered logits to the JVP-based prediction
                cosine_sim = F.cosine_similarity(
                    delta_true.view(-1, delta_true.size(-1)),
                    delta_predicted.view(-1, delta_predicted.size(-1)),
                    dim=-1,
                ).mean()

                mse = F.mse_loss(delta_true, delta_predicted)
                rel_err = (delta_true - delta_predicted).norm() / (delta_true.norm() + 1e-8)
                logger.info(f"rel_err={rel_err:.6f}")
                logger.info(
                    f"[layer {layer_idx}] alpha={cur_approx_alpha:.4f} "
                    f"cos_sim={cosine_sim:.6f} mse={mse.item():.6f} rel_err={rel_err:.6f}"
                )
                cosine_sims[layer_idx].append(cosine_sim.item())
                mses[layer_idx].append(mse.item())
                rel_errs[layer_idx].append(rel_err.item())
        os.makedirs(f"assets/approx/", exist_ok=True)
        torch.save(
            cosine_sims,
            f"assets/approx/cosine_sims_{model_name}_{concept_category_name}.pt",
        )
        torch.save(mses, f"assets/approx/mses_{model_name}_{concept_category_name}.pt")


if __name__ == "__main__":
    approx()
