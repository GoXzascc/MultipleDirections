import argparse
import torch
import os
import transformer_lens
import datasets
import gc
import transformers
import numpy as np
from loguru import logger
from tqdm import tqdm
from utils import set_seed, seed_from_name, _get_layers_container

# TODO: how to handle the last layer?

MODEL_LAYERS = {
    "google/gemma-3-270m-it": 18,
    "google/gemma-3-4b-it": 34,  # TODO: multi model
    "google/gemma-3-12b-it": 48,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8b": 36,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-410m": 24,
    "EleutherAI/pythia-160m": 12,
    "mistralai/Ministral-3-3B-Instruct-2512": 26,
    "mistralai/Mistral-3-8B-Instruct-2512": 34,
    "mistralai/Mistral-3-14B-Instruct-2512": 40,
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
    Config for linearity
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-160m",
        choices=MODEL_LAYERS.keys(),
        help="the model to calculate the linearity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="the device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="the dtype to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed to use")
    parser.add_argument(
        "--concept_vector_dataset_size",
        type=int,
        default=300,
        help="the maximum size of the dataset to calculate the concept vector",
    )
    parser.add_argument(
        "--concept_vector_pretrained",
        action="store_true",
        help="whether to use the pretrained concept vector",
    )
    parser.add_argument(
        "--linearity_dataset_size",
        type=int,
        default=30,
        help="the maximum size of the dataset to calculate the linearity",
    )
    parser.add_argument(
        "--concept_vector_alpha",
        type=float,
        default=100,
        help="the beginning alpha to use to calculate the concept vector",
    )
    parser.add_argument(
        "--alpha_factor",
        type=int,
        default=100,
        help="the factor to multiply the alpha by",
    )
    parser.add_argument(
        "--remove_concept_vector",
        action="store_true",
        help="whether to remove the concept vector",
    )
    parser.add_argument(
        "--linearity_metric",
        type=str,
        nargs="+",
        default=["norm"],
        choices=["lss", "lsr", "norm"],
        help="the metrics to use to calculate the linearity",
    )
    return parser.parse_args()


def linearity() -> None:
    """
    Linearity of llm for random concept vectors and steering concept vectors
    We measure the lss, lsr and norm score under original hidden states difference and the difference after removing the concept vector from the hidden states
    """
    args = config()
    set_seed(args.seed)
    # add logger
    model_name = args.model.split("/")[-1]

    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"assets/linearity/{model_name}", exist_ok=True)
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    max_layers = MODEL_LAYERS[args.model]
    # load model
    logger.info(f"args.linearity_metric: {args.linearity_metric}")

    for concept_category_name, concept_category_metadata in CONCEPT_CATEGORIES.items():
        save_path = f"assets/linearity/{model_name}/{concept_category_name}.pt"
        if args.concept_vector_pretrained:
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
        else:
            model = transformer_lens.HookedTransformer.from_pretrained(
                args.model, device=args.device, dtype=dtype, trust_remote_code=True
            )
            concept_vectors, dataset, dataset_key = obtain_concept_vector(
                concept_category_name,
                concept_category_metadata,
                model,
                model_name,
                max_layers,
                device=args.device,
                max_dataset_size=args.concept_vector_dataset_size,
                save_path=save_path,
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()
        dataset = dataset.shuffle().select(range(args.linearity_dataset_size))
        input_prompts = dataset[: args.linearity_dataset_size][dataset_key]
        logger.info(f"Concept vectors shape: {concept_vectors.shape}")
        for metric in args.linearity_metric:
            logger.info(f"Measuring {metric} for {concept_category_name}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=args.device,
                dtype=dtype,
                trust_remote_code=True,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                dtype=dtype,
            )

            for layer_idx in tqdm(range(max_layers), desc="Measuring linearity"):
                concept_vector = concept_vectors[layer_idx, :]
                logger.info(f"Concept vector: {concept_vector.shape}")
                # Set padding token if not already set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                layers_container = _get_layers_container(model)
                target_layer_module = layers_container[layer_idx]
                input_ids = tokenizer(
                    input_prompts, return_tensors="pt", truncation=True, padding=True
                ).to(args.device)
                input_ids = input_ids.input_ids
                output = model(input_ids, output_hidden_states=True)
                probs_wo_steering = torch.nn.functional.softmax(output.logits, dim=-1)
                idx_base_top1, _ = probs_wo_steering.max(dim=-1)
                logits_entropy_wo_steering = -torch.sum(probs_wo_steering * torch.log(probs_wo_steering + 1e-10), dim=-1)
                last_layer_hidden_states = output.hidden_states[-1]
                dim_middle_states = last_layer_hidden_states.shape[-1]
                seq_len = last_layer_hidden_states.shape[1]
                if metric == "lss":
                    lss_middle_states = torch.zeros(
                        2,
                        seq_len * args.linearity_dataset_size,
                        dim_middle_states,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )  # [0, :] is the base state, [1, :] is the modified state
                elif metric == "lsr":
                    last_logits_with_hook = (
                        last_layer_hidden_states.reshape(-1, dim_middle_states)
                        .detach()
                        .to(torch.float32)
                    )
                    seq_diff_norm = torch.zeros(
                        seq_len * args.linearity_dataset_size,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )
                elif metric == "norm":
                    norm_related = {"top1_probs": [], "entropy_diff_mean": [], "entropy_diff_std": [], "delta_p_flip_mean": [], "delta_p_flip_std": [], "delta_tail_mean": [], "delta_tail_std": [], "norm_diff_mean": [], "norm_diff_std": [], "unique_ratio_steered": [], "unique_ratio_wo_steering": []}
                else:
                    raise ValueError(f"Invalid metric: {metric}")

                for i in tqdm(range(args.alpha_factor), desc="Measuring linearity"):
                    concept_vector_alpha = args.concept_vector_alpha * (i + 1)

                    def _forward_hook(module, inputs, output):
                        # Ensure concept vector matches device/dtype
                        if isinstance(output, tuple):
                            hidden = output[0]
                            vec = concept_vector.to(
                                device=hidden.device, dtype=hidden.dtype
                            )
                            hidden = hidden + concept_vector_alpha * vec
                            return (hidden,) + output[1:]
                        else:
                            vec = concept_vector.to(
                                device=output.device, dtype=output.dtype
                            )
                            return output + concept_vector_alpha * vec

                    hook_handle = target_layer_module.register_forward_hook(
                        _forward_hook
                    )
                    steered_output = model(input_ids, output_hidden_states=True)
                    steered_last_layer_hidden_states = steered_output.hidden_states[-1]
                    hook_handle.remove()
                    if metric == "lss":
                        lss_middle_states = compute_line_shape_score_middle_states(
                            steered_last_layer_hidden_states.reshape(
                                -1, dim_middle_states
                            )
                            .detach()
                            .to(torch.float32),
                            last_layer_hidden_states.reshape(-1, dim_middle_states)
                            .detach()
                            .to(torch.float32),
                            lss_middle_states,
                            i,
                            args.concept_vector_alpha,
                            concept_vector,
                            remove_concept_vector=args.remove_concept_vector,
                        )
                    elif metric == "lsr":
                        last_logits_with_hook, seq_diff_norm = (
                            compute_lsr_middle_states(
                                steered_last_layer_hidden_states.reshape(
                                    -1, dim_middle_states
                                )
                                .detach()
                                .to(torch.float32),
                                last_logits_with_hook,
                                last_layer_hidden_states.reshape(-1, dim_middle_states)
                                .detach()
                                .to(torch.float32),
                                seq_diff_norm,
                                i,
                                args.alpha_factor,
                                args.concept_vector_alpha,
                                concept_vector,
                                remove_concept_vector=args.remove_concept_vector,
                            )
                        )
                    elif metric == "norm":
                        if args.remove_concept_vector:
                            
                            # entropy difference
                            probs_steered = torch.nn.functional.softmax(steered_output.logits, dim=-1)
                            logits_entropy_steered = -torch.sum(probs_steered * torch.log(probs_steered + 1e-10), dim=-1)
                            entropy_diff = ((logits_entropy_steered - logits_entropy_wo_steering) / logits_entropy_wo_steering)
                            
                            # top1 delta p flip
                            top1_probs_index_steered = probs_steered.argmax(dim=-1)
                            top1_probs_index_wo_steering = probs_wo_steering.argmax(dim=-1)
                            p_new_top1 = probs_steered.gather(
                                -1, top1_probs_index_steered.unsqueeze(-1)
                            ).squeeze(-1)
                            p_orig_top1 = probs_wo_steering.gather(
                                -1, top1_probs_index_wo_steering.unsqueeze(-1)
                            ).squeeze(-1)
                            tau = 0.8
                            keep = idx_base_top1 < tau          
                            delta_p_flip = p_new_top1 - p_orig_top1
                            
                            # unique ratio
                            steered_idx_flat = top1_probs_index_steered.reshape(-1)
                            if i == 0:
                                wo_steering_idx_flat = top1_probs_index_wo_steering.reshape(-1)
                                unique_wo_steering_idx = torch.unique(wo_steering_idx_flat)
                                total_num = steered_idx_flat.shape[0]
                                unique_num_wo_steering = unique_wo_steering_idx.shape[0]
                                unique_ratio_wo_steering = unique_num_wo_steering / total_num
                                norm_related["unique_ratio_steered"].append(unique_ratio_wo_steering)
                                unique_token_str = tokenizer.decode(unique_wo_steering_idx)
                                norm_related["top1_probs"].append(p_orig_top1.detach().to(torch.float32).cpu().numpy().reshape(-1))
                            else:
                                unique_steered_idx = torch.unique(steered_idx_flat)
                                unique_token_str = tokenizer.decode(unique_steered_idx)
                                unique_num_steered = unique_steered_idx.shape[0]
                                total_num = steered_idx_flat.shape[0]
                                unique_ratio_steered = unique_num_steered / total_num
                                norm_related["unique_ratio_steered"].append(unique_ratio_steered)
                                norm_related["top1_probs"].append(p_new_top1.detach().to(torch.float32).cpu().numpy().reshape(-1))
                            
                            norm_related["delta_p_flip_mean"].append(float(np.mean(delta_p_flip[keep].detach().to(torch.float32).cpu().numpy())))
                            norm_related["delta_p_flip_std"].append(float(np.std(delta_p_flip[keep].detach().to(torch.float32).cpu().numpy())))
                            norm_related["entropy_diff_mean"].append(float(np.mean(entropy_diff[keep].detach().to(torch.float32).cpu().numpy())))
                            norm_related["entropy_diff_std"].append(float(np.std(entropy_diff[keep].detach().to(torch.float32).cpu().numpy())))
                    else:
                        raise ValueError(f"Invalid metric: {metric}")
                if metric == "lss":
                    lss = compute_line_shape_score(
                        lss_middle_states[1, :, :],
                        last_layer_hidden_states.reshape(-1, dim_middle_states)
                        .detach()
                        .to(torch.float32),
                        args.alpha_factor,
                    )
                    mean_lss = float(np.mean(lss))
                    std_lss = float(np.std(lss))
                    logger.info(f"Mean LSS: {mean_lss}")
                    logger.info(f"Std LSS: {std_lss}")
                    os.makedirs(f"weights/linearity/lss", exist_ok=True)
                    if args.remove_concept_vector:
                        torch.save(
                            lss,
                            f"assets/linearity/{model_name}/lss_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                    else:
                        torch.save(
                            lss,
                            f"assets/linearity/{model_name}/lss_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                elif metric == "lsr":
                    seq_diff_norm_cpu = seq_diff_norm.cpu().numpy()
                    mean_lsr = float(np.mean(seq_diff_norm_cpu))
                    std_lsr = float(np.std(seq_diff_norm_cpu))
                    logger.info(f"Mean LSR: {mean_lsr}")
                    logger.info(f"Std LSR: {std_lsr}")
                    if args.remove_concept_vector:
                        torch.save(
                            seq_diff_norm,
                            f"assets/linearity/{model_name}/lsr_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                    else:
                        torch.save(
                            seq_diff_norm,
                            f"assets/linearity/{model_name}/lsr_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                elif metric == "norm":
                    torch.save(
                        norm_related,
                        f"assets/linearity/{model_name}/norm_related_{concept_category_name}_layer{layer_idx}.pt",
                    )
                else:
                    raise ValueError(f"Invalid metric: {metric}")


def compute_line_shape_score(
    lss_final_states: torch.Tensor, initial_state: torch.Tensor, iteration_times: int
):
    """
    Compute Line-Shape Score.

    Args:
        lss_final_states: The final states.
        initial_state: The initial state.
        iteration_times: The iteration times.

    Returns:
        The Line-Shape Score.
    """
    diff = lss_final_states - initial_state
    diff_norm = diff.norm(p=2, dim=-1)
    diff_norm = torch.clamp(diff_norm, min=1e-8)
    return (iteration_times / diff_norm).detach().to(torch.float32).tolist()


def compute_lsr_middle_states(
    logits_with_hook: torch.Tensor,
    last_logits_with_hook: torch.Tensor,
    logits_without_hook: torch.Tensor,
    seq_diff_norm: torch.Tensor,
    i: int,
    alpha_factor: int,
    concept_vector_alpha: float,
    concept_vector: torch.Tensor,
    remove_concept_vector: bool = False,
):
    """
    Compute LSR (per token, streaming over layers, no full storage).
    Args:
        logits_with_hook: The logits with hook.
        last_logits_with_hook: The last logits with hook.
        logits_without_hook: The logits without hook.
        seq_diff_norm: The sequence difference norm.
        i: The index of the alpha.
        alpha_factor: The alpha factor.
        concept_vector_alpha: The alpha of the concept vector.
        concept_vector: The concept vector.
        remove_concept_vector: Whether to remove the concept vector.
    Returns:
        logits_with_hook: The logits with hook.
        seq_diff_norm: The sequence difference norm.
    """
    if remove_concept_vector:
        updated_concept_vector = concept_vector_alpha * concept_vector.to(
            device=logits_with_hook.device, dtype=logits_with_hook.dtype
        )
    else:
        updated_concept_vector = 0
    if i == alpha_factor - 1:
        diff = logits_with_hook - last_logits_with_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        seq_diff_norm += diff_norm
        overall_diff_norm = (
            logits_with_hook - logits_without_hook - updated_concept_vector
        )
        overall_diff_norm = torch.norm(overall_diff_norm, p=2, dim=-1)
        overall_diff_norm = torch.clamp(overall_diff_norm, min=1e-8)
        return logits_with_hook, seq_diff_norm / overall_diff_norm
    else:
        diff = logits_with_hook - last_logits_with_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        seq_diff_norm += diff_norm
        return logits_with_hook, seq_diff_norm


def compute_line_shape_score_middle_states(
    logits_with_hook: torch.Tensor,
    logits_without_hook: torch.Tensor,
    lss_middle_states: torch.Tensor,
    i: int,
    concept_vector_alpha: float,
    concept_vector: torch.Tensor,
    remove_concept_vector: bool = False,
):
    """
    Compute Line-Shape Score (per token, streaming over layers, no full storage).

    Args:
        logits_with_hook: The logits with hook.
        logits_without_hook: The logits without hook.
        lss_middle_states: The middle states.
        i: The index of the alpha.
        concept_vector_alpha: The alpha of the concept vector.
        concept_vector: The concept vector.
        remove_concept_vector: Whether to remove the concept vector.
    Returns:
        lss_middle_states: The middle states.
    """
    if remove_concept_vector:
        updated_concept_vector = concept_vector_alpha * concept_vector
    else:
        updated_concept_vector = 0
    if i == 0:
        lss_middle_states[0, :, :] = logits_without_hook
        diff = logits_with_hook - logits_without_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = (
            normalized_diff + lss_middle_states[0, :, :] - updated_concept_vector
        )
        lss_middle_states[0, :, :] = logits_without_hook
    else:
        diff = logits_with_hook - lss_middle_states[0, :, :] - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = normalized_diff + lss_middle_states[1, :, :]
        lss_middle_states[0, :, :] = logits_with_hook
    return lss_middle_states




if __name__ == "__main__":
    linearity()
