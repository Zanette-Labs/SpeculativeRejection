import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# NOTE: the following environment variables are set to avoid timeouts in NCCL
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT_MS"] = str(1000 * 60 * 60 * 3)  # ms * s * m * h

import argparse
import gc
import secrets
import torch
from accelerate import PartialState
from accelerate.utils import gather_object, InitProcessGroupKwargs
from best_of_n import BestOfN
from datetime import timedelta
from pprint import pprint
from speculative_rejection import SpeculativeRejection
from utils.read_write_utils import (
    create_output_folder,
    get_generation_prompts,
    write_to_disk,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filename",
        help="relative filename containing sample prompts",
        type=str,
        default="./datasets/alpaca_farm_100.json",
    )
    parser.add_argument(
        "--output_folder",
        help="folder name of output files",
        type=str,
        default="./output_test",
    )
    parser.add_argument(
        "--model_dir",
        help="directory containing model files - leave as '' to instantiate from huggingface",
        type=str,
        default="",
    )
    parser.add_argument(
        "--llm_name", help="model basename for generation", type=str, required=True
    )
    parser.add_argument(
        "--reward_model_name",
        help="model basename for scoring",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--speculative_rejection",
        help="use speculative rejection for generation?",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--alpha",
        help="fraction of trajectories (finished or generating) to reject on each speculative rejection pass",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--max_tokens",
        help="maximum number of tokens to generate per trajectory",
        type=int,
        default=2_048,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size to use for best-of-N - ignored when using speculative rejection",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        help="random seed for transformers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top_k",
        help="top-k parameter for generation model",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--top_p",
        help="top-p parameter for generation model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pretty_print_output",
        help="should output file be easily human-readable?",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--record_memory",
        help="whether to profile memory usage during execution",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--local_files_only",
        help="whether to use local_files_only for HF models",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_gen_tokens",
        help="maximum number of tokens to generate per trajectory (w/o prompt)",
        type=int,
        default=2_048,
    )
    parser.add_argument(
        "--temperature",
        help="temperature parameter for generation model",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3)).to_kwargs()
    distributed_state = PartialState(**kwargs)
    args = get_args()
    pprint(vars(args))

    generator = (
        SpeculativeRejection(args, distributed_state)
        if args.speculative_rejection
        else BestOfN(args, distributed_state)
    )

    generation_prompts = get_generation_prompts(args)
    output_folder = create_output_folder(args)

    latency_list = []
    while len(generation_prompts) > 0:
        print(f"Number of prompts remaining: {len(generation_prompts)}", flush=True)
        prompt_dict = generation_prompts[0]
        pprint(prompt_dict)
        prompt: str = prompt_dict["prompt"]

        generator.generate(prompt, prompt_dict=prompt_dict)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        distributed_state.wait_for_everyone()
        all_data_gather = gather_object(generator.all_data)
        latency_list.append(all_data_gather[0]["elapsed_sec"])
        if distributed_state.is_main_process:
            write_to_disk(
                all_data_gather,
                output_folder,
                generator.initial_memory,
                args.pretty_print_output,
                args.record_memory,
            )
        distributed_state.wait_for_everyone()
        generation_prompts = get_generation_prompts(args)
    print("DONE")


if __name__ == "__main__":
    with torch.no_grad():
        main()
