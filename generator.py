import torch
import transformers
from accelerate import PartialState
from argparse import Namespace
from clock import Clock
from copy import deepcopy
from typing import Any
from utils.generation_utils import (
    get_generation_model,
    get_generation_tokenizer,
    get_terminators,
)
from utils.read_write_utils import save_data
from utils.reward_utils import get_reward_model, get_reward_tokenizer
from utils.trajectory import Trajectory
from utils.validation_utils import (
    get_full_model_name,
    validate_llm_name,
    validate_reward_model_name,
)

from engine.models.llm import LLM


class Generator(object):
    def __init__(
        self,
        args: Namespace,
        distributed_state: PartialState,
    ) -> None:
        validate_llm_name(args.llm_name)
        validate_reward_model_name(args.reward_model_name)
        llm_name = get_full_model_name(args.model_dir, args.llm_name)
        reward_model_name = get_full_model_name(args.model_dir, args.reward_model_name)

        self.llm_name = llm_name
        self.reward_model_name = reward_model_name

        if self.llm_name == self.reward_model_name:
            self.is_self_reward = True
        else:
            self.is_self_reward = False

        self.args = args
        self.distributed_state = distributed_state
        self.clock = Clock()

        self.process_seed = args.seed + distributed_state.local_process_index
        print(f"DEVICE: {distributed_state.device}")
        transformers.set_seed(self.process_seed)

        self.generation_tokenizer = get_generation_tokenizer(
            llm_name, args.local_files_only
        )
        self.stop_tokens = ["</s>", "<|end_of_text|>", "<|eot_id|>"]
        self.terminators = get_terminators(llm_name, self.generation_tokenizer)

        if args.speculative_rejection:
            self.generation_model = LLM(
                llm_name,
                device=distributed_state.device,
                local_files_only=args.local_files_only,
            )
        else:
            self.generation_model = get_generation_model(
                llm_name,
                distributed_state.device,
                local_files_only=args.local_files_only,
            )

        if not self.is_self_reward:
            self.reward_tokenizer = get_reward_tokenizer(
                reward_model_name, local_files_only=args.local_files_only
            )
            self.reward_model = get_reward_model(
                reward_model_name,
                self.reward_tokenizer,
                distributed_state.device,
                local_files_only=args.local_files_only,
            )

        self.templated_prompt = ""

    def prepare_generation(self, prompt_dict: dict | None = None) -> None:
        self.trajectories: list[Trajectory] = []
        self.finished_trajectories: list[Trajectory] = []
        self.all_data: list[dict[str, Any]] = [deepcopy(vars(self.args))]
        self.all_data[0]["process_seed"] = self.process_seed
        self.all_data[0]["prompt"] = prompt_dict
        self.initialize_memory_stats()

    def initialize_memory_stats(self) -> None:
        self.initial_memory = torch.cuda.memory.memory_allocated()
        if self.args.record_memory and self.distributed_state.is_main_process:
            torch.cuda.memory.reset_accumulated_memory_stats()
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context=None,
                stacks="python",
            )

    def post_generation(self) -> None:
        elapsed_time = self.clock.get_time()
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        self.all_data[0]["elapsed_sec"] = elapsed_time
        self.all_data[0]["clock"] = self.clock.get_chunks()
        save_data(self.all_data, self.trajectories)
