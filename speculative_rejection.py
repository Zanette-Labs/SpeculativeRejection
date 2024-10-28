from generator import Generator
from utils.generation_utils import (
    get_input_encoding,
    get_memory_constrained_generation,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)
from utils.read_write_utils import write_to_disk
from utils.reward_utils import compute_scores
from utils.sbon_utils import get_memory_constrained_batch_size
from utils.trajectory import Trajectory
from utils.validation_utils import validate_alpha
import torch, gc
from engine.models.llm import LLM


class SpeculativeRejection(Generator):
    def generate(self, prompt: str, prompt_dict: dict | None = None) -> None:
        if prompt_dict is None:
            prompt_dict = prompt
        self.prepare_generation(prompt_dict)
        self.clock.reset()
        self.clock.start()
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )
        alpha: float = self.args.alpha
        validate_alpha(alpha)
        batch_encoding = get_input_encoding(
            [self.templated_prompt],
            self.generation_model,
            self.generation_tokenizer,
        )
        input_length = batch_encoding.input_ids.shape[-1]
        batch_size = get_memory_constrained_batch_size(input_length, self.args.llm_name)

        # set max tokens for engine
        max_all_tokens = min(
            self.args.max_tokens, self.args.max_gen_tokens + input_length
        )
        # decide init bsz for engine
        if isinstance(self.generation_model, LLM):
            self.generation_model.max_tokens = max_all_tokens
            batch_size = min(int(batch_size * 2), 1000)
            self.generation_model.tokenizer = self.generation_tokenizer

            while True:
                gen_len = self.generation_model.get_gen_len(
                    batch_size=batch_size, cur_len=input_length
                )
                if gen_len >= 8:
                    break
                batch_size = int(batch_size * 0.9)

        current_generations = [self.templated_prompt] * batch_size
        self.clock.stop("hyperparameter selection")
        print(f"input_length: {input_length}")
        self.clock.start()
        current_length = input_length

        while current_length < max_all_tokens:
            if isinstance(self.generation_model, LLM):
                batch_encoding = self.generation_model.batch_encode(current_generations)
            else:
                batch_encoding = get_input_encoding(
                    current_generations,
                    self.generation_model,
                    self.generation_tokenizer,
                )
            self.clock.stop("tokenization")
            self.clock.start()
            try:
                if isinstance(self.generation_model, LLM):
                    batch_size = batch_encoding.shape[0]
                    cur_len = batch_encoding.shape[1]
                    gen_len = self.generation_model.get_gen_len(
                        batch_size=batch_size, cur_len=cur_len
                    )
                    if gen_len < 1:
                        gen_len = 1
                    assert gen_len > 0
                    partial_generation = self.generation_model.generate(
                        input_ids=batch_encoding,
                        batch_size=batch_size,
                        gen_len=gen_len,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        temperature=self.args.temperature,
                    )
                else:
                    partial_generation = get_memory_constrained_generation(
                        self.generation_model,
                        batch_encoding.input_ids,
                        self.terminators,
                        self.generation_tokenizer.pad_token_id,
                        self.args,
                    )
            except Exception as e:
                print(e)
                write_to_disk(
                    self.all_data,
                    "./output_crashes",
                    self.initial_memory,
                    self.args.pretty_print_output,
                    self.args.record_memory,
                    force_dump=True,
                )
                raise Exception("Memory error occurred during generation")
            current_length = partial_generation.shape[-1]
            self.clock.stop(
                f"generation - partial_generation.shape {partial_generation.shape}"
            )
            print(f"partial_generation shape: {partial_generation.shape}")

            self.clock.start()
            padded_output_texts = get_output_texts(
                    partial_generation,
                    self.templated_prompt,
                    self.generation_tokenizer,
                    skip_special_tokens=False,
                )
            unpadded_output_texts = unpad_output_texts(
                    padded_output_texts, self.stop_tokens
                )
            self.clock.stop(f"decoding - current_length {current_length}")
            
            if self.is_self_reward:
                reward_list = self.generation_model.self_evaluate(partial_generation)
            else:
                self.clock.start()
                reward_list = compute_scores(
                    prompt,
                    unpadded_output_texts,
                    self.reward_model_name,
                    self.reward_tokenizer,
                    self.reward_model,
                )
                self.clock.stop(f"reward - current_length {current_length}")
            
            self.clock.start()
            current_trajectories: list[Trajectory] = [
                Trajectory(
                    self.prompt,
                    self.templated_prompt,
                    padded_output_text,
                    unpadded_output_text,
                    score,
                )
                for padded_output_text, unpadded_output_text, score in zip(
                    padded_output_texts, unpadded_output_texts, reward_list
                )
            ]
            current_generations = self.perform_speculative_rejection(
                current_trajectories, alpha
            )
            if len(current_generations) == 0:
                break
            self.clock.stop(f"speculative rejection - current_length {current_length}")
            self.clock.start()
        self.trajectories = (
            self.trajectories + current_trajectories + self.finished_trajectories
        )
        self.clock.stop("finish")
        self.post_generation()

    def perform_speculative_rejection(
        self,
        current_trajectories: list[Trajectory],
        alpha: float,
    ) -> list[str]:
        previous_finished_trajectories = [
            trajectory for trajectory in self.trajectories if trajectory.finished
        ]
        self.finished_trajectories += previous_finished_trajectories
        trajectories_to_rank = previous_finished_trajectories + current_trajectories
        trajectories_to_rank.sort(key=lambda trajectory: trajectory.score, reverse=True)
        keep_fraction = 1.0 - alpha
        keep_amount = int(round(keep_fraction * len(trajectories_to_rank)))
        self.trajectories = trajectories_to_rank[:keep_amount]
        generating_trajectories = [
            trajectory for trajectory in self.trajectories if not trajectory.finished
        ]
        current_generations = [
            trajectory.templated_prompt + trajectory.unpadded_output_text
            for trajectory in generating_trajectories
        ]
        return current_generations
