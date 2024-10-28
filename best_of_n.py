import torch
from generator import Generator
from utils.generation_utils import (
    get_input_encoding,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)
from utils.trajectory import Trajectory
from utils.reward_utils import (
    compute_scores,
)
from engine.models.llm import LLM


class BestOfN(Generator):
    def generate(self, prompt: str, prompt_dict: dict | None = None):
        self.prepare_generation(prompt_dict)
        self.clock.reset()
        self.clock.start()
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )
        templated_prompts = [self.templated_prompt] * self.args.batch_size
        batch_encoding = get_input_encoding(
            templated_prompts,
            self.generation_model,
            self.generation_tokenizer,
        )
        self.clock.stop("tokenization")
        self.clock.start()

        # set max tokens for engine
        input_length = batch_encoding.input_ids.shape[-1]
        max_all_tokens = min(
            self.args.max_tokens, self.args.max_gen_tokens + input_length
        )
        # decide init bsz for engine
        if isinstance(self.generation_model, LLM):
            self.generation_model.max_tokens = max_all_tokens
            # batch_size = 200 # self.generation_model.get_batch_size(max_seq=max_all_tokens)
            # templated_prompts = [self.templated_prompt] * batch_size
            # batch_encoding = get_input_encoding(
            #     templated_prompts,
            #     self.generation_model,
            #     self.generation_tokenizer,
            # )
            batch_size = self.args.batch_size
            gen_len = max_all_tokens - input_length
            try:
                full_generation = self.generation_model.generate(
                    input_ids=batch_encoding.input_ids,
                    batch_size=batch_size,
                    gen_len=gen_len,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    temperature=self.args.temperature,
                )
            except RuntimeError as e:
                print(e)
                # reduce batch size and then try again
                bsz1 = batch_size // 2
                bsz2 = batch_size - bsz1
                full_generation_1 = self.generation_model.generate(
                    input_ids=batch_encoding.input_ids[:bsz1],
                    batch_size=bsz1,
                    gen_len=gen_len,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    temperature=self.args.temperature,
                )
                full_generation_2 = self.generation_model.generate(
                    input_ids=batch_encoding.input_ids[bsz1:],
                    batch_size=bsz2,
                    gen_len=gen_len,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    temperature=self.args.temperature,
                )
                full_generation = torch.cat(
                    [full_generation_1, full_generation_2], dim=0
                )

        else:
            full_generation: torch.LongTensor = self.generation_model.generate(
                input_ids=batch_encoding.input_ids,
                attention_mask=batch_encoding.attention_mask,
                max_length=max_all_tokens,
                eos_token_id=self.terminators,
                pad_token_id=self.generation_tokenizer.pad_token_id,
                do_sample=True,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
            )
        self.clock.stop("generation pass")
        print(f"full_generation shape: {full_generation.shape}")
        self.clock.start()
        padded_output_texts = get_output_texts(
            full_generation,
            self.templated_prompt,
            self.generation_tokenizer,
            skip_special_tokens=False,
        )
        unpadded_output_texts = unpad_output_texts(
            padded_output_texts, self.stop_tokens
        )
        self.clock.stop("decoding")
        self.clock.start()
        reward_list = compute_scores(
            prompt,
            unpadded_output_texts,
            self.reward_model_name,
            self.reward_tokenizer,
            self.reward_model,
        )
        self.clock.stop("reward pass")
        self.clock.start()
        for padded_output_text, unpadded_output_text, score in zip(
            padded_output_texts, unpadded_output_texts, reward_list
        ):
            trajectory = Trajectory(
                self.prompt,
                self.templated_prompt,
                padded_output_text,
                unpadded_output_text,
                score,
            )
            self.trajectories.append(trajectory)
        self.clock.stop("finish")
        self.post_generation()
