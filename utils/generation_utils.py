import torch
import transformers

transformers.logging.set_verbosity_error()

from engine.utils.sampling import sample, norm_logits


def get_generation_tokenizer(
    llm_name: str, local_files_only=True
) -> transformers.PreTrainedTokenizerFast:
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(
        llm_name,
        padding_side="left",
        use_fast=True,
        legacy=False,
        local_files_only=local_files_only,
    )
    generation_tokenizer.pad_token = generation_tokenizer.eos_token
    generation_tokenizer.padding_side = "left"
    return generation_tokenizer


def get_terminators(
    llm_name: str, generation_tokenizer: transformers.PreTrainedTokenizerFast
) -> list[int | None]:
    if "Llama" in llm_name:
        terminators = [
            generation_tokenizer.eos_token_id,
            generation_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    else:
        terminators = [generation_tokenizer.eos_token_id]
    return terminators


def get_generation_model(
    llm_name: str, device: str, local_files_only=True
) -> transformers.LlamaForCausalLM:
    try:
        generation_model = transformers.AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
        ).to(device)
    except:
        print("WARNING: could not load model with flash attention - trying without...")
        generation_model = transformers.AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        ).to(device)
    return generation_model


def get_templated_prompt(
    prompt: str,
    llm_name: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> str:
    if "Instruct" in llm_name:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    elif any(s in llm_name for s in ["sft10k", "alpaca-7b", "dpo", "ppo", "human"]):
        templated_prompt = f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    elif "llama-2" in llm_name.lower():
        templated_prompt = f"<s>[INST]\n{prompt} [/INST]"
    else:
        templated_prompt = generation_tokenizer.bos_token + prompt
    return templated_prompt


def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding


def get_output_texts(
    generation_ids: torch.LongTensor,
    prompt: str,
    generation_tokenizer,
    skip_special_tokens: bool = False,
) -> list[str]:
    generation_texts = generation_tokenizer.batch_decode(
        generation_ids, skip_special_tokens=skip_special_tokens
    )
    output_texts: list[str] = []
    for generation_text in generation_texts:
        generation_text = generation_text.replace(
            "<s> [INST]", "<s>[INST]"
        )  # for llama-2-chat-hf
        split_pieces = generation_text.split(prompt)
        # print(generation_ids)
        # print(generation_tokenizer.decode(generation_ids[0]))
        # print(prompt)
        # print(generation_text)
        # # write to txt:
        # with open('output.txt', 'w') as f:
        #     f.write(generation_text)
        # with open('output2.txt', 'w') as f:
        #     f.write(prompt)
        try:
            assert (
                prompt in generation_text
            ), f"prompt: {prompt} | generation_text: {generation_text}"
            assert (
                len(split_pieces) > 1
            ), f"prompt: {prompt} | generation_text: {generation_text}, {len(split_pieces)}, {split_pieces}"
            output_text = prompt.join(split_pieces[1:])
        except:
            output_text = generation_text[len(prompt) :]
        output_texts.append(output_text)
    return output_texts


def unpad_output_texts(output_texts: list[str], stop_tokens: list[str]) -> list[str]:
    unpadded_texts: list[str] = []
    for output_text in output_texts:
        for stop_token in stop_tokens:
            output_text = output_text.split(stop_token)[0]
        unpadded_texts.append(output_text)
    return unpadded_texts


@torch.inference_mode()
def get_memory_constrained_generation(
    generation_model: transformers.LlamaForCausalLM,
    generation_ids: torch.LongTensor,
    terminators: list[int | None],
    pad_token_id: int | None,
    args,
) -> torch.LongTensor:

    past_key_values = None
    batch_size = generation_ids.shape[0]
    finished_generations = torch.zeros(batch_size).bool().to(generation_model.device)
    while generation_ids.shape[-1] < args.max_tokens:
        try:
            out_dict = generation_model.generate(
                generation_ids,
                pad_token_id=pad_token_id,
                max_new_tokens=1,
                eos_token_id=terminators,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
            )
            if "past_key_values" in out_dict:
                past_key_values = out_dict.past_key_values
            else:
                raise Exception("past_key_values (KV cache) not found in model output")
            generation_ids = out_dict.sequences
        except torch.cuda.OutOfMemoryError:
            break
        just_finished = generation_ids[:, -1] == pad_token_id
        finished_generations = finished_generations | just_finished
        if torch.all(finished_generations):
            break
    return generation_ids
