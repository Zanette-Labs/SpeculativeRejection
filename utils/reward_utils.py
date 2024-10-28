import numpy as np
import transformers
import torch
from .alpaca_farm.reward_model import RewardModel, RewardConfig
from .generation_utils import get_templated_prompt


def is_mistral_type(reward_model_name: str) -> bool:
    return (
        "RM-Mistral-7B" in reward_model_name
        or "FsfairX-LLaMA3-RM-v0.1" in reward_model_name
    )


def get_reward_tokenizer(reward_model_name: str, local_files_only: bool = True):
    if "ArmoRM-Llama3-8B-v0.1" in reward_model_name:
        reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
            reward_model_name,
            use_fast=True,
            legacy=False,
            local_files_only=local_files_only,
        )
    else:
        if "reward-model" in reward_model_name:
            reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "hmomin/sft10k",
                use_fast=True,
                padding_side="left",
                legacy=False,
                local_files_only=local_files_only,
            )
        else:
            reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
                reward_model_name,
                use_fast=True,
                padding_side="left",
                legacy=False,
                local_files_only=local_files_only,
            )
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.padding_side = "left"
    return reward_tokenizer


def get_reward_model(
    reward_model_name: str, reward_tokenizer, device: str, local_files_only: bool = True
):
    if is_mistral_type(reward_model_name):
        reward_model = transformers.pipeline(
            "sentiment-analysis",
            model=reward_model_name,
            tokenizer=reward_tokenizer,
            device=device,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
        )
    elif "ArmoRM-Llama3-8B-v0.1" in reward_model_name:
        reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
        ).to(device)
    elif "Eurus-RM-7b" in reward_model_name:
        reward_model = transformers.AutoModel.from_pretrained(
            reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
        ).to(device)
    elif (
        "reward-model-human" in reward_model_name
        or "reward-model-sim" in reward_model_name
    ):
        reward_model = RewardModel.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            mixed_precision="bf16",
            flash_attn=True,
            config=RewardConfig(
                backbone_model_name_or_path="hmomin/sft10k",
                local_files_only=local_files_only,
            ),
            local_files_only=local_files_only,
        ).to(device)
    else:
        raise Exception(f"Invalid reward model name: {reward_model_name}")
    return reward_model


def create_conversation_object(prompt: str, response: str = "") -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return conversation


def get_texts_for_scoring(
    generation_texts: list[str], input_length: int, stop_tokens: list[str]
) -> list[str]:
    output_texts: list[str] = []
    for generation_text in generation_texts:
        output_text = generation_text[input_length:]
        for stop_token in stop_tokens:
            output_text = output_text.replace(stop_token, "")
        output_texts.append(output_text)
    return output_texts


def compute_scores(
    question: str,
    output_texts: list[str],
    reward_model_name: str,
    reward_tokenizer,
    reward_model,
) -> list[float]:
    reward_tokens = get_reward_tokens(
        question,
        output_texts,
        reward_model_name,
        reward_tokenizer,
        reward_model.device,
    )
    # print(f"reward_tokens: {reward_tokens}")
    reward_list = get_rewards(reward_model_name, reward_model, reward_tokens)

    if reward_list is None:
        raise Exception("Could not compute scores...")
    return reward_list


def get_reward_tokens(
    question: str,
    output_texts: list[str],
    reward_model_name: str,
    reward_tokenizer,
    device: torch.device,
) -> torch.Tensor | list[str]:
    if is_mistral_type(reward_model_name):
        conversation_objects: list[list[dict[str, str]]] = get_conversation_objects(
            question, output_texts
        )
        test_texts = get_test_texts(conversation_objects, reward_tokenizer)
        return test_texts
    elif "ArmoRM-Llama3-8B-v0.1" in reward_model_name:
        conversation_objects: list[list[dict[str, str]]] = get_conversation_objects(
            question, output_texts
        )
        reward_tokens = reward_tokenizer.apply_chat_template(
            conversation_objects, return_tensors="pt", padding=True, tokenize=True
        ).to(device)
        return reward_tokens
    elif "Eurus-RM-7b" in reward_model_name:
        tokenizer_inputs = get_eurus_texts(question, output_texts)
        reward_tokens = reward_tokenizer(
            tokenizer_inputs, return_tensors="pt", padding=True
        ).to(device)
        return reward_tokens
    elif (
        "reward-model-human" in reward_model_name
        or "reward-model-sim" in reward_model_name
    ):
        templated_question = get_templated_prompt(question, "sft10k", reward_tokenizer)
        sequences = [templated_question + output_text for output_text in output_texts]
        reward_tokens = reward_tokenizer(
            sequences,
            return_tensors="pt",
            # padding="max_length",
            padding=True,
            max_length=reward_tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        return reward_tokens
    else:
        raise Exception(f"Invalid reward model name: {reward_model_name}")


def get_conversation_objects(
    question: str, output_texts: list[str]
) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    for output_text in output_texts:
        conversations.append(create_conversation_object(question, output_text))
    return conversations


def get_test_texts(
    conversations: list[list[dict[str, str]]],
    tokenizer,
) -> list[str]:
    test_texts: list[str] = []
    for conversation in conversations:
        tokenization: str = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        ).replace(tokenizer.bos_token, "")
        test_texts.append(tokenization)
    return test_texts


def get_armo_texts(question: str, output_texts: list[str]) -> list[str]:
    templated_texts = [
        f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        for output_text in output_texts
    ]
    return templated_texts


def get_eurus_texts(question: str, output_texts: list[str]) -> list[str]:
    tokenizer_inputs: list[str] = []
    for output_text in output_texts:
        some_input = f"[INST] {question} [/INST]{output_text}"
        tokenizer_inputs.append(some_input)
    return tokenizer_inputs


def get_rewards(
    reward_model_name: str,
    reward_model,
    reward_tokens: torch.Tensor | list[str],
) -> list[float] | None:
    if is_mistral_type(reward_model_name):
        # NOTE: batch_size should be very large to ensure batching with pipelines

        rebatched_tokens = rebatch_tokens_texts(reward_tokens)
        reward_list: list[float] = []
        for tks in rebatched_tokens:
            pipe_kwargs = {
                "top_k": None,
                "function_to_apply": "none",
                "batch_size": len(tks)
            }
            pipe_outputs = reward_model(tks, **pipe_kwargs)
            reward = [output[0]["score"] for output in pipe_outputs]
            reward_list.extend(reward)

    elif "ArmoRM-Llama3-8B-v0.1" in reward_model_name:
        # print(reward_tokens.shape, flush=True)
        rebatched_tokens = rebatch_tokens_tensor(reward_tokens)
        reward_list: list[float] = []
        for tks in rebatched_tokens:
            reward = reward_model(tks).score.squeeze().tolist()
            if type(reward) == float:
                reward = [reward]
            reward_list.extend(reward)
        # print(len(reward_list))
    elif "Eurus-RM-7b" in reward_model_name:
        # NOTE: break up the batch into smaller chunks to avoid out-of-memory errors
        rebatched_tokens = rebatch_tokens_for_eurus(reward_tokens)
        reward_list: list[float] = []
        for token_dict in rebatched_tokens:
            rewards = reward_model(**token_dict).squeeze().tolist()
            if type(rewards) == float:
                rewards = [rewards]
            reward_list.extend(rewards)
    elif (
        "reward-model-human" in reward_model_name
        or "reward-model-sim" in reward_model_name
    ):
        # FIXME: break up the batch into smaller chunks to avoid out-of-memory errors (?)
        # rebatched_tokens = rebatch_tokens_for_eurus(reward_tokens)
        # reward_list: list[float] = []
        # for token_dict in rebatched_tokens:
        #     rewards = reward_model(**token_dict).squeeze().tolist()
        #     if type(rewards) == float:
        #         rewards = [rewards]
        #     reward_list.extend(rewards)
        # try:
        #     outputs: tuple[torch.Tensor] = reward_model(
        #         input_ids=reward_tokens.input_ids,
        #         attention_mask=reward_tokens.attention_mask,
        #         return_dict=False,
        #     )
        #     reward_list = outputs[0].squeeze().tolist()
        # except Exception as e:
        #     # break up the batch into smaller chunks to avoid out-of-memory errors (?)
        rebatched_tokens = rebatch_tokens_for_farm(reward_tokens)
        reward_list: list[float] = []
        for token_dict in rebatched_tokens:
            rewards = (
                reward_model(**token_dict, return_dict=False)[0].squeeze().tolist()
            )
            if type(rewards) == float:
                rewards = [rewards]
            reward_list.extend(rewards)
    else:
        raise Exception(f"Invalid reward model name: {reward_model_name}")
    if type(reward_list) == float:
        reward_list = [reward_list]
    return reward_list


def rebatch_tokens_for_farm(
    reward_tokens: transformers.BatchEncoding,
) -> list[dict[str, torch.Tensor]]:
    input_ids: torch.Tensor = reward_tokens.input_ids
    attention_mask: torch.Tensor = reward_tokens.attention_mask
    token_length = input_ids.shape[0] * input_ids.shape[1]
    num_chunks = int(np.ceil(token_length / 81920))
    rebatched_tokens: list[dict[str, torch.Tensor]] = []
    step_size = max(1, int(np.floor(input_ids.shape[0] / num_chunks)))
    for idx in range(0, input_ids.shape[0], step_size):
        rebatched_tokens.append(
            {
                "input_ids": input_ids[idx : idx + step_size, :],
                "attention_mask": attention_mask[idx : idx + step_size, :],
            }
        )
    return rebatched_tokens


def rebatch_tokens_for_eurus(
    reward_tokens: transformers.BatchEncoding,
) -> list[dict[str, torch.Tensor]]:
    input_ids: torch.Tensor = reward_tokens.input_ids
    attention_mask: torch.Tensor = reward_tokens.attention_mask
    token_length = input_ids.shape[-1]
    num_chunks = int(np.ceil(token_length / 8_000))
    rebatched_tokens: list[dict[str, torch.Tensor]] = []
    step_size = max(1, int(np.floor(input_ids.shape[0] / num_chunks)))
    for idx in range(0, input_ids.shape[0], step_size):
        rebatched_tokens.append(
            {
                "input_ids": input_ids[idx : idx + step_size, :],
                "attention_mask": attention_mask[idx : idx + step_size, :],
            }
        )
    return rebatched_tokens


def rebatch_tokens_tensor(
    input_ids: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    token_length = input_ids.shape[-1] * input_ids.shape[0]
    num_chunks = int(np.ceil(token_length / 300_000))
    rebatched_tokens = []
    step_size = max(1, int(np.floor(input_ids.shape[0] / num_chunks)))
    for idx in range(0, input_ids.shape[0], step_size):
        rebatched_tokens.append(
            input_ids[idx : idx + step_size, :],
        )
    return rebatched_tokens


def rebatch_tokens_texts(
    input_ids: list,
) -> list[dict[str, torch.Tensor]]:
    token_length = len(input_ids) * len(input_ids[0])
    num_chunks = int(np.ceil(token_length / 100_000))
    rebatched_tokens = []
    step_size = max(1, int(np.floor(len(input_ids) / num_chunks)))
    for idx in range(0, len(input_ids), step_size):
        rebatched_tokens.append(
            input_ids[idx : idx + step_size],
        )
    return rebatched_tokens
