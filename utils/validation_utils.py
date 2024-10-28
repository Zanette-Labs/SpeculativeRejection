def validate_llm_name(llm_name: str) -> None:
    return  # NOTE: changing this is getting annoying... just let it break the normal way
    valid_llm_names = [
        "gpt2",
        "sft10k",
        "dpo-sft10k",
        "ppo-human",
        "Meta-Llama-3-8B",
        "Meta-Llama-3-8B-Instruct",
        "Llama-2-7b-chat-hf",
        "Mistral-7B-v0.3",
    ]
    if llm_name not in valid_llm_names:
        raise Exception(
            f"Invalid LLM name - '{llm_name}' not found in {valid_llm_names}."
        )


def validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 1.0):
        raise Exception("args.alpha expected to be in [0.0, 1.0)")


def validate_reward_model_name(reward_model_name: str) -> None:
    return  # NOTE: changing this is getting annoying... just let it break the normal way
    valid_reward_models = [
        "reward-model-human",
        "reward-model-sim",
        "RM-Mistral-7B",
        "FsfairX-LLaMA3-RM-v0.1",
        "ArmoRM-Llama3-8B-v0.1",
        "Eurus-RM-7b",
    ]
    if reward_model_name not in valid_reward_models:
        raise Exception(
            f"Invalid reward model name - '{reward_model_name}' not found in {valid_reward_models}."
        )


BASENAME2HF = {
    "sft10k": "hmomin/sft10k",
    "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "RM-Mistral-7B": "weqweasdas/RM-Mistral-7B",
    "ArmoRM-Llama3-8B-v0.1": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "FsfairX-LLaMA3-RM-v0.1": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
    "reward-model-human": "hmomin/reward-model-human",
    "reward-model-sim": "hmomin/reward-model-sim",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
}


def get_full_model_name(model_dir: str, model_basename: str) -> str:
    if model_dir is None or model_dir == "":
        if model_basename in BASENAME2HF:
            print(f"loading model from {BASENAME2HF[model_basename]}")
            return BASENAME2HF[model_basename]
        else:
            raise Exception(f"Model directory not provided for {model_basename}")
    print(f"loading model from {model_dir}/{model_basename}")
    return f"{model_dir}/{model_basename}"
