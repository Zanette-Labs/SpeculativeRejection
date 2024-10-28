import torch
import transformers
from typing import Any


def get_cuda_devices() -> list[str]:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise Exception("No GPUs detected.")
    cuda_devices = [f"cuda:{i}" for i in range(num_gpus)]
    return cuda_devices


def get_device_specific_map(device: str) -> dict[str, Any]:
    split_device = device.split(":")
    if len(split_device) == 1:
        return device_map_template(0)
    device_num = int(split_device[-1])
    return device_map_template(device_num)


def device_map_template(device_num: int) -> dict[str, Any]:
    return {
        "": device_num,
    }


def swap_models(
    accelerator_owner: str,
    generation_model: transformers.LlamaForCausalLM,
    reward_model,
) -> str:
    if accelerator_owner == "generation_model":
        print("Moving generation model to CPU, reward model to GPU...")
        device = generation_model.device
        generation_model.to("cpu")
        try:
            reward_model.to(device)
        except:
            reward_model.device = device
            reward_model.model.to(device)
        return "reward_model"
    elif accelerator_owner == "reward_model":
        print("Moving reward model to CPU, generation model to GPU...")
        device = reward_model.device
        try:
            reward_model.to("cpu")
        except:
            reward_model.device = torch.device("cpu")
            reward_model.model.to("cpu")
        generation_model.to(device)
        return "generation_model"
    else:
        raise Exception("Invalid accelerator owner...")
