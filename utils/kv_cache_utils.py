import torch
import transformers


def move_kv_cache(
    past_key_values: transformers.DynamicCache, device: str | torch.device
) -> None:
    move_tensor_list(past_key_values.key_cache, device)
    move_tensor_list(past_key_values.value_cache, device)


def move_tensor_list(
    tensor_list: list[torch.Tensor], device: str | torch.device
) -> None:
    for idx, tensor in enumerate(tensor_list):
        tensor_list[idx] = tensor.to(device)
        del tensor


def prune_kv_cache(
    past_key_values: transformers.DynamicCache, indices: list[int]
) -> None:
    prune_tensor_list(past_key_values.key_cache, indices)
    prune_tensor_list(past_key_values.value_cache, indices)


def prune_tensor_list(tensor_list: list[torch.Tensor], indices: list[int]) -> None:
    for idx, tensor in enumerate(tensor_list):
        tensor_list[idx] = tensor[indices, :, :, :]
