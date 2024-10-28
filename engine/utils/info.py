import torch


def gpu_memory(device) -> str:
    current_device = torch.cuda.current_device()

    memory_info = torch.cuda.mem_get_info(current_device)
    free_memory = memory_info[0] / (1024 ** 3)
    total_memory = memory_info[1] / (1024 ** 3)
    ret = f"{round((total_memory - free_memory) , 3)} / {round(total_memory, 3)} GB"
    return ret


def gpu_free_memory(device) -> str:
    current_device = torch.cuda.current_device()

    memory_info = torch.cuda.mem_get_info(current_device)
    free_memory = memory_info[0] / (1024 ** 3)
    return free_memory
