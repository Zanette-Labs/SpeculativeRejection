import argparse
import json
import os
import re
import torch
from .trajectory import Trajectory
from typing import Any


def create_output_folder(args: argparse.Namespace) -> str:
    output_folder_name: str = args.output_folder
    if not os.path.exists(output_folder_name):
        os.mkdir(output_folder_name)
    return output_folder_name


def get_generation_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_filename = args.data_filename
    output_folder = args.output_folder
    with open(data_filename, "r") as f:
        generation_prompts: list[dict[str, Any]] = json.load(f)
    remaining_prompts = remove_generated_prompts(generation_prompts, output_folder)
    return remaining_prompts


def remove_generated_prompts(
    generation_prompts: list[dict[str, Any]], output_folder: str
) -> list[dict[str, Any]]:
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    generated_prompt_files = os.listdir(output_folder)
    generated_prompt_indices: list[int] = []
    for generated_filename in generated_prompt_files:
        split_filename = re.split("_|\\.", generated_filename)
        generated_prompt_idx = int(split_filename[-2])
        generated_prompt_indices.append(generated_prompt_idx)
    remaining_prompts = [
        prompt
        for prompt in generation_prompts
        if prompt["JSON_idx"] not in generated_prompt_indices
    ]
    return remaining_prompts


def save_data(
    all_data: list[dict[str, Any]], trajectory_list: list[Trajectory]
) -> None:
    all_data[0]["trajectories"] = [
        trajectory.get_json_representation() for trajectory in trajectory_list
    ]


def write_to_disk(
    all_data: list[dict[str, Any]],
    output_folder: str,
    initial_memory: int,
    pretty_print_output: bool = False,
    record_memory: bool = False,
    force_dump: bool = False,
) -> None:
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    prompt_idx: int = (
        all_data[0]["prompt"]["JSON_idx"]
        if "prompt" in all_data[0]
        and type(all_data[0]["prompt"]) == dict
        and "JSON_idx" in all_data[0]["prompt"]
        else 0
    )
    llm_name: str = all_data[0]["llm_name"]
    reward_model_name: str = all_data[0]["reward_model_name"]
    write_filename = f"{llm_name}_{reward_model_name}_prompt_{prompt_idx:04d}.json"
    write_path = os.path.join(output_folder, write_filename)
    if force_dump or (record_memory and prompt_idx == 0):
        dump_memory_snapshot(write_path, initial_memory)
    if force_dump:
        return
    print_best_trajectory(all_data)
    with open(write_path, "w") as fp:
        if pretty_print_output:
            json.dump(all_data, fp, indent=4)
        else:
            json.dump(all_data, fp)
        print(f"Wrote data to {write_filename}")


def dump_memory_snapshot(json_write_path: str, initial_memory: int) -> None:
    torch.cuda.memory._dump_snapshot(
        filename=f"{json_write_path[:-5]}_init_{initial_memory}.pickle"
    )


def print_best_trajectory(all_data: list[dict[str, Any]]) -> None:
    prompt = all_data[0]["prompt"]
    if type(prompt) == dict:
        prompt = prompt["prompt"]
    best_response, best_score = get_best_response(all_data)
    print("PROMPT:")
    print("*" * 20)
    print(prompt)
    print("*" * 20)
    print("BEST RESPONSE:")
    print("*" * 20)
    print(best_response)
    print("*" * 20)
    print(f"REWARD OF BEST RESPONSE: {best_score}")


def get_best_response(all_data: list[dict[str, Any]]) -> tuple[str, float]:
    best_trajectory = all_data[0]["trajectories"][0]
    for data_dict in all_data:
        trajectories: list[dict[str, Any]] = data_dict["trajectories"]
        for trajectory in trajectories:
            if trajectory["score"] > best_trajectory["score"]:
                best_trajectory = trajectory
    return best_trajectory["output"], best_trajectory["score"]
