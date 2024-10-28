# Checks score and relative compute time of speculative rejection

import json
import numpy as np
import os
from copy import deepcopy
from glob import glob
from matplotlib import pyplot as plt
from pprint import pprint
from time import sleep
from typing import Any
import os

LM_NAME = "Meta-Llama-3-8B"
LM_NAME = "Meta-Llama-3-8B-Instruct"
LM_NAME = "Mistral-7B-v0.3"

RM_NAME = "RM-Mistral-7B"

ROOT = 'archive'

BASELINE_FOLDER_PATHS = [
    f"{ROOT}/Bo3840_{LM_NAME}_{RM_NAME}_0",
]

COMPARE_FOLDER_PATHS = [
    f"{ROOT}/Bo120_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/Bo240_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/Bo480_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/Bo960_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/Bo1920_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/Bo3840_{LM_NAME}_{RM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.9_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.8_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.7_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.6_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.5_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.4_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.3_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.2_{LM_NAME}_{LM_NAME}_0",
    f"{ROOT}/SpR_alpha_0.1_{LM_NAME}_{LM_NAME}_0",
]


def get_json_filepaths(json_folder_path: str) -> list[str]:
    return glob(os.path.join(json_folder_path, "*.json"))


def get_num_gpus(json_folder_path: str) -> int:
    'Bo240_{LM_NAME}_{RM_NAME}_0'
    try:
        num_gpus = int(json_folder_path.split("/")[-1].split("_")[0].split('Bo')[-1]) // 120
        print(num_gpus, json_folder_path.split("/")[-1].split("_")[0].split('Bo')[-1])
    except ValueError:
        print("num_gpus not found, defaulting to 1")
        num_gpus = 1
    return num_gpus


def get_alpha_value(json_folder_path: str) -> float:
    alpha_value = float(json_folder_path.split("/")[-1].split("_")[2])
    return alpha_value


def get_parsed_data(filepath: str) -> dict[str, Any]:
    # print(f"Reading {filepath}")
    with open(filepath, "r") as f:
        full_data: list[dict[str, Any]] = json.load(f)
    gen_times = 0.0
    for data_dict in full_data:
        # print(data_dict["clock"]) # clock is a list
        assert type(data_dict["clock"]) == list
        for clock_dict in data_dict["clock"]: # ['tokenization', 0.006562471389770508]
            if clock_dict[0] == "generation pass" or clock_dict[0].startswith("generation"):
                gen_times += clock_dict[1]
    # print(gen_times)
    return gen_times



def compute_improvement(
    bon_data: dict[str, Any], spec_rej_data: dict[str, Any]
) -> float:
    bon_trajectories = bon_data["trajectories"]
    spec_rej_trajectories = spec_rej_data["trajectories"]
    bon_scores: list[float] = [traj["score"] for traj in bon_trajectories]
    spec_rej_scores: list[float] = [traj["score"] for traj in spec_rej_trajectories]
    # best_bon_response = [traj["output"] for traj in bon_trajectories if traj["score"] == max(bon_scores)]
    # best_spec_rej_response = [traj["output"] for traj in spec_rej_trajectories if traj["score"] == max(spec_rej_scores)]
    absolute_difference = max(bon_scores) - min(bon_scores)
    improvement = max(spec_rej_scores) - max(bon_scores)
    return improvement / absolute_difference


def validate_prompt(
    bon_data: dict[str, Any],
    spec_rej_data: dict[str, Any],
    bon_filepath: str,
    spec_rej_filepath: str,
) -> None:
    warned = False
    bon_prompt = bon_data["trajectories"][0]["prompt"]
    for idx in range(len(bon_data["trajectories"])):
        assert (
            bon_data["trajectories"][idx]["prompt"] == bon_prompt
        ), "Prompts are not the same!"
    idx = 0
    while idx < len(spec_rej_data["trajectories"]):
        if spec_rej_data["trajectories"][idx]["prompt"] != bon_prompt:
            spec_rej_data["trajectories"].pop(idx)
            if not warned:
                print(f"WARNING: {spec_rej_filepath} inconsistent!")
                warned = True
        else:
            idx += 1


def plot_data(
    bon_plot_points: dict[str, list[Any]],
    spec_eff_plot_points: dict[str, list[Any]],
) -> None:
    line_width = 2
    marker_size = 6

    label_x_offset = -14
    label_y_offset = -3

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    color1 = '#66c2a5'
    color2 = '#fc8d62'
    color3 = '#8da0cb'
    color4 = '#e78ac3'
    color5 = '#a6d854'

    plt.rcParams.update({"font.size": 12})
    plt.figure(figsize=(6, 5))

    bon_labels = bon_plot_points["labels"]
    bon_x = bon_plot_points["compute"]
    bon_y = bon_plot_points["score"]
    spec_eff_labels = spec_eff_plot_points["labels"]
    spec_eff_x = spec_eff_plot_points["compute"]
    spec_eff_y = spec_eff_plot_points["score"]

    plt.plot(
        bon_plot_points["compute"],
        bon_plot_points["score"],
        label="BoN",
        marker="o",
        linestyle="--",
        color=color2,
        linewidth=line_width,
        markersize=marker_size,
    )
    plt.plot(
        spec_eff_plot_points["compute"],
        spec_eff_plot_points["score"],
        label="Speculative Rejection",
        marker="o",
        linestyle="--",
        color=color1,
        linewidth=line_width,
        markersize=marker_size,
    )
    plt.xscale("log")
    # plt.grid(alpha=0.5, zorder=0)
    x_ticks = get_x_ticks()
    plt.xticks(x_ticks, labels=[f"{x:.1f}" for x in x_ticks], fontsize=15)
    plt.yticks(fontsize=15)

    for idx, label in enumerate(bon_labels):
        plt.annotate(
                str(int(label)*120),
                (bon_x[idx], bon_y[idx]),
                textcoords="offset points",
                xytext=(label_x_offset, label_y_offset),
                ha="left",
                va="top",
            )
    for idx, label in enumerate(spec_eff_labels):
        if idx % 2 == 0:
            plt.annotate(
                label,
                (spec_eff_x[idx], spec_eff_y[idx]),
                textcoords="offset points",
                xytext=(label_x_offset, label_y_offset),
                ha="left",
                va="top",
            )

    plt.xlabel("Relative GPU Compute", fontsize=15)
    plt.ylabel("Improvement Score", fontsize=15)
    plt.ylim(bottom=98)
    plt.title(f"{LM_NAME.replace('Meta-','')} w/ {RM_NAME.replace('-v0.1','')}", fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"imgs/{LM_NAME}_{RM_NAME}.pdf", bbox_inches="tight")
    plt.show()


def get_x_ticks() -> list[int]:
    axes = plt.gca()
    x_min, x_max = axes.get_xlim()
    min_log_value = np.ceil(np.log2(x_min))
    max_log_value = np.floor(np.log2(x_max))
    x_ticks = [2 ** i for i in range(int(min_log_value), int(max_log_value) + 1)]
    return x_ticks


def compute_speedups(
    bon_plot_points: dict[str, list[Any]],
    spec_rej_plot_points: dict[str, list[Any]],
) -> None:
    bon_x, bon_y = bon_plot_points["compute"], bon_plot_points["score"]
    spec_rej_x, spec_rej_y = (
        spec_rej_plot_points["compute"],
        spec_rej_plot_points["score"],
    )
    for x_s, y_s in zip(spec_rej_x, spec_rej_y):
        for idx, (x_b, y_b) in enumerate(zip(bon_x, bon_y)):
            if y_s < y_b or idx == len(bon_x) - 1:
                x_prev, y_prev = bon_x[idx - 1], bon_y[idx - 1]
                interpolated_x = interpolate_log(x_prev, y_prev, x_b, y_b, y_s)
                speedup = interpolated_x / x_s
                print(f"({x_s}, {y_s:.1f}) -> {speedup}, (idx {idx}, x_b:{x_b}, y_b: {y_b}, interpolated_x: {interpolated_x})")
                break


def interpolate_log(x1: float, y1: float, x2: float, y2: float, y: float) -> float:
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)
    log_x = log_x1 + (y - y1) * (log_x2 - log_x1) / (y2 - y1)
    return np.exp(log_x)


def main() -> None:
    while len(BASELINE_FOLDER_PATHS) < len(COMPARE_FOLDER_PATHS):
        BASELINE_FOLDER_PATHS.append(BASELINE_FOLDER_PATHS[-1])
    bon_plot_points = {
        "labels": [],
        "compute": [],
        "score": [],
    }
    spec_eff_plot_points = deepcopy(bon_plot_points)
    for baseline_path, compare_path in zip(BASELINE_FOLDER_PATHS, COMPARE_FOLDER_PATHS):
        print(f"{baseline_path} vs {compare_path}")
        print("****************************************************")
        bon_filepaths = sorted(get_json_filepaths(baseline_path))
        spec_rej_filepaths = sorted(get_json_filepaths(compare_path))

        bon_gpus = get_num_gpus(baseline_path)
        spec_rej_gpus = get_num_gpus(compare_path)

        assert (
            len(bon_filepaths) == len(spec_rej_filepaths) == 100
        ), f"len(bon_filepaths): {len(bon_filepaths)}, len(spec_rej_filepaths): {len(spec_rej_filepaths)}, path: {bon_filepaths}, {spec_rej_filepaths}"

        improvements: list[float] = []
        total_BoN_time = 0.0
        total_spec_rej_time = 0.0

        for bon_filepath, spec_rej_filepath in zip(bon_filepaths, spec_rej_filepaths):
            bon_filepath_ending = bon_filepath.split("_")[-1]
            spec_rej_filepath_ending = spec_rej_filepath.split("_")[-1]
            assert (
                bon_filepath_ending == spec_rej_filepath_ending
            ), f"{bon_filepath} and {spec_rej_filepath} have different endings"
            bon_data = get_parsed_data(bon_filepath)
            spec_rej_data = get_parsed_data(spec_rej_filepath)

            total_BoN_time += bon_data
            total_spec_rej_time += spec_rej_data
            del bon_data, spec_rej_data

        print(total_spec_rej_time, total_BoN_time, bon_gpus)
        relative_compute_time = total_spec_rej_time / total_BoN_time
        relative_gpu_compute = relative_compute_time * spec_rej_gpus / bon_gpus
        print(f"relative compute time: {(relative_compute_time)}")
        print(f"relative GPU compute: {(relative_gpu_compute)}")
        print("****************************************************")
        if "SpR_alpha" in compare_path:
            alpha_value = get_alpha_value(compare_path)
            spec_eff_plot_points["labels"].append(alpha_value)
            spec_eff_plot_points["compute"].append(relative_gpu_compute)
        elif "Bo" in compare_path:
            bon_plot_points["labels"].append(spec_rej_gpus)
            bon_plot_points["compute"].append(relative_gpu_compute)
        else:
            raise ValueError(f"Unknown baseline: {compare_path}")
    # plot_data(bon_plot_points, spec_eff_plot_points)
    # compute_speedups(bon_plot_points, spec_eff_plot_points)

if __name__ == "__main__":
    main()