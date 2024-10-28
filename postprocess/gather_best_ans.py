import os
from termcolor import colored
from typing import Any
import json

ROOT = 'archive'
RESULTS = 'results'

MODELs = ['Meta-Llama-3-8B', 'Mistral-7B-v0.3', 'Meta-Llama-3-8B-Instruct']
RMs = ['ArmoRM-Llama3-8B-v0.1', 'RM-Mistral-7B', 'FsfairX-LLaMA3-RM-v0.1']
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def get_parsed_data(filepath: str) -> dict[str, Any]:
    # print(f"Reading {filepath}")
    with open(filepath, "r") as f:
        full_data: list[dict[str, Any]] = json.load(f)
    parsed_data: dict[str, Any] = {}
    for data_dict in full_data:
        # add trajectories to parsed_data for every data_dict
        if "trajectories" in parsed_data:
            parsed_data["trajectories"].extend(data_dict["trajectories"])
        else:
            parsed_data["trajectories"] = data_dict["trajectories"]
        # add elapsed_sec to parsed_data for every data_dict
        if "elapsed_sec" in parsed_data:
            parsed_data["elapsed_sec"] = max(
                data_dict["elapsed_sec"], parsed_data["elapsed_sec"]
            )
        else:
            parsed_data["elapsed_sec"] = data_dict["elapsed_sec"]
    return parsed_data

def gather_best_ans(src, dst):
    if not os.path.exists(src):
        print(colored(f'[ERROR] {src} does not exist', 'red'))
    else:
        file_list = os.listdir(src)
        num_files = len(file_list)
        if num_files != 100:
            print(colored(f'[ERROR] {src} does not have 100 files, but {num_files}', 'yellow'))
        else:
            # do the collection
            dst_data = []
            
            for file in file_list:
                _data = get_parsed_data(os.path.join(src, file))
                _trajectories = _data["trajectories"]
                _scores: list[float] = [traj["score"] for traj in _trajectories]

                # get best one
                _best_score = max(_scores)
                _best_traj = _trajectories[_scores.index(_best_score)]
                dst_data.append(_best_traj)

            os.makedirs(RESULTS, exist_ok=True)
            # write to file
            with open(dst, "w") as f:
                json.dump(dst_data, f)
            print(colored(f'[INFO] {src} has been gathered to {dst}', 'green'))

for model in MODELs:
    for rm in RMs:
        print(colored(f'============[INFO] Gathering {model} {rm}============', 'blue'))

        # check SpR logs
        for alpha in alphas:
            out = gather_best_ans(f'{ROOT}/SpR_alpha_{alpha}_{model}_{rm}_0', f'{RESULTS}/SpR_alpha_{alpha}_{model}_{rm}_0.json')
        
        # check BoN logs
        out = gather_best_ans(f'{ROOT}/Bo120_{model}_{rm}_0', f'{RESULTS}/Bo120_{model}_{rm}_0.json')
        out = gather_best_ans(f'{ROOT}/Bo240_{model}_{rm}_0', f'{RESULTS}/Bo240_{model}_{rm}_0.json')
        out = gather_best_ans(f'{ROOT}/Bo480_{model}_{rm}_0', f'{RESULTS}/Bo480_{model}_{rm}_0.json')
        out = gather_best_ans(f'{ROOT}/Bo960_{model}_{rm}_0', f'{RESULTS}/Bo960_{model}_{rm}_0.json')
        out = gather_best_ans(f'{ROOT}/Bo1920_{model}_{rm}_0', f'{RESULTS}/Bo1920_{model}_{rm}_0.json')
        out = gather_best_ans(f'{ROOT}/Bo3840_{model}_{rm}_0', f'{RESULTS}/Bo3840_{model}_{rm}_0.json')

for model in MODELs:
    print(colored(f'============[INFO] Gathering {model} {model}============', 'blue'))

    # check SpR logs
    for alpha in alphas:
        out = gather_best_ans(f'{ROOT}/SpR_alpha_{alpha}_{model}_{model}_0', f'{RESULTS}/SpR_alpha_{alpha}_{model}_{model}_0.json')

out = gather_best_ans(f'Meta-Llama-3-8B-Instruct', f'{RESULTS}/Meta-Llama-3-8B-Instruct-ref.json')