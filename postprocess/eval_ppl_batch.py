# CUDA_VISIBLE_DEVICES=0 python postprocess/eval_ppl_batch.py --model Meta-Llama-3-8B --rm ArmoRM-Llama3-8B-v0.1
# CUDA_VISIBLE_DEVICES=1 python postprocess/eval_ppl_batch.py --model Mistral-7B-v0.3 --rm ArmoRM-Llama3-8B-v0.1
# CUDA_VISIBLE_DEVICES=2 python postprocess/eval_ppl_batch.py --model Meta-Llama-3-8B-Instruct --rm ArmoRM-Llama3-8B-v0.1

import torch
from termcolor import colored
import gc
import time
import json
from tqdm import tqdm
import pandas as pd

from utils.validation_utils import (
    get_full_model_name,
    validate_llm_name,
    validate_reward_model_name,
)

from utils.generation_utils import (
    get_generation_model,
    get_generation_tokenizer,
    get_terminators,
)

from utils.generation_utils import (
    get_input_encoding,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)

import os

ROOT = 'archive'

from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model", type=str, default='Meta-Llama-3-8B')
    p.add_argument("--rm", type=str, default='ArmoRM-Llama3-8B-v0.1')
    return p.parse_args()

args = parse_args()

MODELs = ['Meta-Llama-3-8B', 'Mistral-7B-v0.3', 'Meta-Llama-3-8B-Instruct']
RMs = ['ArmoRM-Llama3-8B-v0.1', 'RM-Mistral-7B', 'FsfairX-LLaMA3-RM-v0.1']

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

all_stats = []

def get_parsed_data(filepath: str):
    # print(f"Reading {filepath}")
    with open(filepath, "r") as f:
        full_data: list = json.load(f)
    parsed_data: dict = {}
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

@torch.inference_mode()
def calculate_perplexity(
    generation_model, input_encoding: torch.Tensor
) -> list[float]:
    outputs = generation_model(
        **input_encoding,
        labels=input_encoding.input_ids,
    )
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return perplexity.item()

@torch.inference_mode()
def compute_json_file(src: str, generation_model, generation_tokenizer, llm_name, setting) -> float:

    all_min_ppl = []
    
    assert os.path.exists(src), f'[ERROR] {src} does not exist'

    file_list = os.listdir(src)
    num_files = len(file_list)
    assert num_files == 100, f'[ERROR] {src} does not have 100 files, but {num_files}'
    
    for file in tqdm(file_list):
        ppl = []
        _data = get_parsed_data(os.path.join(src, file))
        _trajectories = _data["trajectories"]

        for _traj in tqdm(_trajectories):
            texts = _traj["prompt"] + _traj["output"]
            texts = get_templated_prompt(
                texts, llm_name, generation_tokenizer
            )
            input_encoding = get_input_encoding(
                texts,
                generation_model,
                generation_tokenizer,
            )
            ppl.append(calculate_perplexity(generation_model, input_encoding))
        
        ppl = torch.Tensor(ppl).min().item()
        all_min_ppl.append(ppl)

    all_stats.append(
        {
            'model': llm_name,
            'ppl': torch.Tensor(all_min_ppl).mean().item(),
            'setting': setting,
        }
    )

    print(torch.Tensor(all_min_ppl).mean().item())


def get_ppl_SpR(filepath, llm_name, setting):
    ppl = []
    with open(filepath, "r") as f:
        full_data: list = json.load(f)

    for data_dict in full_data:
        ppl.append(-data_dict["score"][0])

    all_stats.append(
        {
            'model': llm_name,
            'ppl': torch.Tensor(ppl).mean().item(),
            'setting': setting,
        }
    )

    print(torch.Tensor(ppl).mean().item())

model = args.model
rm = args.rm

llm_name = get_full_model_name("", model)
generation_tokenizer = get_generation_tokenizer(llm_name, False)
generation_model = get_generation_model(llm_name, 'cuda:0',local_files_only=False)
print(colored(f'============[INFO] Computing {model} {rm}============', 'blue'))
for alpha in alphas:
    out = get_ppl_SpR(f'results/SpR_alpha_{alpha}_{model}_{model}_0.json', llm_name, f'SpR_{alpha}')
        
# check BoN logs
out = compute_json_file(f'{ROOT}/Bo120_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo120')
out = compute_json_file(f'{ROOT}/Bo240_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo240')
out = compute_json_file(f'{ROOT}/Bo480_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo480')
out = compute_json_file(f'{ROOT}/Bo960_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo960')
out = compute_json_file(f'{ROOT}/Bo1920_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo1920')
out = compute_json_file(f'{ROOT}/Bo3840_{model}_{rm}_0', generation_model, generation_tokenizer, llm_name, 'Bo3840')

df = pd.DataFrame(all_stats)
print(df.to_markdown(index=False))