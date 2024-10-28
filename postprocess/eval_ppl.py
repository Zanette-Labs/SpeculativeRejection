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

ROOT = 'results'

MODELs = ['Meta-Llama-3-8B', 'Mistral-7B-v0.3', 'Meta-Llama-3-8B-Instruct']
RMs = ['ArmoRM-Llama3-8B-v0.1']

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

all_stats = []


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

def compute_json_file(filepath: str, generation_model, generation_tokenizer, llm_name, setting) -> float:
    
    ppl = []
    
    with open(filepath, "r") as f:
        full_data: list = json.load(f)

    for data_dict in tqdm(full_data):
        texts = data_dict["prompt"] + data_dict["output"]
        texts = get_templated_prompt(
            texts, llm_name, generation_tokenizer
        )
        input_encoding = get_input_encoding(
            [texts],
            generation_model,
            generation_tokenizer,
        )
        ppl.append(calculate_perplexity(generation_model, input_encoding))

    ppl = torch.Tensor(ppl).mean().item()
    all_stats.append(
        {
            'model': llm_name,
            'ppl': ppl,
            'setting': setting,
        }
    )

    return ppl

def get_ppl_for_spr(filepath: str) -> float:
    
    ppl = []
    
    with open(filepath, "r") as f:
        full_data: list = json.load(f)

    for data_dict in tqdm(full_data):
        texts = data_dict["prompt"] + data_dict["output"]
        texts = get_templated_prompt(
            texts, llm_name, generation_tokenizer
        )
        input_encoding = get_input_encoding(
            [texts],
            generation_model,
            generation_tokenizer,
        )
        ppl.append(calculate_perplexity(generation_model, input_encoding))

    ppl = torch.Tensor(ppl).mean().item()
    all_stats.append(
        {
            'model': llm_name,
            'ppl': ppl,
            'setting': setting,
        }
    )

    return ppl

rm = RMs[0]

for model in MODELs:

    llm_name = get_full_model_name("", model)
    generation_tokenizer = get_generation_tokenizer(llm_name, False)
    generation_model = get_generation_model(llm_name, 'cuda:0',local_files_only=False)


    print(colored(f'============[INFO] Computing {model}============', 'blue'))

    # check SpR logs
    for alpha in alphas:
        out = compute_json_file(f'{ROOT}/SpR_alpha_{alpha}_{model}_{model}_0.json', generation_model, generation_tokenizer, llm_name, f'SpR_{alpha}')
        
    # check BoN logs
    out = compute_json_file(f'{ROOT}/Bo120_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo120')
    out = compute_json_file(f'{ROOT}/Bo240_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo240')
    out = compute_json_file(f'{ROOT}/Bo480_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo480')
    out = compute_json_file(f'{ROOT}/Bo960_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo960')
    out = compute_json_file(f'{ROOT}/Bo1920_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo1920')
    out = compute_json_file(f'{ROOT}/Bo3840_{model}_{rm}_0.json', generation_model, generation_tokenizer, llm_name, 'Bo3840')

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(30)

df = pd.DataFrame(all_stats)
print(df.to_markdown(index=False))