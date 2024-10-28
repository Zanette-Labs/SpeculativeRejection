import os
import json
from termcolor import colored

path = f"./results"

MODELs = ['Meta-Llama-3-8B', 'Mistral-7B-v0.3', 'Meta-Llama-3-8B-Instruct']
RMs = ['ArmoRM-Llama3-8B-v0.1', 'RM-Mistral-7B', 'FsfairX-LLaMA3-RM-v0.1']
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def add_json_file(results, path, generator):
    with open(path, "r") as f:
        data = json.load(f)
        for data_item in data:
            results.append(
                {
                    "instruction": data_item["prompt"],
                    "output": data_item["output"],
                    "score": data_item["score"],
                    "generator": generator,
                }
            )
    return results

os.makedirs('win_rate', exist_ok=True)
for model in MODELs:
    for rm in RMs:
        print(colored(f'============[INFO] Gathering {model} {rm}============', 'blue'))
        # gather ref
        results = []
        add_json_file(results, f"./results/Bo120_{model}_{rm}_0.json", f"Bo120_{model}_{rm}")
        json_file_name = f"win_rate/{model}_{rm}_ref.json"
        json.dump(results, open(json_file_name, "w"))

        # gather alpha
        results = []
        add_json_file(results, f"./results/Bo120_{model}_{rm}_0.json", f"Bo120_{model}_{rm}")
        add_json_file(results, f"./results/Bo240_{model}_{rm}_0.json", f"Bo240_{model}_{rm}")
        add_json_file(results, f"./results/Bo480_{model}_{rm}_0.json", f"Bo480_{model}_{rm}")
        add_json_file(results, f"./results/Bo960_{model}_{rm}_0.json", f"Bo960_{model}_{rm}")
        add_json_file(results, f"./results/Bo1920_{model}_{rm}_0.json", f"Bo1920_{model}_{rm}")
        add_json_file(results, f"./results/Bo3840_{model}_{rm}_0.json", f"Bo3840_{model}_{rm}")
        
        for alpha in alphas:
            add_json_file(results, f"./results/SpR_alpha_{alpha}_{model}_{rm}_0.json", f"SpR_{alpha}_{model}_{rm}")

        json_file_name = f"win_rate/{model}_{rm}_compare.json"
        json.dump(results, open(json_file_name, "w"))


results = []
add_json_file(results, f"./results/Meta-Llama-3-8B-Instruct-ref.json", f"Meta-Llama-3-8B-Instruct-ref")
json_file_name = f"win_rate/Meta-Llama-3-8B-Instruct-ref.json"
json.dump(results, open(json_file_name, "w"))