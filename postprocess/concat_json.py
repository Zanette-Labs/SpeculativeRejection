import json
import os
from tqdm import tqdm
from typing import Any
from glob import glob


LM_NAME_LIST = ["Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct", "Mistral-7B-v0.3"]

RM_NAME_LIST = ["RM-Mistral-7B", "FsfairX-LLaMA3-RM-v0.1", "ArmoRM-Llama3-8B-v0.1"]

NUM_LIST = [2, 4]

ROOT = 'archive'


def get_json_filepaths(json_folder_path: str) -> list[str]:
    return glob(os.path.join(json_folder_path, "*.json"))

def get_data(filepath: str) -> list[dict[str, Any]]:
    with open(filepath, "r") as f:
        file_data: list[dict[str, Any]] = json.load(f)
    return file_data


def write_to_disk(data: list[dict[str, Any]], basename: str, MERGE_NAME: str) -> None:
    write_path = os.path.join(MERGE_NAME, basename)
    with open(write_path, "w") as fp:
        json.dump(data, fp)


def main() -> None:

    for LM_NAME in LM_NAME_LIST:
        for RM_NAME in RM_NAME_LIST:
            for NUM in NUM_LIST:

                MERGE_FOLDERS = [
                    f"{ROOT}/Bo960_{LM_NAME}_{RM_NAME}_0",
                    f"{ROOT}/Bo960_{LM_NAME}_{RM_NAME}_8",
                    f"{ROOT}/Bo960_{LM_NAME}_{RM_NAME}_16",
                    f"{ROOT}/Bo960_{LM_NAME}_{RM_NAME}_24",
                ]

                MERGE_FOLDERS = MERGE_FOLDERS[:NUM]

                MERGE_NAME = f"{ROOT}/Bo{NUM*960}_{LM_NAME}_{RM_NAME}_0"

                if not os.path.isdir(MERGE_NAME):
                    os.mkdir(MERGE_NAME)
                nested_filenames: list[list[str]] = []
                num_filepaths = -1
                for merge_folder in MERGE_FOLDERS:
                    json_filepaths = sorted(get_json_filepaths(merge_folder))
                    if num_filepaths == -1:
                        num_filepaths = len(json_filepaths)
                    else:
                        assert num_filepaths == len(
                            json_filepaths
                        ), f"num_filepaths: {num_filepaths}, len(json_filepaths): {len(json_filepaths)}"
                    nested_filenames.append(json_filepaths)
                for idx in tqdm(range(num_filepaths)):
                    all_data: list[dict[str, Any]] = []
                    for filenames in nested_filenames:
                        filename = filenames[idx]
                        data = get_data(filename)
                        all_data.extend(data)
                    write_to_disk(all_data, os.path.basename(filename), MERGE_NAME)
                
                print(f"{MERGE_NAME} done.")


if __name__ == "__main__":
    main()