import os
from termcolor import colored

root = 'archive/'

MODELs = ['Meta-Llama-3-8B', 'Mistral-7B-v0.3', 'Meta-Llama-3-8B-Instruct']
RMs = ['ArmoRM-Llama3-8B-v0.1', 'RM-Mistral-7B', 'FsfairX-LLaMA3-RM-v0.1']

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def check_folder(root, folder_name):
    flag = True
    folder_path = os.path.join(root, folder_name)
            
    if not os.path.exists(folder_path):
        print(colored(f'[ERROR] {folder_path} does not exist', 'red'))
        flag = False
                
    else:
        file_list = os.listdir(folder_path)
        num_files = len(file_list)
        if num_files != 100:
            print(colored(f'[ERROR] {folder_path} does not have 100 files, but {num_files}', 'yellow'))
            flag = False
        # else:
        #     print(colored(f'[PASS] {folder_path} checked!', 'green'))
    
    return int(flag)

num_files = 0
checked_files = 0

for model in MODELs:
    for rm in RMs:
        print(colored(f'============[INFO] Checking {model} {rm}============', 'blue'))
        flag = True
        # check SpR logs
        for alpha in alphas:
            out = check_folder(root, f'SpR_alpha_{alpha}_{model}_{rm}_0')
            flag &= out
            checked_files += out
        
        # check BoN logs
        out = check_folder(root, f'Bo120_{model}_{rm}_0')
        flag &= out
        checked_files += out
        
        out = check_folder(root, f'Bo240_{model}_{rm}_0')
        flag &= out
        checked_files += out

        out = check_folder(root, f'Bo480_{model}_{rm}_0')
        flag &= out
        checked_files += out

        out = check_folder(root, f'Bo960_{model}_{rm}_0')
        flag &= out
        checked_files += out

        out = check_folder(root, f'Bo960_{model}_{rm}_8')
        flag &= out
        checked_files += out

        out = check_folder(root, f'Bo960_{model}_{rm}_16')
        flag &= out
        checked_files += out

        out = check_folder(root, f'Bo960_{model}_{rm}_24')
        flag &= out
        checked_files += out

        if flag:
            print(colored(f'[PASS] {model} {rm} checked!', 'green'))
        
        num_files += 7 + 9

for model in MODELs:
    rm = model
    print(colored(f'============[INFO] Checking {model} {rm}============', 'blue'))
    flag = True
    # check SpR logs
    for alpha in alphas:
        out = check_folder(root, f'SpR_alpha_{alpha}_{model}_{rm}_0')
        flag &= out
        checked_files += out

    if flag:
        print(colored(f'[PASS] {model} {rm} checked!', 'green'))
        
    num_files += 9

print(colored(f'[INFO] Checked {checked_files} files, {num_files} files in total, progress {round(checked_files/num_files * 100,2)}%', 'blue'))

