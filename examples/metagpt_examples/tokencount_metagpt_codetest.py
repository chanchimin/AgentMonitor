import sys
sys.path.append("../..")
import json
import os
from agentmonitor import doTokenCount
import fire
from tqdm import tqdm
import glob


def main(task="gsm8k", input_path="output/test/3.5_3.5_3.5_3.5/perturbation_config/test_turn_3"):

    output_path = f"{input_path}/{task}/token_count_output"
    glob_input_path = f"{input_path}/{task}/task_*.json"

    # an easy way to approximately determine whether to skip this config
    if os.path.exists(output_path):
        total_file = len(glob.glob(glob_input_path))
        token_count_file_file = len(glob.glob(f"{output_path}/task_*.json"))
        if total_file == token_count_file_file:
            print(f"skip {input_path}, all files have been count.")
            exit()

    # sanity check, whether this config has its final output (task_results.json)
    if not os.path.exists(f"{input_path}/{task}/{task}_result.json"):
        print(f"skip {input_path}, task_result.json not found.")
        exit()


    for input_file in tqdm(glob.glob(glob_input_path)):
        print(f"counting token file {input_file} now ...")
        file_name = input_file.split("/")[-1]

        doTokenCount(input_path=f"{input_file}",
                     output_path=os.path.join(output_path, file_name),
                     use_name=True)


if __name__ == '__main__':
    fire.Fire(main)
