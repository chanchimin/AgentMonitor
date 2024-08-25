import sys
sys.path.append("..")
import json
import os
from agentmonitor import doJudge
import fire
import glob
import re
from tqdm import tqdm

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

def get_name_from_config_name(config_name):

    # because llama3_70b_instruct_8001.json, we do not want _8001.json

    if "llama3_70b_instruct" in config_name:
        return "llama3_70b_instruct"
    elif "llama3_8b_instruct" in config_name:
        return "llama3_8b_instruct"
    elif "gpt_3.5_turbo" in config_name:
        return "gpt_3.5_turbo"
    elif "gpt4" in config_name:
        return  "gpt4"

# this is for current architecture only
def main(task="gsm8k", judge_llm_config_files="gpt_3.5_turbo_proxy.json", input_path="output/test/3.5_3.5_3.5_3.5/perturbation_config/test_turn_3"):

    output_path = f"{input_path}/{task}/judge_output/judged_by_{get_name_from_config_name(judge_llm_config_files)}"
    glob_input_path = f"{input_path}/{task}/task_*.json"

    # an easy way to approximately determine whether to skip this config
    if os.path.exists(output_path):
        total_file = len(glob.glob(glob_input_path))
        judged_file = len(glob.glob(f"{output_path}/task_*.json"))
        if total_file == judged_file:
            print(f"skip {input_path}, all files have been judged.")
            exit()

    # sanity check, whether this config has its final output (task_results.json)
    if not os.path.exists(f"{input_path}/{task}/{task}_result.json"):
        print(f"skip {input_path}/{task}/{task}_result.json, not found.")
        exit()

    for input_file in tqdm(glob.glob(glob_input_path)):
        print(f"judging file {input_file} now ...")
        file_name = input_file.split("/")[-1]

        try:
            doJudge(task=task,
                    input_path=f"{input_file}",
                    output_path=os.path.join(output_path, file_name),
                    llm_config_files=judge_llm_config_files,
                    use_name=True)
        except KeyboardInterrupt:
            print("Process interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"skip {input_file}, error raised:\n{e}")
            exit()


if __name__ == '__main__':
    fire.Fire(main)
