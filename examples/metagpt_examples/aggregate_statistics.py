import logging
import os
import ast
import fire
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import glob
from typing import Dict
import re
from math import comb, perm
from collections import Counter

predefined_agents = ['simplecoder', 'simpletester', 'simplereviewer', 'answerextractor', 'webbrowser', 'codemodifier', 'executor', 'dummyagent', 'system']

arch_task2agents_in_order = {
    "base": {
        "humaneval": ["simplecoder", "simpletester", "simplereviewer"],
        "mmlu": ["simplecoder", "simpletester", "simplereviewer", "answerextractor"],
        "gsm8k": ["simplecoder", "simpletester", "simplereviewer", "answerextractor"],
    },
    "arch2": {
        "humaneval": ["simplecoder", "simpletester", "simplereviewer", "dummyagent"],
        "mmlu": ["simplecoder", "simpletester", "simplereviewer", "answerextractor", "dummyagent"],
        "gsm8k": ["simplecoder", "simpletester", "simplereviewer", "answerextractor", "dummyagent"],
    },
    "arch3": {
        "humaneval": ["simplecoder", "simpletester", "simplereviewer", "webbrowser"],
        "mmlu": ["simplecoder", "simpletester", "simplereviewer", "webbrowser", "answerextractor"],
        "gsm8k": ["simplecoder", "simpletester", "simplereviewer", "webbrowser", "answerextractor"],
    },
    "arch4": {
        "humaneval": ["simplecoder", "codemodifier", "simpletester", "simplereviewer"],
        "mmlu": ["simplecoder", "codemodifier", "simpletester", "simplereviewer", "answerextractor"],
        "gsm8k": ["simplecoder", "codemodifier", "simpletester", "simplereviewer", "answerextractor"],
    },
    "arch5": {
        "humaneval": ["executor", "webbrowser",],
        "mmlu": ["executor", "webbrowser", "answerextractor"],
        "gsm8k": ["executor", "webbrowser", "answerextractor"],
    },


}


model2capability = {
    "70b": 3,
    "8b": 2,
    "u8b": 2,
    "3.5": 1,
}

def get_agent_capabilities(config, task, arch):
    agents_in_order = arch_task2agents_in_order[arch][task]

    models = re.sub(r'turn_\d+', '', config).split("_")

    agent2capability = {}
    for agent, model in zip(agents_in_order, models):
        agent2capability[agent + "_capabilities"] = model2capability.get(model, 0) # return 0 if not get the config name parsed correctly

    return agent2capability


def calc_heterogeneous_score(config):

    config = re.sub(r'turn_\d+', '', config)

    elements = [el for el in config.split('_') if el]

    n = len(elements)

    total_pairs = perm(n, 2)

    hetero_count = 0
    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            if elements[i] != elements[j]:
                hetero_count += 1
    try:
        heterogeneous_score = hetero_count / total_pairs
    except Exception as E:
        heterogeneous_score = 0
        logging.warning(f"error: {E} when calculating heterogeneous_score. return 0")

    return heterogeneous_score



def extract_score(score_string):
    # Use regex to find the score
    match = re.search(r'<score>\s*(\d+)\s*</score>', score_string)

    # If a score is found, return it as a float
    if match:
        return int(match.group(1))

    # If no score is found, return 0
    return 0


def process_list(items):

    if isinstance(items, str):
        try:
            items = ast.literal_eval(items)
            if not isinstance(items, list):
                raise ValueError
        except (ValueError, SyntaxError):

            items = items.split(',')
    print("Received list:", items)
    return items

def get_average_scores(total_scores_list):

    aggregated_scores = defaultdict(lambda: defaultdict(list))

    for agent_dict in total_scores_list:
        for person_ins in agent_dict:

            personal_score = extract_score(person_ins['Expected_Duties_Judgement'])
            collective_score = extract_score(person_ins['Overall_Performance_Judgement'])

            aggregated_scores[person_ins["Agent"]["Name"]]["personal_score"].append(personal_score)
            aggregated_scores[person_ins["Agent"]["Name"]]["collective_score"].append(collective_score)

    average_scores = {}
    for person, scores in aggregated_scores.items():
        average_scores[person] = {score_type: sum(score_list) / len(score_list) for score_type, score_list in
                                  scores.items()}

    return average_scores

def check_all_files_sanity(input_path, task, judge_llm):

    if not os.path.exists(f"{input_path}/{task}/{task}_result.json"):
        print(f"Error: {input_path}/{task}/{task}_result.json not found")
        return False
    if not os.path.exists(f"{input_path}/{task}/graph_attributes.json"):
        print(f"Error: {input_path}/{task}/graph_attributes.json not found")
        return False

    if not glob.glob(f"{input_path}/{task}/judge_output/{judge_llm}/task_*.json"):
        print(f"Error: {input_path}/{task}/judge_output/{judge_llm}/task_*.json not found")
        return False

    return True


def get_agents_perturbation_impact(original_score, perturbation_results, task, arch):

    agents_in_order = arch_task2agents_in_order[arch][task]
    name2score = {}

    # "["n000", "n000", "n000", ]"
    initial_config = len(agents_in_order) * ["n000"]
    for agent_index, agent_name in enumerate(agents_in_order):

        overall_perturbation_impact = 0

        for perturbation_weight, perturbation_name in zip([0.9, 0.6, 0.3], ["g010", "g040", "g070"]):
            cur_config = initial_config.copy()
            cur_config[agent_index] = perturbation_name
            cur_config_name = "_".join(cur_config)

            # this config name do not exist in the perturbation results
            if cur_config_name not in perturbation_results:
                continue

            after_score = perturbation_results[cur_config_name]['success_rate']
            di = (original_score - after_score) / original_score if original_score != 0 else 0
            overall_perturbation_impact += di * perturbation_weight

        name2score[agent_name + "_perturbation_impact"] = overall_perturbation_impact

    return name2score

def main(task="gsm8k",
         arch="base",
         input_paths: list = ['output/test/3.5_3.5_3.5_3.5/perturbation_config/test_turn_3'],
         output_path: str = "statistic_output/test/gsm8k/total_results.csv",
         judge_llm: str = "judged_by_gpt_3.5_turbo"):

    pd_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    input_paths = process_list(input_paths)

    # score path, and we assert the config name can be splitted by "/"
    # and the config_name will serve as the key in our stored csv file
    for input_path in tqdm(input_paths):

        config_name = input_path.split("/")[-1]
        print(f"deal with config {config_name} now ...")

        if not check_all_files_sanity(input_path, task, judge_llm):
            continue

        # target score
        with open(f"{input_path}/{task}/{task}_result.json", "r") as fin:
            target_dict: Dict = json.load(fin)

        target_score = target_dict['success_rate']

        # 1. get all agents score by gpt4
        glob_score_path = f"{input_path}/{task}/judge_output/{judge_llm}/task_*.json"
        total_scores_list = []
        # this is per instance statistic, so we add them together
        for input_file in glob.glob(glob_score_path):
            # print(f"merge agent score {input_file} now ...")

            with open(input_file, "r") as fin:
                score_output: Dict = json.load(fin)

            total_scores_list.append(score_output)

        average_scores = get_average_scores(total_scores_list)

        # 2. get graph attributes

        graph_attributes_path = f"{input_path}/{task}/graph_attributes.json"
        with open(graph_attributes_path, "r") as fin:
            graph_attributes: Dict = json.load(fin)

        # 3. get heterogeneous (not done yet, skip)

        heterogeneous_score = calc_heterogeneous_score(config_name)
        agent_capabilities = get_agent_capabilities(config_name, task, arch)

        # 4. put the field together and save the total results.csv
        # Create a dictionary where each key is a field name and the corresponding value is the attribute value
        row = {**average_scores, **graph_attributes, **agent_capabilities, "heterogeneous_score": heterogeneous_score, "config_name": task + "_" + arch + "_" + config_name}

        flattened_data = {}
        for agent in predefined_agents:
            if agent in row:
                flattened_data[f'{agent}_personal_score'] = row[agent].get('personal_score', 0)
                flattened_data[f'{agent}_collective_score'] = row[agent].get('collective_score', 0)
                flattened_data[f'{agent}_pagerank'] = row.get('pagerank', {}).get(agent, 0)

                # for like turn_1 turn_2, other agent do not have effect we do not take capabilities into account
                flattened_data[f'{agent}_capabilities'] = row.get(f'{agent}_capabilities', 0)

            else:
                flattened_data[f'{agent}_personal_score'] = None
                flattened_data[f'{agent}_collective_score'] = None
                flattened_data[f'{agent}_pagerank'] = None
                flattened_data[f'{agent}_capabilities'] = None


        flattened_data['config_name'] = row.get('config_name', '')
        flattened_data['total_number_of_nodes'] = row.get('total_number_of_nodes', 0)
        flattened_data['total_number_of_edges'] = row.get('total_number_of_edges', 0)
        flattened_data['average_clustering'] = row.get('average_clustering', 0)
        flattened_data['transitivity'] = row.get('transitivity', 0)
        flattened_data['average_degree_centrality'] = row.get('average_degree_centrality', 0)
        flattened_data['average_closeness_centrality'] = row.get('average_closeness_centrality', 0)
        flattened_data['average_betweenness_centrality'] = row.get('average_betweenness_centrality', 0)
        flattened_data['target_score'] = target_score
        flattened_data['heterogeneous_score'] = row.get('heterogeneous_score', 0)

        pd_data.append(flattened_data)

    if len(pd_data) == 0:
        print("No valid data found, exit.")
        exit()

    df = pd.DataFrame(pd_data)
    df.set_index("config_name")
    df.to_csv(output_path, index=False)

    print(df)


if __name__ == '__main__':
    fire.Fire(main)
