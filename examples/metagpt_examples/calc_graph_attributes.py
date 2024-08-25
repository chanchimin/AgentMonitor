import sys
sys.path.append("../..")
import os
from agentmonitor.visualizer import generate_node_color, graph_analyze
import fire
from tqdm import tqdm
import glob
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict
from collections import defaultdict

def merge_dicts(dicts):
    merged = defaultdict(lambda: defaultdict(int))
    for d in dicts:
        for key, subdict in d.items():
            for subkey, value in subdict.items():
                merged[key][subkey] += value
    return dict(merged)


def normalize_weights(weights):
    sorted_weights = sorted(weights.items(), key=lambda item: item[1])

    max_weight = sorted_weights[-1][1]
    min_weight = sorted_weights[0][1]

    if max_weight == min_weight:
        return [1] * len(weights)
    else:
        return [(weight - min_weight) / (max_weight - min_weight) * 10 + 1 for weight in weights.values()]


def main(task="gsm8k", input_path="output/test/3.5_3.5_3.5_3.5/perturbation_config/test_turn_3"):

    attribute_output_path = f"{input_path}/{task}/graph_attributes.json"
    # directly use token_count_output
    glob_input_path = f"{input_path}/{task}/token_count_output/task_*.json"

    if not os.path.exists(os.path.dirname(glob_input_path)):
        print(f"input path {glob_input_path} does not exist, skip")
        exit()


    token_count_list = []
    # this is per instance statistic, so we add them together
    print(f"counting token file {glob_input_path} now ...")
    for input_file in tqdm(glob.glob(glob_input_path)):
        file_name = input_file.split("/")[-1]

        with open(input_file, "r") as fin:
            monitor_output: Dict = json.load(fin)
        token_count_list.append(monitor_output)

    total_token_count = merge_dicts(token_count_list)

    dg = nx.DiGraph()
    # add colorful node, first count the weight (input num + output num) of the node, and afterwards discrete it
    node_color = generate_node_color()
    total_node = list(total_token_count.keys())
    node2weight = defaultdict(int)
    # either the node send token or receive token we add the token count
    for node_from in total_node:
        for node_to in total_token_count[node_from]:
            node2weight[node_from] += total_token_count[node_from][node_to]
            node2weight[node_to] += total_token_count[node_from][node_to]

    sorted_keys = sorted(node2weight, key=lambda x: node2weight[x], reverse=True)
    # create ranking dict
    node2rank = {key: rank for rank, key in enumerate(sorted_keys)}
    # use ranking to get the color is fine
    for name, rank in node2rank.items():
        if rank > 29:
            rank = 29
        dg.add_node(node_for_adding=f"{name}", label=f"{name}", color=node_color[rank])

    # add weighted edge
    edges = [] # record for add curve
    for node_from, targets in total_token_count.items():
        for node_to, weight in targets.items():
            if weight > 0:  # only add the edge has weight greater than 0
                dg.add_edge(node_from, node_to, weight=weight)
                edges.append((node_from, node_to))

    # get the graph attributes and save it
    attributes_dict = graph_analyze(dg)
    with open(attribute_output_path, "w") as fout:
        fout.write(json.dumps(attributes_dict))


if __name__ == '__main__':
    fire.Fire(main)


