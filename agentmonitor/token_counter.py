import json
import nltk
import os
from typing import Dict

# nltk.download("punkt")
def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)

token_map = {}

def init_token_map(agents, use_name):
    global token_map
    for i in range(len(agents)):
        if use_name:
            token_map[agents[i]["Name"]] = {}
        else:
            token_map[agents[i]["ID"]] = {}
        for j in range(len(agents)):
            if use_name:
                token_map[agents[i]["Name"]][agents[j]["Name"]] = 0
            else:
                token_map[agents[i]["ID"]][agents[j]["ID"]] = 0


def update_token_map(agent1, agent2, num):
    global token_map
    token_map[agent1][agent2] += num


def get_token_map(output_path):
    global token_map
    output_folder = os.path.dirname(output_path)
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    with open(output_path, "w") as fout:
        fout.write(json.dumps(token_map))

def load_data(input_path: str):
    with open(input_path, "r") as fin:
        monitor_output: Dict = json.load(fin)
    return monitor_output

def build_map(agents, history, use_name):
    input_history = [
        {
            "Agent": agents[-1],
            "Input": history[0]["Content"]["input"]
        }
    ]
    for info in history:
        message = info["Content"]["input"]
        for input_record in input_history:
            if message == input_record["Input"]:
                token_num = count_tokens(message)
                if use_name:
                    update_token_map(input_record["Agent"]["Name"], info["Agent"]["Name"], token_num)
                else:
                    update_token_map(input_record["Agent"]["ID"], info["Agent"]["ID"], token_num)
                break
        if info["Content"]["output"] != "":
            input_history.append(
                {
                    "Agent": info["Agent"],
                    "Input": info["Content"]["output"]
                }
            )

def build_influence_map(agents, history, use_name):
    notused_token_map = {}
    for i in range(len(agents)):
        if use_name:
            notused_token_map[agents[i]["Name"]] = {}
        else:
            notused_token_map[agents[i]["ID"]] = {}
        for j in range(len(agents)):
            if use_name:
                notused_token_map[agents[i]["Name"]][agents[j]["Name"]] = 0
            else:
                notused_token_map[agents[i]["ID"]][agents[j]["ID"]] = 0
    input_history = [
        {
            "Agent": agents[-1],
            "Input": history[0]["Content"]["input"]
        }
    ]
    for info in history:
        # this info is the meta information of all the history, that contains {"input": ..., "output": ""},
        # (the agent received message, but do not response)
        message = info["Content"]["input"]
        for input_record in input_history:
            # input_record is the record of what the agent says (in the speak order)
            if message == input_record["Input"]:
                token_num = count_tokens(message)
                if use_name:
                    notused_token_map[input_record["Agent"]["Name"]][info["Agent"]["Name"]] += token_num
                else:
                    notused_token_map[input_record["Agent"]["ID"]][info["Agent"]["ID"]] += token_num
                break
        if info["Content"]["output"] != "":
            input_history.append(
                {
                    "Agent": info["Agent"],
                    "Input": info["Content"]["output"]
                }
            )
            for agent in agents:
                if use_name:
                    update_token_map(agent["Name"], info["Agent"]["Name"], notused_token_map[agent["Name"]][info["Agent"]["Name"]])
                else:
                    update_token_map(agent["ID"], info["Agent"]["ID"], notused_token_map[agent["ID"]][info["Agent"]["ID"]])

def doTokenCount(input_path: str, output_path: str, use_name=False):
    monitor_output = load_data(input_path)

    agents = monitor_output["Agents"]
    history = monitor_output["History"]
    agents.append(
        {
            "ID": "system",
            "Name": "system"
        }
    )
    init_token_map(agents, use_name)
    build_influence_map(agents, history, use_name)
    get_token_map(output_path)
