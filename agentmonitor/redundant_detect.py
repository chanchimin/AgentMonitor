import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
def load_data(input_path: str):
    with open(input_path, "r") as fin:
        monitor_output: Dict = json.load(fin)
    return monitor_output

def calc_self_similarity(history):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(history)
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    self_similarity = similarities.mean(axis=1).mean()
    return self_similarity

def calc_cross_similarity(history, other_history):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(history+other_history)
    similarities = cosine_similarity(tfidf_matrix[:len(history)], tfidf_matrix[len(history):])
    self_similarity = similarities.mean(axis=1).mean()
    return self_similarity

def detectRedundantWork(input_path: str, output_path: str, use_name=False):
    monitor_output = load_data(input_path)

    agents = monitor_output["Agents"]
    history = monitor_output["History"]
    output = {}
    for agent in agents:
        agent_id = agent["ID"]
        agent_name = agent["Name"]
        output[agent_id] = {
            "name": agent_name,
            "history": []
        }
    for chat in history:
        if chat["Content"]["output"] != "":
            output[chat["Agent"]["ID"]]["history"].append(chat["Content"]["output"])
    self_similarity = {}
    cross_similarity = {}
    for agent in output.keys():
        if use_name:
            self_similarity[output[agent]["name"]] = calc_self_similarity(output[agent]["history"])
        else:
            self_similarity[agent] = calc_self_similarity(output[agent]["history"])
        other_history = []
        for other_agent in output.keys():
            if other_agent != agent:
                other_history += output[other_agent]["history"]
        if use_name:
            cross_similarity[output[agent]["name"]] = calc_cross_similarity(output[agent]["history"], other_history)
        else:
            cross_similarity[agent] = calc_cross_similarity(output[agent]["history"], other_history)
    detect_result = {
        "self similarity": self_similarity,
        "cross similarity": cross_similarity
    }
    output_folder = os.path.dirname(output_path)
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    with open(output_path, "w") as fout:
            fout.write(json.dumps(detect_result))