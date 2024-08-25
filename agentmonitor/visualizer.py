import json
import os
from pyvis.network import Network
import matplotlib.pyplot as plt
import networkx as nx
import colorsys
import nltk
from typing import Dict


# nltk.download("punkt")
def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def generate_node_color(start_hue=120, end_hue=0, num_colors=30, start_alpha=0.3, end_alpha=0.8):
    colors = []
    for i in range(num_colors):
        hue = start_hue + (end_hue - start_hue) * (i / (num_colors - 1))
        alpha = start_alpha + (end_alpha - start_alpha) * (i / (num_colors - 1))
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.5, 1.0)
        color = f'#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}{int(alpha * 255):02X}'
        colors.append(color)  
    return colors

def load_data(input_path: str):
    with open(input_path, "r") as fin:
        monitor_output: Dict = json.load(fin)
    return monitor_output

def build_net(agents, history):
    agents_mapping = {}
    info_net = {
        "node": [],
        "edge": [],
        "tokens": []
    }
    i = 0
    for agent in agents:
        agent_name = agent["Name"]
        agent_id = agent["ID"]
        agents_mapping[agent_id] = i
        info_net["node"].append(
            {
                "num": i,
                "weight": 0,
                "name": agent_name
            }
        )
        i = i + 1
    total = i
    for node_from_init in range(total):
        weights = []
        for node_to_init in range(total):
            weights.append(0)
        info_net["edge"].append(weights)
        info_net["tokens"].append(weights)
    input_history = [
        {
            # this is system agent added in the last action
            "Agent": agents[-1],
            "Input": history[0]["Content"]["input"]
        }
    ]
    for info in history:
        message = info["Content"]["input"]
        for input_record in input_history:

            # NOTE: the logic here is because we store all the message during the env in metagpt runs, and it
            # contains case like {"input": "...", "output": ""}, that means cur agent do not subscribe to
            # the other agents' message, but metagpt framework will broadcast all the messages, so it means cur agent
            # receive the message.
            # so here we find whose messages route to the input of other agent, then we connect them.

            if message == input_record["Input"]:
                node_from = agents_mapping[input_record["Agent"]["ID"]]
                node_to = agents_mapping[info["Agent"]["ID"]]
                info_net["edge"][node_from][node_to] += 1
                token_num = count_tokens(message)
                info_net["tokens"][node_from][node_to] += token_num
                info_net["node"][node_from]["weight"] += 1
                info_net["node"][node_to]["weight"] += 1
                break
        if info["Content"]["output"]:
            input_history.append(
                {
                    "Agent": info["Agent"],
                    "Input": info["Content"]["output"]
                }
            )
    return info_net, total

def graph_analyze(G):
    # print("Total Number of Nodes: ", G.number_of_nodes())
    # print("Total Number of Edges: ", G.number_of_edges())
    # print("Average Clustering: ", nx.average_clustering(G))
    # print("Transitivity: ", nx.transitivity(G))
    degree_centralities = nx.degree_centrality(G)
    average_degree_centrality = 0
    degree_num = 0
    for degree_centrality in degree_centralities.keys():
        average_degree_centrality += degree_centralities[degree_centrality]
        degree_num += 1
    average_degree_centrality = average_degree_centrality / degree_num if degree_num > 0 else 0
    # print("Average Degree Centrality: ", average_degree_centrality)
    closeness_centralities = nx.closeness_centrality(G)
    average_closeness_centrality = 0
    closeness_num = 0
    for closeness_centrality in closeness_centralities.keys():
        average_closeness_centrality += closeness_centralities[closeness_centrality]
        closeness_num += 1
    average_closeness_centrality = average_closeness_centrality / closeness_num if closeness_num > 0 else 0
    # print("Average Closeness Centrality: ", average_closeness_centrality)
    betweenness_centralities = nx.betweenness_centrality(G)
    average_betweenness_centrality = 0
    betweenness_num = 0
    for betweenness_centrality in betweenness_centralities.keys():
        average_betweenness_centrality += betweenness_centralities[betweenness_centrality]
        betweenness_num += 1
    average_betweenness_centrality = average_betweenness_centrality / betweenness_num if betweenness_num > 0 else 0
    # print("Average Betweenness Centrality: ", average_betweenness_centrality)

    # print("PageRank: ", nx.pagerank(G, alpha=0.85))

    attributes_dict = {
        "total_number_of_nodes": G.number_of_nodes(),
        "total_number_of_edges": G.number_of_edges(),
        "average_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "average_degree_centrality": average_degree_centrality,
        "average_closeness_centrality": average_closeness_centrality,
        "average_betweenness_centrality": average_betweenness_centrality,
        "pagerank": nx.pagerank(G, alpha=0.85)
    }

    return attributes_dict
    # print("Diameter: ", nx.diameter(G))
    # print("Average Shortest Path Length): ", nx.average_shortest_path_length(G))

def draw_picture_png(info_net, total, node_color, netx_path):
    dg = nx.DiGraph()
    for node in info_net["node"]:
        # TODO, ADD explanation of why hard decoding here
        weight = node["weight"]
        if weight > 29:
            weight = 29
        name = node["name"]
        dg.add_node(node["num"], label=f"{name}", color=node_color[weight])
    for node_from in range(total):
        for node_to in range(total):
            if info_net["edge"][node_from][node_to] != 0:
                dg.add_edge(node_from, node_to, weight=info_net["tokens"][node_from][node_to])
    pos = nx.spring_layout(dg)
    netnode_labels = nx.get_node_attributes(dg, "label")
    netnode_colors = [dg.nodes[node]["color"] for node in dg.nodes]
    nx.draw(dg, pos=pos, node_size=800, node_color=netnode_colors, labels=netnode_labels, with_labels=True)
    graph_analyze(dg)
    plt.show()
    print(netx_path)
    plt.savefig(netx_path)

def draw_picture_html(info_net, total, node_color, output_path):
    net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
    for node in info_net["node"]:
        weight = node["weight"]
        if weight > 29:
            weight = 29
        name = node["name"]
        net.add_node(node["num"], label=f"{name}", color=node_color[weight])
    for node_from in range(total):
        for node_to in range(total):
            if info_net["edge"][node_from][node_to] != 0:
                net.add_edge(node_from, node_to, value=info_net["tokens"][node_from][node_to])
    net.barnes_hut(gravity=-10000)
    output_folder = os.path.dirname(output_path)
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    net.show(output_path, notebook=False)

def visualize(input_path: str, output_path: str):
    monitor_output = load_data(input_path)

    agents = monitor_output["Agents"]
    history = monitor_output["History"]

    agents.append(
        {
            "ID": "system",
            "Name": "system"
        }
    )
    info_net, total = build_net(agents, history)
    node_color = generate_node_color()
    netx_path = output_path.replace(".html", ".png")
    draw_picture_png(info_net, total, node_color, netx_path)
    draw_picture_html(info_net, total, node_color, output_path)