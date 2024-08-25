import sys
sys.path.append("..")
from agentmonitor import AgentMonitor
from typing import List
import os
import logging
import asyncio
import json
import shutil
from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from argparse import ArgumentParser
from dataloader import dataloader_registry

parser = ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="codetest/gsm8k",
)
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "agentverse_task"),
)
parser.add_argument("--dataset_path", type=str, default="../codetest_data")
parser.add_argument("--output_path", type=str, default="output/agentverse_codetest/3.5_8b_8b_3.5_3.5_8b")
parser.add_argument("--has_tools", action="store_true")
parser.add_argument("--tool_tmp_path", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)

def get_dataloader(task, dataset_path):
    current_path = os.path.dirname(__file__)
    return dataloader_registry.build(task, path=f"{current_path}/{dataset_path}")

async def do_monitor(agentversepipeline, monitor):
    for agent_or_agents in agentversepipeline.environment.agents.values():
        if isinstance(agent_or_agents, List):
            for agent in agent_or_agents:
                await monitor.register(agent, agent.add_message_to_memory, agent.astep, name=agent.name, use_partial=True)
                print(f"agent monitor 1: {agent.name}")
        else:
            await monitor.register(agent_or_agents, agent_or_agents.add_message_to_memory, agent_or_agents.astep, name=agent_or_agents.name, use_partial=True)
            print(f"agent monitor 2: {agent_or_agents.name}")

def cli_main(
    debug=False,
):
    
    """

    llm_config_list: list of llm config file names, stored in META_GPT_ROOT/config/ , and currently order is hard coded for simplicity

    :param task:
    :param total_num:
    :param llm_config_list:
    :return:
    """
    task = args.task.split("/")[-1]
    print(task)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(f"{args.output_path}/{task}/agentverse_output.jsonl"), exist_ok=True)
    if os.path.exists(f"{args.output_path}_turn_1/{task}/{task}_result.json"):
        logger.warning(f"{args.output_path}_turn_1/{task}/{task}_result.json exists, skip this config.")
        exit()
    dataset_path = f"{args.dataset_path}/{task}.jsonl"
    dataloader = get_dataloader(args.task, dataset_path)

    shutil.copyfile(
        f"{args.tasks_dir}/{args.task}/config.yaml",
        f"{args.output_path}/{task}/config.yaml",
    )

    skip_cnt = 0
    if not args.overwrite and os.path.exists(f"{args.output_path}/{task}/agentverse_output.jsonl"):
        with open(f"{args.output_path}/{task}/agentverse_output.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    skip_cnt += 1
    f = open(f"{args.output_path}/{task}/agentverse_output.jsonl", "w" if args.overwrite else "a")
    test_num = 0
    for i, example in enumerate(dataloader):
        if i < skip_cnt:
            continue
        logger.info(f"Input: {example['input']}\nAnswer: {example['answer']}")
        if args.has_tools:
            assert args.tool_tmp_path is not None
            with open(args.tool_tmp_path, "w") as f:
                f.write(json.dumps(example["tools"]))
        agentverse = TaskSolving.from_task(args.task, args.tasks_dir)
        agentverse.environment.set_task_description(example['input'])
        monitor = AgentMonitor()
        asyncio.run(do_monitor(agentverse, monitor))
        print(len(monitor.agents.keys()))
        kwargs = {
            "store_intermediate_step": True,
            "monitor": monitor,
            "task_instruction": example['input'],
            "output_path_prefix": f"{args.output_path}",
            "output_path_postfix": f"{task}/task_{str(i)}.json",
        }
        plan, result, logs, plans = agentverse.run(**kwargs)
        # monitor.recording(f"{args.output_path}/{task}/recording_result.json")
        f.write(
            json.dumps(
                {
                    "input": example['input'],
                    "responses": plans,
                    "label": example['answer'],
                    "logs": logs,
                }
            )
            + "\n"
        )
        f.flush()
        test_num += 1
        if test_num == 1:
            if debug:
                break
    f.close()

if __name__ == "__main__":
    cli_main()
