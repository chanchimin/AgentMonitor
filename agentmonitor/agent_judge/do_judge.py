import os
import json
import sys

sys.path.append("../")
from model_path_mapping import path_mapping

from typing import Dict
from agentmonitor.agent_judge.base_judge import BaseJudge
from agentmonitor.agent_judge.openai_judge import OpenaiJudge


def doJudge(task, input_path, output_path, llm_config_files, use_name=False):
    judge_config_path = os.path.join(os.path.dirname(__file__), "judge_config", llm_config_files)

    with open(judge_config_path, "r") as fin:
        config: Dict = json.load(fin)
    assert "judge_type" in config.keys()

    config["model_name"] = path_mapping[config["model_name"]]

    if config["judge_type"] == "OpenaiJudge":
        judge = OpenaiJudge(config)
    else:
        judge = BaseJudge(config)
    judge.judging(task, input_path, output_path, use_name)