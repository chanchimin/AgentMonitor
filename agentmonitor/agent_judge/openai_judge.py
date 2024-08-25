import os
from openai import OpenAI
import openai
import traceback
import backoff
import json
from agentmonitor.agent_judge.base_judge import BaseJudge
# for openai==1.x
# from openai import OpenAI
# for openai==0.x
# import openai
# import traceback
# import backoff

class OpenaiJudge(BaseJudge):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        assert "api_key" in config.keys()

        if "base_url" not in config:
            base_url = "https://api.openai.com/v1"
        else:
            base_url = config["base_url"]

        self.client = OpenAI(
            base_url=base_url,
            api_key=config["api_key"],
        )

        self.stop = config.get("stop", None)

        llm_as_a_judge_prompt_path = os.path.join(os.path.dirname(__file__), f"../../prompt/llm_as_a_judge_no_rationale.txt")
        with open(llm_as_a_judge_prompt_path, "r") as fin:
            self.llm_as_a_judge_prompt = fin.read()
        llm_as_a_judge_overall_prompt_path = os.path.join(os.path.dirname(__file__), f"../../prompt/llm_as_a_judge_overall_no_rationale.txt")
        with open(llm_as_a_judge_overall_prompt_path, "r") as fin:
            self.llm_as_a_judge_overall_prompt = fin.read()

    @backoff.on_exception(
        backoff.fibo,
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
        jitter=backoff.full_jitter,
    )
    def call_lm(self, messages):

        chat_completion = self.client.chat.completions.create(
            model=self.config["model_name"],
            stop=self.stop,
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
        )
        return chat_completion.choices[0].message.content


    def judging(self, task, input_path, output_path, use_name):

        with open(input_path, "r") as fin:
            monitor_output = json.load(fin)
        history = monitor_output["History"]
        trajectory = monitor_output["Task_trajectory"]
        agents = monitor_output["Agents"]
        node_prompt_response = monitor_output["Node_prompt_response"]
        judge_rules = {}
        for agent in agents:
            agent_id = agent["ID"]
            agent_name = agent["Name"]
            if use_name:
                prompt_path = os.path.join(os.path.dirname(__file__), f"../../prompt/{agent_name}.txt")
            else:
                prompt_path = os.path.join(os.path.dirname(__file__), f"../../prompt/{agent_id}.txt")
            with open(prompt_path, "r") as fin:
                judge_rules[agent_id] = fin.read()
        judge_output = []
        for agent in agents:
            agent_id = agent["ID"]
            agent_name = agent["Name"]

            for agent_action in node_prompt_response[agent_id]:

                # TODO judge the agents' own duty
                expected_duties_prompt = self.llm_as_a_judge_prompt.format_map({
                    "expected_duties": judge_rules[agent_id],
                    "conversation_history": agent_action["prompt"] + agent_action["response"],
                    "agent_id": agent_id,
                    "agent_name": agent_name
                })

                expected_duties_response = self.call_lm(
                    [{"role": "user", "content": expected_duties_prompt}]
                )

                system_goal = monitor_output["Task_instruction"]
                if task == "humaneval":
                    system_goal = "Please provide a self-contained code that solves the following problem in a markdown code block:\n" + system_goal

                # TODO judge the agents' impact on the overall system performance
                overall_performance_prompt = self.llm_as_a_judge_overall_prompt.format_map({
                    "system_goal": system_goal,
                    "conversation_history": trajectory,
                    "agent_id": agent_id,
                    "agent_name": agent_name
                })

                overall_performance_response = self.call_lm(
                    [{"role": "user", "content": overall_performance_prompt}]
                )

                judge_output.append(
                    {
                        "Agent": {
                            "ID": agent["ID"],
                            "Name": agent["Name"],
                        },
                        "Action": agent_action,
                        "Expected_Duties_Judgement": expected_duties_response,
                        "Overall_Performance_Judgement": overall_performance_response,
                    }
                )
        output_folder = os.path.dirname(output_path)
        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        with open(output_path, "w") as fout:
            fout.write(json.dumps(judge_output))
