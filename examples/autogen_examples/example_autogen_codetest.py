import sys
sys.path.append("..")
import os
import jsonlines
import asyncio
from multibench import AgentMonitor
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from contextlib import redirect_stdout, redirect_stderr
import io
import traceback
import multiprocessing
import re
import json

def act_code_with_timeout(code, timeout=2):

    local_vars = {}

    def target(output_queue):
        captured_output = io.StringIO()

        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            error_info = ""
            try:
                exec(code, None, local_vars)
            except Exception as e:
                error_info += "An error occurred:\n" + traceback.format_exc()

        output_queue.put({"error_info": error_info, "captured_output": captured_output.getvalue()})

    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(output_queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return "An error occurred, execution timed out"
    else:
        returned_info = output_queue.get()

        if returned_info["error_info"] != "" or "An error occurred" in returned_info["captured_output"]:
            return returned_info["error_info"] + returned_info["captured_output"]
        else:
            return returned_info["captured_output"]

def calc_exec(response, reference):
    code = str(response) + "\n" + str(reference)
    try:
        result = act_code_with_timeout(code, timeout=2)
        if "error occurred" in result:
            return False
        return True
    except Exception as e:
        return False

def calc_diff(response, reference):
    if isinstance(response, list):
        response = response[0]
    if response.startswith("[") and response.endswith("]"):
        response = response.split("[")[-1].split("]")[0].split(",")[0].strip()
        if response.startswith("'") or response.endswith("'"):
            response = response.strip("'")
    if reference.lower() == response.lower():
        return True
    else:
        return False

async def do_monitor(agents, monitor):
    for agent in agents:
        await monitor.register(agent, agent.receive, agent.generate_reply, name=agent.name)

def cli_main(
    config_list = [
        {"model": "gpt-3.5-turbo", "api_key": "sk-xAaNIS7rt4wKdwXX18406e5842034f12A857776822015803", "base_url": "https://api3.apifans.com/v1"},
        {"model": "gpt-3.5-turbo", "api_key": "sk-xAaNIS7rt4wKdwXX18406e5842034f12A857776822015803", "base_url": "https://api3.apifans.com/v1"},
        {"model": "gpt-3.5-turbo", "api_key": "sk-xAaNIS7rt4wKdwXX18406e5842034f12A857776822015803", "base_url": "https://api3.apifans.com/v1"},
        {"model": "gpt-3.5-turbo", "api_key": "sk-xAaNIS7rt4wKdwXX18406e5842034f12A857776822015803", "base_url": "https://api3.apifans.com/v1"},
        {"model": "gpt-3.5-turbo", "api_key": "sk-xAaNIS7rt4wKdwXX18406e5842034f12A857776822015803", "base_url": "https://api3.apifans.com/v1"},
    ],
    task = "gsm8k",
    input_path = "../codetest_data",
    output_path = "output/autogen_codetest/3.5_3.5_3.5_3.5_3.5",
    debug = True,
):
    final_result = []
    total_num = 0
    success_num = 0
    llm_config = {"config_list": config_list, "cache_seed": 42}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    task_data = []
    with jsonlines.open(f"{input_path}/{task}.jsonl") as reader:
        for line in reader:
            task_data.append(line)
    if debug:
        task_data = task_data[:2]
    for task_line in task_data:
        monitor = AgentMonitor()
        task_message = task_line["prompt"]
        task_id = task_line["id"]
        reference = task_line["reference"]
        user_proxy = UserProxyAgent(
            name="User_proxy",
            system_message="A human admin.",
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "groupchat",
                "use_docker": False,
            },
            human_input_mode="NEVER",
        )
        if task == "humaneval":           
            coder = AssistantAgent(
                name="Coder",
                system_message=f"Finish the following python function as prompted:<instruction>{task_message}</instruction>, Please provide a self-contained python function that can solve the task and response it in a markdown code block. Reply \"TERMINATE\" in the end when everything is done. Your code: ",
                llm_config=llm_config,
            )
        else:
            if task == "gsm8k":
                coder = AssistantAgent(
                    name="Responser",
                    system_message=f"You are a responser to provide a final response for the following task: {task_message}. Please consider the provided conversation between other agents. You should put your final answer (ONE INTEGER ONLY) in <answer>your final answer only</asnwer>. Reply \"TERMINATE\" in the end when everything is done. Your response:",
                    llm_config=llm_config,
                )
            else:
                coder = AssistantAgent(
                    name="Responser",
                    system_message=f"You are a responser to provide a final response for the following task: {task_message}. Please consider the provided conversation between other agents. You should put your final answer (ONE CHARACTER ONLY) in <answer>your final answer only</asnwer>. Reply \"TERMINATE\" in the end when everything is done. Your response:",
                    llm_config=llm_config,
                )
        tester = AssistantAgent(
            name="Tester",
            system_message=f"Write unit tests using pytest for the given function, assuming you have imported it. Return a python code in a markdown code block. Reply \"TERMINATE\" in the end when everything is done. Your code:",
            llm_config=llm_config,
        )
        reviewer = AssistantAgent(
            name="Reviewer",
            system_message=f"Review the test cases and provide one critical comments. Reply \"TERMINATE\" in the end when everything is done. Your comments: ",
            llm_config=llm_config,
        )
        agents = [user_proxy, coder, tester, reviewer]
        groupchat = GroupChat(agents=agents, messages=[], max_round=3)
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        agents.append(manager)
        asyncio.run(do_monitor(agents, monitor))
        chat_result = user_proxy.initiate_chat(
            manager, message=task_message
        )
        if task == "humaneval":
            coder_responses = [entry['content'] for entry in chat_result.chat_history if entry.get('name') == 'Coder']
            if len(coder_responses) > 0:
                response = coder_responses[-1]
                pattern = r'```python(.*?)```'
                matches = re.findall(pattern, response, re.DOTALL)
                response = "\n".join(matches)
                text = response
                import_lines = re.findall(r'^(?!import pytest$).*?^import .+|^from .+?$', text, re.MULTILINE)
                imports = '\n'.join(import_lines)
                first_function = re.search(r'def .*?[\s\S]*?(?=\n\n|$)', text)
                function_code = first_function.group(0) if first_function else "# Not Found"
                response = f"{imports.strip()}\n\n{function_code.strip()}"
                is_solved = calc_exec(response, reference)
            else:
                response = "Fail to response."
                is_solved = False
        else:
            all = ""
            for cdict in chat_result.chat_history:
                all += cdict['content']
            matches = re.findall(r'<answer>(.*?)</answer>', all, re.DOTALL)
            response = matches[0]
            is_solved = calc_diff(response, reference)
        final_result.append(
            {
                "task_id": task_id,
                "response": response,
                "reference": reference,
                "solved": is_solved
            }
        )
        total_num += 1
        if is_solved:
            success_num += 1
        monitor.recording(f"{output_path}/{task}/monitor_output_autogen_{task_id}.json")
    final_result_output = {
        "Results": final_result,
        "Success Rate": float(success_num)/float(total_num)
    }
    with open(f"{output_path}/{task}/final_result.json", "w") as fout:
        fout.write(json.dumps(final_result_output))

if __name__ == "__main__":
    cli_main()