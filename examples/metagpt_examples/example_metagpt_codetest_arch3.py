import os
import sys
sys.path.append("..")
from metagpt.config2 import Config
from model_path_mapping import path_mapping
from agentmonitor import AgentMonitor
import re
import ast
import fire
import json
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
import jsonlines
import yaml
from metagpt.const import METAGPT_ROOT
from tqdm import tqdm
import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import traceback
import multiprocessing
import serpapi
# from tools.duckduckgo_search import DuckDuckGoSearch
from tools.google_search import GoogleSearch
from tools.Serper import SerperGoogleSearch
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, TimeoutException
import backoff

# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY = ""

global_instruction = ""
global_code = ""
global_test = ""
global_answer = ""


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

def get_global_instruction():
    global global_instruction
    return str(global_instruction)

def set_global_instruction(instruction):
    global global_instruction
    global_instruction = instruction

def get_global_code():
    global global_code
    return str(global_code)

def set_global_code(code):
    global global_code
    global_code = code

def get_global_test():
    global global_test
    return str(global_test)

def set_global_test(test):
    global global_test
    global_test = test

def get_global_answer():
    global global_answer
    return str(global_answer)

def set_global_answer(answer):
    global global_answer
    global_answer = answer

def read_yaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

def extract_answer(raw_answer):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, raw_answer, re.DOTALL)
    ans = ""
    if len(matches) > 0:
        ans = str(matches[0])
    return ans.strip()


def act_code_with_timeout(code, timeout=2):

    local_vars = {}
    logger.warning(f"currently acting the code:\n {code}")

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

def act_code(code):
    local_vars = {}

    logger.warning(f"currently acting the code:\n {code}")

    try:
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            exec(code, None, local_vars)
        output = captured_output.getvalue().strip()

        result = local_vars.get("result", "")
        if result == "":
            result += f"\n Captured Output:\n{output}\n"
            for var_name in local_vars:
                if callable(local_vars[var_name]) and output == "":
                    result += local_vars[var_name]()
                    break
        return result

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n"
        error_message += "Traceback:\n"
        error_message += traceback.format_exc()
        return error_message

# TODO deprecated global usage, it might not be suitable if we want to store per step results
def task_test_using_global(task, reference):
    if task == "humaneval":
        code = str(get_global_code()) + "\n" + str(reference)
        try:
            result = act_code_with_timeout(code, timeout=2)
            if "error occurred" in result:
                 return False
            return True
        except Exception as e:
            return False
    else:
        if reference.lower() == get_global_answer().lower():
            return True
        else:
            return False

def task_test(task, code_or_answer, reference):
    if task == "humaneval":
        code = code_or_answer + "\n" + str(reference)
        try:
            result = act_code_with_timeout(code, timeout=2)
            if "error occurred" in result:
                return False
            return True
        except Exception as e:
            return False
    else:
        if reference.lower() == code_or_answer.lower():
            return True
        else:
            return False

class SimpleWriteCode(Action):
    # PROMPT_TEMPLATE: str = """
    # Write a python function with no arguments that can solve the following task: {instruction}.
    # Return ```python your_code_here ``` with NO other texts,
    # your code:
    # """
    PROMPT_TEMPLATE: str = """\
Finish the following python function as prompted: 

<instruction>
{instruction}
</instruction>

Below is the conversation history, you can use it as context to help you modify or maintain your original answer.

<conversation_history>
{conversation_history}
</conversation_history>

Please provide a self-contained python function that can solve the task and response it in a markdown code block.

For example:

Your code:
```python
your code here
```
---

Your code:
    """
    name: str = "SimpleWriteCode"
    cur_step_prompt: str = None
    cur_step_response: str = None

    # This line is aim to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**
    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=context[0], conversation_history="\n".join(str(i) for i in context[1:]))
        rsp = await self._aask(prompt)
        code_text = parse_code(rsp)
        set_global_code(str(code_text))
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return code_text


class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement, SimpleWriteTest, SimpleWriteReview])
        self.set_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        # context = self.get_memories(k=1)[0].content # use the most recent memory as context
        context = self.get_memories()  # use all memories as context
        code_text = await todo.run(context,)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        
        self.rc.memory.add(msg)

        return msg


class ModifyCode(Action):

    PROMPT_TEMPLATE: str = """\
<instruction>
{instruction}
</instruction>

<conversation_history>
{conversation_history}
</conversation_history>

You are given the above instructions and conversation history. You are acting as an engineer to modify the code. Your peers have proposed the initial code and some have also reviewed and tested it. 
Please take this information into account and provide a refined and self-contained Python function that can solve the task. 
Please respond using a markdown Python code block.

For example:

Your code:
```python
your code here
```
---

Your code:
    """
    name: str = "SimpleWriteCode"
    cur_step_prompt: str = None
    cur_step_response: str = None

    # This line is aim to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**
    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=context[0], conversation_history="\n".join(str(i) for i in context[1:]))
        rsp = await self._aask(prompt)
        code_text = parse_code(rsp)
        set_global_code(str(code_text))
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return code_text

class CodeModifier(Role):
    name: str = "David"
    profile: str = "CodeModifier"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([SimpleWriteCode])
        self.set_actions([ModifyCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        # context = self.get_memories(k=1)[0].content # use the most recent memory as context
        context = self.get_memories()  # use all memories as context
        code_text = await todo.run(context,)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        
        self.rc.memory.add(msg)

        return msg

class SimpleWriteTest(Action):
    PROMPT_TEMPLATE: str = """\
<context>
{context}
</context>

Write {k} unit tests using pytest for the given function, assuming you have imported it.
Return a python code in a markdown code block.
your code:
    """
    name: str = "SimpleWriteTest"
    cur_step_prompt: str = None
    cur_step_response: str = None

    # This line is aim to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**
    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context="\n".join(str(i) for i in context), k=k)
        rsp = await self._aask(prompt)
        code_text = parse_code(rsp)
        set_global_test(str(code_text))
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return code_text

class SimpleTester(Role):
    name: str = "Bob"
    profile: str = "SimpleTester"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteTest])
        # self._watch([SimpleWriteCode])
        self._watch([SimpleWriteCode, SimpleWriteReview])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        # context = self.get_memories(k=1)[0].content # use the most recent memory as context
        context = self.get_memories()  # use all memories as context
        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        
        self.rc.memory.add(msg)
        return msg

class SimpleWriteReview(Action):
    PROMPT_TEMPLATE: str = """\
<context>
{context}
</context>

Review the test cases and provide one critical comments:
    """
    name: str = "SimpleWriteReview"
    cur_step_prompt: str = None
    cur_step_response: str = None

    # This line is aim to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**
    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context="\n".join(str(i) for i in context))
        rsp = await self._aask(prompt)
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return rsp

class SimpleReviewer(Role):
    name: str = "Charlie"
    profile: str = "SimpleReviewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteReview])
        self._watch([SimpleWriteTest])

class AnswerExtract(Action):
    PROMPT_TEMPLATE: str = """\
<context>
{context}
</context>

<code> 
{code}
</code>

<test>
{test}
</test>

After executing the code, the result is {result}.

<task>
{instruction}
</task>

Based on the upper information, provide an answer for the original task. If you are not sure, provide an answer anyway. 
Return your answer only wrapped by <answer> and </answer>, do not contain other irrelevant words.

For example:
<answer> A </answer>

Your Answer: 

    """
    name: str = "AnswerExtract"
    cur_step_prompt: str = None
    cur_step_response: str = None
    # This line aims to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**
    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str, result: str):
        prompt = self.PROMPT_TEMPLATE.format(context="\n".join(str(i) for i in context), code=get_global_code(), test=get_global_test(), result=result, instruction=get_global_instruction())
        rsp = await self._aask(prompt)
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return rsp

class AnswerExtractor(Role):
    name: str = "Danny"
    profile: str = "AnswerExtractors"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([AnswerExtract])
        self._watch([SimpleWriteTest])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        context = self.get_memories()
        # result = act_code(str(get_global_code()) + "\n" + str(get_global_test()))
        result = act_code_with_timeout(str(get_global_code()))
        ouput = await todo.run(context=context, result=result)
        set_global_answer(extract_answer(ouput))
        msg = Message(content=ouput, role=self.profile, cause_by=type(todo))
        
        self.rc.memory.add(msg)
        return msg

class WebBrowse(Action):
    PROMPT_TEMPLATE: str = """\
<instruction>
{instruction}
</instruction>


<information>
{information}
</information>


You are given the instruction and also the relevant documents retrieved from the internet website, please give your suggestions towards solving the task.
Your suggestions:
    """
    QUERY_REFINE_TEMPLATE: str = """\
<instruction>
{instruction}
</instruction>

<previous search results>
{search_results}
</previous search results>

You are given the above instruction, and the corresponding histories of previous searched results.
Please check whether it is expected and provide a more appropriate query for searching on the internet.
Please directly output your refined query without any explanation.

Refined Query:

"""
    name: str = "WebBrowse"
    cur_step_prompt: str = None
    cur_step_response: str = None
    # This line aims to make recording each step's prompt and response easier, maybe we need to refactor it in the future.
    # **make our framework do this automatically**


    def record_cur_step_prompt_and_response(self, prompt: str, response: str):
        self.cur_step_prompt = prompt
        self.cur_step_response = response

    async def run(self, context: str, web_search: SerperGoogleSearch):
        instruction = context[0]
        query = await self._aask(self.QUERY_REFINE_TEMPLATE.format(instruction=instruction, search_results="\n".join(str(i) for i in context[1:])))

        try:
            search_snippets = web_search.search(query.strip('"').strip("'"))
            search_snippets = search_snippets[:10]
            logger.info(f"web browser get search_snippets: {search_snippets}")
        except Exception as e:
            search_snippets = f"Failed to retrieve search results. Error: {str(e)}"

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction, information=search_snippets)
        rsp = await self._aask(prompt)
        self.record_cur_step_prompt_and_response(prompt, rsp)
        return rsp


    @backoff.on_exception(
        backoff.expo,
        (DuckDuckGoSearchException, RatelimitException, TimeoutException),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def duckduckgo_search(self, query):
        results = DDGS().text(query, max_results=5)

        snippets = ""
        for result in results:
            snippets += result["body"] + "\n"

        return snippets


    def google_search(self, query):
        params = {
            "engine": "google",
            "q": query,
            "api_key": GOOGLE_API_KEY,
            "google_domain": "google.com",
        }
        search = serpapi.search(params)
        search_result_raw = search.get_dict()
        search_results = search_result_raw.get("organic_results", [])
        snippets = [result["snippet"] for result in search_results[:5]]

        return "\n".join(snippets)


    def scrape_search(self, query):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
        if response.status_code != 200:
            return "Failed to retrieve search results."
        soup = BeautifulSoup(response.text, "html.parser")
        # results = soup.select(".tF2Cxc")
        # snippets = [result.select_one(".aCOpRe").get_text() for result in results[:5] if result.select_one(".aCOpRe")]
        # return " ".join(snippets)
        all_texts = soup.get_text(separator="\n")
        lines = all_texts.split("\n")
        filtered_lines = [line for line in lines if line.strip() != ""]
        return "\n".join(filtered_lines[:5])




class WebBrowser(Role):
    name: str = "Emily"
    profile: str = "WebBrowser"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([WebBrowse])
        self.web_search = SerperGoogleSearch()


    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        context = self.get_memories()
        ouput = await todo.run(context, self.web_search)
        msg = Message(content=ouput, role=self.profile, cause_by=type(todo))
        
        self.rc.memory.add(msg)
        return msg

async def main(
    task: str = "humaneval",
    output_path: str = "output/test/test_8b",
    llm_config_files=["llama3_8b_instruct_8003.yaml", "llama3_8b_instruct_8003.yaml", "llama3_8b_instruct_8003.yaml", "llama3_8b_instruct_8003.yaml", "llama3_8b_instruct_8003.yaml"],
    # llm_config_files=["gpt_3.5_turbo.yaml", "gpt_3.5_turbo.yaml", "gpt_3.5_turbo.yaml", "gpt_3.5_turbo.yaml"]
    # llm_config_files=["llama3_70b_instruct.yaml", "llama3_8b_instruct.yaml", "llama3_8b_instruct.yaml", "llama3_8b_instruct.yaml", "llama3_8b_instruct.yaml"],
    perturbation_config=None,
    debug=False,
    overwrite_output=False,
    n_round=3, # for arch other than base we use turn 3
):

    """

    llm_config_list: list of llm config file names, stored in META_GPT_ROOT/config/ , and currently order is hard coded for simplicity

    :param task:
    :param total_num:
    :param llm_config_list:
    :return:
    """

    # sanity check
    if perturbation_config is not None:
        assert "perturbation_config" in output_path, "if you specify perturbation_config, then it should store in perturbation dir"

    task_data = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # first check whether the final turn results exist, if so exit directly.
    if os.path.exists(f"{output_path}_turn_1/{task}/{task}_result.json"):
        logger.warning(f"{output_path}_turn_1/{task}/{task}_result.json exists, skip this config or overwrite.")

        if not overwrite_output:
            exit()
        else:
            logger.warning(f"Overwriting existing output path {output_path}_turn_1/{task}/{task}_result.json!!!")
            # actually we do not need to remove it, we just need to overwrite it.

    with jsonlines.open(f"../codetest_data/{task}.jsonl") as reader:
        for line in reader:
            task_data.append(line)

    if debug:
        task_data = task_data[20:22]
    """
    user config llm
    example:
    llm_config = {"api_type": "xxx", "api_key": "xxx", "model": "xxx"}
    gpt4 = Config.from_llm_config(llm_config)
    A = Role(name="A", profile="Democratic candidate", goal="Win the election", actions=[a1], watch=[a2], config=gpt4)
    """

    llm_configs_dicts = []
    for llm_config_file in process_list(llm_config_files):
        cur_llm_config = read_yaml(METAGPT_ROOT / "config" / llm_config_file)
        if cur_llm_config["llm"]["use_vllm"]:
            cur_llm_config["llm"]["model"] = path_mapping[cur_llm_config["llm"]["model"]]

        llm_configs_dicts.append(Config.from_llm_config(cur_llm_config["llm"]))

    perturbation2int = {"gibberish": 1, "mask": 2, "shuffle": 3, "no_perturbation": 0}
    if perturbation_config is not None:
        perturbation_config = read_yaml(perturbation_config)
        perturbation_config = perturbation_config["perturbation"]
    else:
        # a large number that index will not overflow
        perturbation_config = [{'type': 'no_perturbation', 'ratio': 0} for _ in range(10)]
        perturbation_remain_config = None

    # it contains total turn results as [{"turn1": ..., "turn2": ..., "turn3": ...}, {"turn1": ..., "turn2": ...}]
    total_results = []
    for task_line in tqdm(task_data):
        # task configuration
        idea: str = task_line["prompt"]
        set_global_instruction(idea)
        task_id = task_line["id"]
        reference = task_line["reference"]
        investment: float = 3.0
        add_human: bool = False
        # metagpt solve the task
        logger.info(idea)
        monitor = AgentMonitor()
        team = Team()
        simplecoder = SimpleCoder(config=llm_configs_dicts[0])
        simpletester = SimpleTester(config=llm_configs_dicts[1])
        simplereviewer = SimpleReviewer(is_human=add_human, config=llm_configs_dicts[2])
        webbrowser = WebBrowser(config=llm_configs_dicts[3])
        if task != "humaneval":
            answerextractor = AnswerExtractor(config=llm_configs_dicts[4])

        await monitor.register(simplecoder, simplecoder.put_message, simplecoder._act, simplecoder._think, context_in_str="rc.memory.storage", prompt=simplecoder.actions[0].PROMPT_TEMPLATE, name="simplecoder", input_turbulence_type=perturbation2int[perturbation_config[0]["type"]], input_noise_prob=perturbation_config[0]["ratio"], **(perturbation_remain_config or {}))
        await monitor.register(simpletester, simpletester.put_message, simpletester._act, simpletester._think, context_in_str="rc.memory.storage", prompt=simpletester.actions[0].PROMPT_TEMPLATE, name="simpletester", input_turbulence_type=perturbation2int[perturbation_config[1]["type"]], input_noise_prob=perturbation_config[1]["ratio"], **(perturbation_remain_config or {}))
        await monitor.register(simplereviewer, simplereviewer.put_message, simplereviewer._act, simplereviewer._think, context_in_str="rc.memory.storage", prompt=simplereviewer.actions[0].PROMPT_TEMPLATE, name="simplereviewer", input_turbulence_type=perturbation2int[perturbation_config[2]["type"]], input_noise_prob=perturbation_config[2]["ratio"], **(perturbation_remain_config or {}))
        await monitor.register(webbrowser, webbrowser.put_message, webbrowser._act, webbrowser._think,
                               context_in_str="rc.memory.storage", prompt=webbrowser.actions[0].PROMPT_TEMPLATE,
                               name="webbrowser",
                               input_turbulence_type=perturbation2int[perturbation_config[3]["type"]],
                               input_noise_prob=perturbation_config[3]["ratio"],
                               **(perturbation_remain_config or {}))
        if task != "humaneval":
            await monitor.register(answerextractor, answerextractor.put_message, answerextractor._act, answerextractor._think, context_in_str="rc.memory.storage", name="answerextractor", input_turbulence_type=perturbation2int[perturbation_config[4]["type"]], input_noise_prob=perturbation_config[4]["ratio"], **(perturbation_remain_config or {}))
        if task != "humaneval":
            team.hire(
                [
                    simplecoder,
                    simpletester,
                    simplereviewer,
                    answerextractor,
                    webbrowser
                ]
            )
        else:
            team.hire(
                [
                    simplecoder,
                    simpletester,
                    simplereviewer,
                    webbrowser
                ]
            )
        team.invest(investment=investment)
        team.run_project(idea)


        kwargs = {
            "store_intermediate_step": True,
            "monitor": monitor,
            "output_path_prefix": f"{output_path}",
            "output_path_postfix": f"{task}/task_{str(task_id)}.json",
            "task_instruction": task_line["prompt"],
            # "task_trajectory": team.env.history, # not this one, it just contain task instruction
            ## for access to global function
            "get_global_code": get_global_code,
            "get_global_test": get_global_test,
            "get_global_answer": get_global_answer,
            "task_test_using_global": task_test_using_global
        }

        task_history, all_turn_results = await team.run(n_round=n_round, **kwargs)
        total_results.append(all_turn_results)
        # Note, I move these lines into team.run, because we need intermediate step's results
        # monitor.recording(f"{output_path}/{task}/task_{str(task_id)}.json", task_instruction=task_line["prompt"], task_trajectory=team.env.history)
    # TODO, we do not use global_var here, we use the per step results return from team.run
    turn_success_dict = {}
    for cur_result, meta_info in zip(total_results, task_data):
        for cur_turn, cur_turn_ins in cur_result.items():

            # {"turn1": ..., "turn2": ...}
            # cur_turn["answer"], cur_turn["code"], cur_turn["test"] already exists, we store it in team.run()
            if task == "humaneval":
                is_solve = task_test(task, cur_turn_ins["get_global_code"], meta_info["reference"])
                cur_turn_ins["is_solve"] = is_solve
            else:
                is_solve = task_test(task, cur_turn_ins["get_global_answer"], meta_info["reference"])
                cur_turn_ins["is_solve"] = is_solve
                cur_turn_ins["result"] = str(act_code_with_timeout(str(cur_turn_ins["get_global_code"])))

            if cur_turn not in turn_success_dict:
                turn_success_dict[cur_turn] = {"success_num": 0}

            if is_solve:
                turn_success_dict[cur_turn]["success_num"] += 1


    # calculate success rate
    for cur_turn, cur_turn_success_dict in turn_success_dict.items():
        cur_turn_success_dict["total_num"] = len(total_results)
        success_rate = cur_turn_success_dict["success_num"] / cur_turn_success_dict["total_num"]
        cur_turn_success_dict["success_rate"] = success_rate

    # save success_rate into cur_turn dir, and name it as {task}_result.json as previous.
        with open(f"{output_path}_{cur_turn}/{task}/{task}_result.json", "w") as fout:
            fout.write(json.dumps(cur_turn_success_dict))

if __name__ == "__main__":
    fire.Fire(main)
