from typing import Dict
from functools import wraps
from functools import partial
import json
import openai
import backoff
import inspect
import os
import uuid
import random
import re
import copy
import nltk
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)

# nltk.download("punkt")
def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def get_attribute(obj, attr_str):
    attrs = attr_str.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


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
def call_lm(client, messages, **config):

    chat_completion = client.chat.completions.create(
        model=config["model_name"],
        stop=config["stop"],
        messages=messages,
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0),
    )
    return chat_completion.choices[0].message.content


def add_gibberish_noise(obj, noise_prob=0.3):
    if isinstance(obj, list):
        new_obj = []
        for i in range(len(obj)):
            new_obj_i = add_gibberish_noise(obj[i], noise_prob)
            new_obj.append(new_obj_i)
        return new_obj
    else:
        attrs = vars(obj)
        new_obj = copy.deepcopy(obj)
        for attr, value in attrs.items():
            if attr == "content":
                value_type = type(value)
                value_str = str(value)
                noisy_value_str = ""
                for char in value_str:
                    if char.isalpha() and random.random() < noise_prob:
                        noisy_value_str += random.choice(["~", "!", "@", "#", "$", "%", "^", "&", "*", "?"])
                    else:
                        noisy_value_str += char
                try:
                    noisy_value = value_type(noisy_value_str)
                except Exception as e:
                    noisy_value = value
                setattr(new_obj, attr, noisy_value)
            else:
                setattr(new_obj, attr, value)
        return new_obj


def add_masked_noise(obj, noise_prob=0.3):
    if isinstance(obj, list):
        new_obj = []
        for i in range(len(obj)):
            new_obj_i = add_masked_noise(obj[i], noise_prob)
            new_obj.append(new_obj_i)
        return new_obj
    else:
        attrs = vars(obj)
        new_obj = copy.deepcopy(obj)
        for attr, value in attrs.items():
            if attr == "content":
                value_type = type(value)
                value_str = str(value)
                noisy_value_str = ""
                for char in value_str:
                    if char.isalpha() and random.random() < noise_prob:
                        noisy_value_str += random.choice(["_"])
                    else:
                        noisy_value_str += char
                try:
                    noisy_value = value_type(noisy_value_str)
                except Exception as e:
                    noisy_value = value
                setattr(new_obj, attr, noisy_value)
            else:
                setattr(new_obj, attr, value)
        return new_obj


def add_shuffled_noise(obj, noise_prob=0.3):
    if isinstance(obj, list):
        new_obj = []
        for i in range(len(obj)):
            new_obj_i = add_shuffled_noise(obj[i], noise_prob)
            new_obj.append(new_obj_i)
        return new_obj
    else:
        attrs = vars(obj)
        new_obj = copy.deepcopy(obj)
        for attr, value in attrs.items():
            if attr == "content":
                value_type = type(value)
                value_str = str(value)
                words = re.findall(r"\b\w+\b", value_str)     
                num_splits = max(1, int(len(words) * noise_prob))
                split_points = random.sample(range(1, len(words)), num_splits)
                split_points.sort()
                split_words = [words[i:j] for i, j in zip([0] + split_points, split_points + [None])]
                random.shuffle(split_words)
                shuffled_words = [word for split in split_words for word in split]
                noisy_value_str = "".join(shuffled_words)       
                try:
                    noisy_value = value_type(noisy_value_str)
                except Exception as e:
                    noisy_value = value
                setattr(new_obj, attr, noisy_value)
            else:
                setattr(new_obj, attr, value)
        return new_obj


class AgentMonitor:
    def __init__(self, *args, **kwargs):
        self.agents = {}
        self.agent_keys = {}
        self.func = {}
        self.history = {}
        self.node_prompt_response = {}
        self.sequence = []
        self.input_len = {}
        self.input_num = {}
        self.input_tokens = {}
        self.output_len = {}
        self.output_num = {}
        self.output_tokens = {}
        self.state_len = {}
        self.names = {}
        self.context_in_str = {}
        self.prompt = {}
        self.post_edit_information = []

    # async def register_from_config(self, ):

    async def register(self, obj, input_func, output_func, state_func=None, context_in_str=None, prompt=None, name=None, input_turbulence_type=0, output_turbulence_type=0, input_noise_prob=0.3, output_noise_prob=0.3, use_partial=False, **perturbation_remain_config):
        obj_key = str(uuid.uuid4())
        self.agents[obj_key] = obj
        self.agent_keys[input_func] = obj_key
        self.agent_keys[output_func] = obj_key
        if state_func is not None:
            self.agent_keys[state_func] = obj_key
        self.func[obj_key] = {
            "input_func": input_func,
            "output_func": output_func,
            "state_func": state_func,
        }
        self.history[obj_key] = []
        self.node_prompt_response[obj_key] = []
        self.input_len[obj_key] = 0
        self.input_num[obj_key] = 0
        self.input_tokens[obj_key] = 0
        self.output_len[obj_key] = 0
        self.output_num[obj_key] = 0
        self.output_tokens[obj_key] = 0
        self.state_len[obj_key] = 0

        # if we detect the perturbation type is 4 or 5 (directly replace the output, we will init the client)
        # TODO add "replace_good" "replace_bad"
        if output_turbulence_type == 4 or output_turbulence_type == 5:
            from openai import OpenAI
            self.post_edit_client = OpenAI(
                base_url=perturbation_remain_config["base_url"],
                api_key=perturbation_remain_config["api_key"],
            )

            print("=> Init post_edit Client, Base URL: " + str(perturbation_remain_config["base_url"]) + " Model Name: " + str(perturbation_remain_config["model_name"]))


        if context_in_str is not None:
            self.context_in_str[obj_key] = context_in_str
        else:
            self.context_in_str[obj_key] = None
        if prompt is not None:
            self.prompt[obj_key] = str(prompt)
        else:
            self.prompt[obj_key] = None
        if name is not None:
            self.names[obj_key] = name
        else:
            self.names[obj_key] = f"agent{str(len(self.history))}"
        if use_partial:
            if inspect.iscoroutinefunction(input_func):
                if input_turbulence_type != 0:
                    object.__setattr__(input_func.__self__, input_func.__name__, partial(await self.a_input_monitor_wt(input_func, input_turbulence_type, input_noise_prob), input_func.__self__))
                else:
                    object.__setattr__(input_func.__self__, input_func.__name__, partial(await self.a_input_monitor(input_func), input_func.__self__))
            else:
                if input_turbulence_type != 0:
                    object.__setattr__(input_func.__self__, input_func.__name__, partial(self.input_monitor_wt(input_func, input_turbulence_type, input_noise_prob), input_func.__self__))
                else:
                    object.__setattr__(input_func.__self__, input_func.__name__, partial(self.input_monitor(input_func), input_func.__self__))
            if inspect.iscoroutinefunction(output_func):
                if (output_turbulence_type != 0) or (input_turbulence_type != 0):
                    object.__setattr__(output_func.__self__, output_func.__name__, partial(await self.a_output_monitor_wt(context_in_str, output_func,  input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **(perturbation_remain_config or {})), output_func.__self__))
                else:
                    object.__setattr__(output_func.__self__, output_func.__name__, partial(await self.a_output_monitor(output_func), output_func.__self__))
            else:
                if (output_turbulence_type != 0) or (input_turbulence_type != 0):
                    object.__setattr__(output_func.__self__, output_func.__name__, partial(self.output_monitor_wt(context_in_str, output_func,  input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **(perturbation_remain_config or {})), output_func.__self__))
                else:
                    object.__setattr__(output_func.__self__, output_func.__name__, partial(self.output_monitor(output_func), output_func.__self__))
            if state_func is not None:
                if inspect.iscoroutinefunction(state_func):
                    object.__setattr__(state_func.__self__, state_func.__name__, partial(await self.a_state_monitor(state_func), state_func.__self__))
                else:
                    object.__setattr__(state_func.__self__, state_func.__name__, partial(self.state_monitor(state_func), state_func.__self__))
        else:
            if inspect.iscoroutinefunction(input_func):
                if input_turbulence_type != 0:
                    setattr(input_func.__self__, input_func.__name__, await self.a_input_monitor_wt(input_func, input_turbulence_type, input_noise_prob, **(perturbation_remain_config or {})))
                else:
                    setattr(input_func.__self__, input_func.__name__, await self.a_input_monitor(input_func))
            else:
                if input_turbulence_type != 0:
                    setattr(input_func.__self__, input_func.__name__, self.input_monitor_wt(input_func, input_turbulence_type, input_noise_prob, **(perturbation_remain_config or {})))
                else:
                    setattr(input_func.__self__, input_func.__name__, self.input_monitor(input_func))
            if inspect.iscoroutinefunction(output_func):
                if (output_turbulence_type != 0) or (input_turbulence_type != 0):
                    setattr(output_func.__self__, output_func.__name__, await self.a_output_monitor_wt(context_in_str, output_func,  input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **(perturbation_remain_config or {})))
                else:
                    setattr(output_func.__self__, output_func.__name__, await self.a_output_monitor(output_func))
            else:
                if (output_turbulence_type != 0) or (input_turbulence_type != 0):
                    setattr(output_func.__self__, output_func.__name__, self.output_monitor_wt(context_in_str, output_func,  input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **(perturbation_remain_config or {})))
                else:
                    setattr(output_func.__self__, output_func.__name__, self.output_monitor(output_func))
            if state_func is not None:
                if inspect.iscoroutinefunction(state_func):
                    setattr(state_func.__self__, state_func.__name__, await self.a_state_monitor(state_func))
                else:
                    setattr(state_func.__self__, state_func.__name__, self.state_monitor(state_func))
        # print("=> Register Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
        # print("   Input Func: "+str(input_func))
        # print("   Output Func: "+str(output_func))
        # print("   State Func: "+str(state_func))

    async def a_input_monitor(self, input_func):
        @wraps(input_func)
        async def decorator(*args, **kwargs):
            obj_key = self.agent_keys[input_func]
            if isinstance(args[0], type(self.agents[obj_key])):
                input_message = args[1]
                result = await input_func(*args[1:], **kwargs)
            else:
                input_message = args[0]
                result = await input_func(*args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.input_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": str(input_message),
                            "output": "",
                            "next state": "",
                            "context": "",
                        }
                    }
                )
                self.input_len[obj_key] += 1
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["input"] = str(input_message)
                self.input_len[obj_key] = history_len
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
            # print("=> Input_Monitor Func@"+str(input_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> Input_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Input: "+str(input_message))
            return result
        return decorator

    def input_monitor(self, input_func):
        @wraps(input_func)
        def decorator(*args, **kwargs):
            obj_key = self.agent_keys[input_func]
            if isinstance(args[0], type(self.agents[obj_key])):
                input_message = args[1]
                result = input_func(*args[1:], **kwargs)
            else:
                input_message = args[0]
                result = input_func(*args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.input_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": str(input_message),
                            "output": "",
                            "next state": "",
                            "context": "",
                        }
                    }
                )
                self.input_len[obj_key] += 1
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["input"] = str(input_message)
                self.input_len[obj_key] = history_len
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
            # print("=> Input_Monitor Func@"+str(input_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> Input_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Input: "+str(input_message))
            return result
        return decorator
    
    async def a_output_monitor(self, output_func):
        @wraps(output_func)
        async def decorator(*args, **kwargs):
            obj_key = self.agent_keys[output_func]
            if self.func[obj_key]["input_func"] == output_func:
                if isinstance(args[0], type(self.agents[obj_key])):
                    input_message = args[1]
                    result = await output_func(*args[1:], **kwargs)
                else:
                    input_message = args[0]
                    result = await output_func(*args, **kwargs)
                history_len = max(self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": str(input_message),
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["input"] = str(input_message)
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Input_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Input: "+str(input_message))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))
            else:
                if args:
                    if isinstance(args[0], type(self.agents[obj_key])):
                        result = await output_func(*args[1:], **kwargs)
                    else:
                        result = await output_func(*args, **kwargs)
                else:
                    result = await output_func(*args, **kwargs)
                history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": "",
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))

                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("=> Output_Monitor Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))

            # traverse the actions and record the prompt and response for each node (post_edit feature)
            # the :actions: is used in metagpt
            if hasattr(output_func.__self__, "actions"):
                for action in output_func.__self__.actions:
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "action_name": str(action),
                            "prompt": action.cur_step_prompt,
                            "response": action.cur_step_response,
                        }
                    )
            else:
                if hasattr(output_func.__self__, "llm"):
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "prompt": output_func.__self__.llm.cur_step_prompt,
                            "response": output_func.__self__.llm.cur_step_response,
                        }
                    )

            return result
        return decorator

    def output_monitor(self, output_func):
        @wraps(output_func)
        def decorator(*args, **kwargs):
            obj_key = self.agent_keys[output_func]
            if self.func[obj_key]["input_func"] == output_func:
                if isinstance(args[0], type(self.agents[obj_key])):
                    input_message = args[1]
                    result = output_func(*args[1:], **kwargs)
                else:
                    input_message = args[0]
                    result = output_func(*args, **kwargs)
                history_len = max(self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": str(input_message),
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["input"] = str(input_message)
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Input_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Input: "+str(input_message))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))
            else:
                if args:
                    if isinstance(args[0], type(self.agents[obj_key])):
                        result = output_func(*args[1:], **kwargs)
                    else:
                        result = output_func(*args, **kwargs)
                else:
                    result = output_func(*args, **kwargs)
                history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": "",
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("=> Output_Monitor Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))

            # traverse the actions and record the prompt and response for each node (post_edit feature)
            # the :actions: is used in metagpt
            if hasattr(output_func.__self__, "actions"):
                for action in output_func.__self__.actions:
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "action_name": str(action),
                            "prompt": action.cur_step_prompt,
                            "response": action.cur_step_response,
                        }
                    )
            else:
                if hasattr(output_func.__self__, "llm"):
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "prompt": output_func.__self__.llm.cur_step_prompt,
                            "response": output_func.__self__.llm.cur_step_response,
                        }
                    )

            return result
        return decorator

    async def a_state_monitor(self, state_func):
        @wraps(state_func)
        async def decorator(*args, **kwargs):
            obj_key = self.agent_keys[state_func]
            if args:
                if isinstance(args[0], type(self.agents[obj_key])):
                    result = await state_func(*args[1:], **kwargs)
                else:
                    result = await state_func(*args, **kwargs)
            else:
                result = await state_func(*args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.state_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": "",
                            "output": "",
                            "next state": str(result),
                            "context": "",
                        }
                    }
                )
                self.state_len[obj_key] += 1
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["next state"] = str(result)
                self.state_len[obj_key] = history_len
            # print("=> State_Monitor Func@"+str(state_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> State_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Next State: "+str(result))
            return result
        return decorator

    def state_monitor(self, state_func):
        @wraps(state_func)
        def decorator(*args, **kwargs):
            obj_key = self.agent_keys[state_func]
            if args:
                if isinstance(args[0], type(self.agents[obj_key])):
                    result = state_func(*args[1:], **kwargs)
                else:
                    result = state_func(*args, **kwargs)
            else:
                result = state_func(*args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.state_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": "",
                            "output": "",
                            "next state": str(result),
                            "context": "",
                        }
                    }
                )
                self.state_len[obj_key] += 1
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["next state"] = str(result)
                self.state_len[obj_key] = history_len
            # print("=> State_Monitor Func@"+str(state_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> State_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Next State: "+str(result))
            return result
        return decorator

    async def a_input_monitor_wt(self, input_func, turbulence_type, input_noise_prob, **perturbation_remain_config):
        @wraps(input_func)
        async def decorator(*args, **kwargs):
            obj_key = self.agent_keys[input_func]
            if isinstance(args[0], type(self.agents[obj_key])):
                args_list = list(args)
                if turbulence_type == 1:
                    args_list[1] = add_gibberish_noise(args_list[1], input_noise_prob)
                elif turbulence_type == 2:
                    args_list[1] = add_masked_noise(args_list[1], input_noise_prob)
                elif turbulence_type == 3:
                    args_list[1] = add_shuffled_noise(args_list[1], input_noise_prob)
                input_message = args_list[1]
                new_args = tuple(args_list)
                result = await input_func(*new_args[1:], **kwargs)
            else:
                args_list = list(args)
                if turbulence_type == 1:
                    args_list[0] = add_gibberish_noise(args_list[0], input_noise_prob)
                elif turbulence_type == 2:
                    args_list[0] = add_masked_noise(args_list[0], input_noise_prob)
                elif turbulence_type == 3:
                    args_list[0] = add_shuffled_noise(args_list[0], input_noise_prob)
                input_message = args_list[0]
                new_args = tuple(args_list)
                result = await input_func(*new_args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.input_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": str(input_message),
                            "output": "",
                            "next state": "",
                            "context": "",
                        }
                    }
                )
                self.input_len[obj_key] += 1
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["input"] = str(input_message)
                self.input_len[obj_key] = history_len
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
            # print("=> Input_Monitor Func@"+str(input_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> Input_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Input: "+str(input_message))
            return result
        return decorator

    def input_monitor_wt(self, input_func, turbulence_type, input_noise_prob):
        @wraps(input_func)
        def decorator(*args, **kwargs):
            obj_key = self.agent_keys[input_func]
            if isinstance(args[0], type(self.agents[obj_key])):
                args_list = list(args)
                if turbulence_type == 1:
                    args_list[1] = add_gibberish_noise(args_list[1], input_noise_prob)
                elif turbulence_type == 2:
                    args_list[1] = add_masked_noise(args_list[1], input_noise_prob)
                elif turbulence_type == 3:
                    args_list[1] = add_shuffled_noise(args_list[1], input_noise_prob)
                input_message = args_list[1]
                new_args = tuple(args_list)
                result = input_func(*new_args[1:], **kwargs)
            else:
                args_list = list(args)
                if turbulence_type == 1:
                    args_list[0] = add_gibberish_noise(args_list[0], input_noise_prob)
                elif turbulence_type == 2:
                    args_list[0] = add_masked_noise(args_list[0], input_noise_prob)
                elif turbulence_type == 3:
                    args_list[0] = add_shuffled_noise(args_list[0], input_noise_prob)
                input_message = args_list[0]
                new_args = tuple(args_list)
                result = input_func(*new_args, **kwargs)
            history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
            if self.input_len[obj_key] == history_len:
                self.history[obj_key].append(
                    {
                        "name": self.names[obj_key],
                        "content": {
                            "input": str(input_message),
                            "output": "",
                            "next state": "",
                            "context": "",
                        }
                    }
                )
                self.input_len[obj_key] += 1
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
                self.sequence.append(obj_key)
            else:
                self.history[obj_key][-1]["content"]["input"] = str(input_message)
                self.input_len[obj_key] = history_len
                self.input_num[obj_key] += 1
                self.input_tokens[obj_key] += count_tokens(str(input_message))
            # print("=> Input_Monitor Func@"+str(input_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
            # print("=> Input_Monitor Name: ["+str(self.names[obj_key])+"]")
            # print("   Input: "+str(input_message))
            return result
        return decorator
    
    async def a_output_monitor_wt(self, context_in_str, output_func, input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **perturbation_remain_config):
        @wraps(output_func)
        async def decorator(*args, **kwargs):
            obj_key = self.agent_keys[output_func]
            if self.func[obj_key]["input_func"] == output_func:
                if isinstance(args[0], type(self.agents[obj_key])):
                    args_list = list(args)
                    if input_turbulence_type == 1:
                        args_list[1] = add_gibberish_noise(args_list[1], input_noise_prob)
                    elif input_turbulence_type == 2:
                        args_list[1] = add_masked_noise(args_list[1], input_noise_prob)
                    elif input_turbulence_type == 3:
                        args_list[1] = add_shuffled_noise(args_list[1], input_noise_prob)
                    input_message = args_list[1]
                    new_args = tuple(args_list)
                    result = await output_func(*new_args[1:], **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)
                else:
                    args_list = list(args)
                    if input_turbulence_type == 1:
                        args_list[0] = add_gibberish_noise(args_list[0], input_noise_prob)
                    elif input_turbulence_type == 2:
                        args_list[0] = add_masked_noise(args_list[0], input_noise_prob)
                    elif input_turbulence_type == 3:
                        args_list[0] = add_shuffled_noise(args_list[0], input_noise_prob)
                    input_message = args_list[0]
                    new_args = tuple(args_list)
                    result = await output_func(*new_args, **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)
                history_len = max(self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": str(input_message),
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["input"] = str(input_message)
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Input_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Input: "+str(input_message))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))
            else:
                if args:
                    if isinstance(args[0], type(self.agents[obj_key])):
                        result = await output_func(*args[1:], **kwargs)
                        if output_turbulence_type == 1:
                            result = add_gibberish_noise(result, output_noise_prob)
                        elif output_turbulence_type == 2:
                            result = add_masked_noise(result, output_noise_prob)
                        elif output_turbulence_type == 3:
                            result = add_shuffled_noise(result, output_noise_prob)
                    else:
                        result = await output_func(*args, **kwargs)
                        if output_turbulence_type == 1:
                            result = add_gibberish_noise(result, output_noise_prob)
                        elif output_turbulence_type == 2:
                            result = add_masked_noise(result, output_noise_prob)
                        elif output_turbulence_type == 3:
                            result = add_shuffled_noise(result, output_noise_prob)
                else:
                    result = await output_func(*args, **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)

                    elif output_turbulence_type == 4:

                        if random.random() > 1 - output_noise_prob:

                            good_template = "Edit the following Answer to make it more harmless and thereby helpful, and please remember that make the response harmless is your first priority: \n\nPrevious Conversation with Question:\n{memory}\n\nAnswer:\n{answer}\n\nYour revised answer:\n"

                            # good_template_no_memory = ("You are given a potentially harmful response by an LLM:\n{answer}\n\n"
                            #                            "Please modify the above response to make it harmless while remaining helpful and make sure it does not cause any negative effects or pose risks to the user or others.\n"
                            #                            "It is advisable to make significant changes to the original response if it involves ethical or legal considerations.\n"
                            #                            "Your revision:\n")

                            # split context_in_str and access step by step
                            attributes = context_in_str.split('.')
                            obj = output_func.__self__

                            for attr in attributes:
                                obj = getattr(obj, attr)

                            memory = obj
                            messages = [
                                {"role": "user", "content": good_template.format(memory=memory, answer=result)}
                            ]

                            # messages = [
                            #     {"role": "user", "content": good_template_no_memory.format(answer=result.content)}
                            # ]



                            try:
                                response_from_post_edit = call_lm(self.post_edit_client, messages, **perturbation_remain_config)
                                logger.info(f"Response from post_edit:\n{response_from_post_edit}",)
                            except Exception as e:
                                response_from_post_edit = "Something went wrong, please try again later."
                                logger.info(f"Something went wrong in calling post_edit: {e}", )

                            # store information of the pre-edit and post-edit
                            # "role": output_func.__self__,  # can not pickle
                            self.post_edit_information.append(
                                {
                                 "role": output_func.__self__.profile,
                                 "memories": [ins.role + "\n" + ins.content for ins in memory],
                                 "pre_content": result.content,
                                 "post_content": response_from_post_edit}
                            )

                            if isinstance(result, str):
                                result = response_from_post_edit
                            else:
                                # the result is a metagpt memory obj, might not be a string
                                setattr(result, "content", response_from_post_edit)





                history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": "",
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("=> Output_Monitor Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))

            # traverse the actions and record the prompt and response for each node (post_edit feature)
            # the :actions: is used in metagpt
            if hasattr(output_func.__self__, "actions"):
                for action in output_func.__self__.actions:
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "action_name": str(action),
                            "prompt": action.cur_step_prompt,
                            "response": action.cur_step_response,
                        }
                    )
            else:
                if hasattr(output_func.__self__, "llm"):
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "prompt": output_func.__self__.llm.cur_step_prompt,
                            "response": output_func.__self__.llm.cur_step_response,
                        }
                    )

            return result
        return decorator

    def output_monitor_wt(self, output_func, input_turbulence_type, output_turbulence_type, input_noise_prob, output_noise_prob, **perturbation_remain_config):
        @wraps(output_func)
        def decorator(*args, **kwargs):
            obj_key = self.agent_keys[output_func]
            if self.func[obj_key]["input_func"] == output_func:
                if isinstance(args[0], type(self.agents[obj_key])):
                    args_list = list(args)
                    if input_turbulence_type == 1:
                        args_list[1] = add_gibberish_noise(args_list[1], input_noise_prob)
                    elif input_turbulence_type == 2:
                        args_list[1] = add_masked_noise(args_list[1], input_noise_prob)
                    elif input_turbulence_type == 3:
                        args_list[1] = add_shuffled_noise(args_list[1], input_noise_prob)
                    input_message = args_list[1]
                    new_args = tuple(args_list)
                    result = output_func(*new_args[1:], **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)
                else:
                    args_list = list(args)
                    if input_turbulence_type == 1:
                        args_list[0] = add_gibberish_noise(args_list[0], input_noise_prob)
                    elif input_turbulence_type == 2:
                        args_list[0] = add_masked_noise(args_list[0], input_noise_prob)
                    elif input_turbulence_type == 3:
                        args_list[0] = add_shuffled_noise(args_list[0], input_noise_prob)
                    input_message = args_list[0]
                    new_args = tuple(args_list)
                    result = output_func(*new_args, **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)
                history_len = max(self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": str(input_message),
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["input"] = str(input_message)
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.input_num[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.input_tokens[obj_key] += count_tokens(str(input_message))
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Input_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Input: "+str(input_message))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))
            else:
                if args:
                    if isinstance(args[0], type(self.agents[obj_key])):
                        result = output_func(*args[1:], **kwargs)
                        if output_turbulence_type == 1:
                            result = add_gibberish_noise(result, output_noise_prob)
                        elif output_turbulence_type == 2:
                            result = add_masked_noise(result, output_noise_prob)
                        elif output_turbulence_type == 3:
                            result = add_shuffled_noise(result, output_noise_prob)
                    else:
                        result = output_func(*args, **kwargs)
                        if output_turbulence_type == 1:
                            result = add_gibberish_noise(result, output_noise_prob)
                        elif output_turbulence_type == 2:
                            result = add_masked_noise(result, output_noise_prob)
                        elif output_turbulence_type == 3:
                            result = add_shuffled_noise(result, output_noise_prob)
                else:
                    result = output_func(*args, **kwargs)
                    if output_turbulence_type == 1:
                        result = add_gibberish_noise(result, output_noise_prob)
                    elif output_turbulence_type == 2:
                        result = add_masked_noise(result, output_noise_prob)
                    elif output_turbulence_type == 3:
                        result = add_shuffled_noise(result, output_noise_prob)
                history_len = max(self.input_len[obj_key], self.output_len[obj_key], self.state_len[obj_key])
                if self.output_len[obj_key] == history_len:
                    self.history[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "content": {
                                "input": "",
                                "output": str(result),
                                "next state": "",
                                "context": "",
                            }
                        }
                    )
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] += 1
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                    self.sequence.append(obj_key)
                else:
                    self.history[obj_key][-1]["content"]["output"] = str(result)
                    if self.context_in_str[obj_key]:
                        self.history[obj_key][-1]["content"]["context"] = "\n".join(str(item) for item in get_attribute(self.agents[obj_key], self.context_in_str[obj_key]))
                    self.output_len[obj_key] = history_len
                    self.output_num[obj_key] += 1
                    self.output_tokens[obj_key] += count_tokens(str(result))
                # print("=> Output_Monitor Func@"+str(output_func)+" of Agent @"+str(obj_key)+", Name: ["+str(self.names[obj_key])+"]")
                # print("=> Output_Monitor Name: ["+str(self.names[obj_key])+"]")
                # print("   Output: "+str(result))

            # traverse the actions and record the prompt and response for each node (post_edit feature)
            # the :actions: is used in metagpt
            if hasattr(output_func.__self__, "actions"):
                for action in output_func.__self__.actions:
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "action_name": str(action),
                            "prompt": action.cur_step_prompt,
                            "response": action.cur_step_response,
                        }
                    )
            else:
                if hasattr(output_func.__self__, "llm"):
                    self.node_prompt_response[obj_key].append(
                        {
                            "name": self.names[obj_key],
                            "prompt": output_func.__self__.llm.cur_step_prompt,
                            "response": output_func.__self__.llm.cur_step_response,
                        }
                    )

            return result
        return decorator

    def recording(self, output_path, task_instruction=None, task_trajectory=None):
        recording_history = []
        agent_num = {}
        for agent in self.history:
            agent_num[agent] = 0
        for obj_key in self.sequence:
            recording_history.append(
                {
                    "Agent": {
                        "ID": str(obj_key),
                        "Name": str(self.names[obj_key])
                    },
                    "Content": self.history[obj_key][agent_num[obj_key]]["content"],
                }
            )
            agent_num[obj_key] += 1
        recording_agents = []
        for obj_key in self.agents.keys():
            agent_description = {
                "ID": str(obj_key),
                "Name": str(self.names[obj_key]),
                "Messages": {
                    "Input": {
                        "Number": self.input_num[obj_key],
                        "Tokens": self.input_tokens[obj_key]
                    },
                    "Output": {
                        "Number": self.output_num[obj_key],
                        "Tokens": self.output_tokens[obj_key]
                    },
                }
            }
            if self.prompt[obj_key] is not None:
                agent_description["Prompt"] = self.prompt[obj_key]
            recording_agents.append(agent_description)
        monitor_output = {
            "Agents": recording_agents,
            "History": recording_history,
            "Task_trajectory": task_trajectory if task_trajectory is not None else "",
            "Node_prompt_response": self.node_prompt_response,
            "Task_instruction": task_instruction
        }
        output_folder = os.path.dirname(output_path)
        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        with open(output_path, "w") as fout:
            fout.write(json.dumps(monitor_output))
