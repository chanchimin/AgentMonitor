<div align="center">
<img src="figs/crop_agentmonitor/crop_agentmonitor_1.png" width="500px"/>
<br />
<br />



<h1 align="center"> AgentMonitor </h1>


</div>

## Introduction
AgentMonitor is a framework designed to capture and analyze inputs and outputs at each step of the agent interaction process. By transforming this data into meaningful statistics, AgentMonitor enables the (1) training of regression models—such as XGBoost—to predict the downstream task performance of multi-agent teams. (2) post-edit the responses generated each step on-the-fly.


## Get Started


## Installation

~~~bash
git clone https://github.com/chanchimin/agentmonitor.git
cd agentmonitor
conda create -n agentmonitor python=3.9
pip install -r requirements.txt
~~~


## Basic Usage



~~~py
import fire
from multibench import AgentMonitor

class Agent:
    # ...

    # get messages / update history
    def put_message(self, message):
        # ...
        return result

    # take actions / generate responses
    def act(self,
        # ...
    ):
        # ...
        return result

async def main():
    monitor = AgentMonitor()
    agent = Agent()
    await monitor.register(agent, agent.put_message, agent.act) # register target agent
    # arguments of function register:
    #     obj: necessary, agent object
    #     input_func: necessary, input function of specific agent object
    #     output_func: necessary, output function of specific agent object
    #     state_func: state function of specific agent object, default=None
    #     context_in_str: context attribution of agent as str, default=None
    #     prompt: role prompt of agent as str, default=None
    #     input_turbulence_type: turbulence type of input function, can be 0/1/2/3, default=0 (no turbulence)
    #     output_turbulence_type: turbulence type of output function, can be 0/1/2/3, default=0 (no turbulence)
    #     input_noise_prob: probability of adding turbulence for input function, default=0.3
    #     output_noise_prob: probability of adding turbulence for output function, default=0.3
    #     name: name of agent as str, default=None
    #     use_partial: necessary, use partial tool or not, default=False
    # for example: await monitor.register(simplecoder, simplecoder.put_message, simplecoder._act, simplecoder._think, context_in_str="rc.memory", prompt=simplecoder.actions[0].PROMPT_TEMPLATE, name="simplecoder")
    # ...
    # Note: output path must be a json file ("monitor_output_test.json").
    monitor.recording("monitor_output_test.json") # record monitor history 

if __name__ == "__main__":
    fire.Fire(main)
~~~

### Disclaimer

Please note that the code provided above are for illustrative purposes only. They demonstrate the basic usage, but may not be directly applicable to all scenarios or use cases.

For specific experiments or applications, modifications may be necessary. We strongly recommend referring to the `examples` directory for more detailed and specific use cases.

## RoadMap

Our framework is designed to provide non-invasive, plug-and-play monitoring behavior that integrates seamlessly with various multi-agent frameworks. While this design is intended to be broadly applicable, different frameworks have unique requirements, which may necessitate specific adjustments.

For instance, back to the time we run our experiments, we encountered a limitation with MetaGPT, which did not support passing stop-word arguments—a feature necessary for custom models like "llama3_8b_instruct" at that moment. Therefore, we store the modifications in the ```examples/metagpt_examples/metagpt``` dir for replication purpose. 
Moving forward, we plan to generalize the framework further, making it easier for users to adopt in diverse environments.

Currently, supported frameworks include:
- [x] MetaGPT
- [x] AgentVerse
- [ ] AutoGen
- [ ] ChatDev

---
## Experiment Usage

Here, we provide the detailed steps for running experiments used in our paper.

### Step 1: Run the Multi-Agent Experiments with our AgentMonitor

```shell

cd examples/metagpt_examples

python example_metagpt_codetest_base.py \
--task gsm8k \
--output_path "output/base/8b_8b_8b_8b" \
--llm_config_files [\'llama3_8b_instruct_8003.yaml\',\'llama3_8b_instruct_8003.yaml\',\'llama3_8b_instruct_8003.yaml\',\'llama3_8b_instruct_8003.yaml\'] \

```

### Step 2: Use the stored information to calculate indicators (Agent Scores)

```shell

python judge_metagpt_codetest.py \
--task gsm8k \
--judge_llm_config_files "llama3_70b_instruct_8002.json" \
--input_path "output/base/8b_8b_8b_8b_turn_3"

```


### Step 3: Use the stored information to calculate indicators (Graph Attributes)

```shell


# 3.1 Calculate the token count for graph analysis
python tokencount_metagpt_codetest.py \
--task gsm8k \
--input_path "output/base/8b_8b_8b_8b_turn_3"

# 3.2 Calculate the graph attributes
python calc_graph_attributes.py \
--task gsm8k \
--input_path "output/base/8b_8b_8b_8b_turn_3"

```

### Step 4: Aggregate all the information

```shell


# aggregate all the runs into one file
python aggregate_statistics.py \
--task gsm8k \
--arch base \
--input_paths ['output/base/8b_8b_8b_8b_turn_3', 'output/base/8b_8b_8b_70b_turn_3', 'output/base/8b_8b_70b_8b_turn_3', ...] \
--output_path "statistic_output/base/gsm8k/total_results.csv" \
--judge_llm "judged_by_gpt_3.5_turbo"

```


### Step 5: Calculate the regression

```shell

python calc_regression.py \
--input_csv_paths \
"statistic_output/base/gsm8k/total_results.csv" \
... \
... \
--output_path \
"statistic_output/regression_results"

```
