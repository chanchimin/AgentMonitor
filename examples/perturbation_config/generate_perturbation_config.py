import yaml
import itertools


perturbation_type = "replace"
perturbation_ratios = [1.0]
post_edit_model = "u8b"
agent_num = 3



# perturb cur agent
for agent_idx in range(agent_num):
    perturbation_config = [{'type': 'no_perturbation', 'ratio': 0} for _ in range(agent_num)]
    for ratio in perturbation_ratios:
        perturbation_config[agent_idx] = {'type': perturbation_type, 'ratio': ratio}

        filename = ""

        for cur_perturbation in perturbation_config:
            filename += f"{cur_perturbation['type'][0]}{int(cur_perturbation['ratio'] * 100):03d}"
            filename += "_"

        filename = filename[:-1] + f"_{post_edit_model}" + ".yaml"

        if perturbation_type != "replace":
            with open(filename, 'w') as file:
                yaml.dump({'perturbation': perturbation_config}, file, default_flow_style=False)

        else:
            with open(filename, 'w') as file:

                if post_edit_model == "u8b":
                    remain_config = {
                        "model_name": "Orenguteng/Llama-3-8B-Lexi-Uncensored",
                        "base_url": "http://localhost:8003/v1",
                        "api_key": "123",
                        "stop": "<|eot_id|>",
                    }
                elif post_edit_model == "8b":
                    remain_config = {
                        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                        "base_url": "http://localhost:8004/v1",
                        "api_key": "123",
                        "stop": "<|eot_id|>",
                    }

                yaml.dump({'perturbation': perturbation_config, 'remain_config': remain_config}, file, default_flow_style=False)

        print(f"Generated {filename}")
