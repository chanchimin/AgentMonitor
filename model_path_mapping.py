# Description: Mapping the model path to the actual path
llama3_8b_path = "meta-llama/Meta-Llama-3-8B-Instruct"
llama3_70b_path = "casperhansen/llama-3-70b-instruct-awq"
uncensored_path = "Orenguteng/Llama-3-8B-Lexi-Uncensored"

path_mapping = {
    # for the machine which can connect to net
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
    "meta-llama/Meta-Llama-3-8B-Instruct": llama3_8b_path,
    "casperhansen/llama-3-70b-instruct-awq": llama3_70b_path,
    "Orenguteng/Llama-3-8B-Lexi-Uncensored": uncensored_path,
    # for the mainland machine
    # add your custom path mapping
}

