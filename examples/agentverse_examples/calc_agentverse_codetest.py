import jsonlines
import json
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr
import io
import traceback

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

def main(
    # task = "gsm8k",
    # task = "humaneval",
    task = "mmlu",
    task_path = "output/agentverse_codetest/8b_8b_8b_3.5_3.5_3.5", # input_path & output_path
):
    agentverse_output = []
    with jsonlines.open(f"{task_path}/{task}/agentverse_output.jsonl") as reader:
        for line in reader:
            agentverse_output.append(line)
    total_num = []
    success_num = []
    result_list = []
    final_total_num = 0
    final_success_num = 0
    final_result_list = []
    # example = agentverse_output[0]
    # for _ in range(len(example["responses"])):
    #     total_num.append(0)
    #     success_num.append(0)
    #     result_list.append([])
    if task == "humaneval":
        for line in agentverse_output:
            for i in range(len(line["responses"])):
                if i >= len(total_num):
                    total_num.append(0)
                if i >= len(success_num):
                    success_num.append(0)
                if i >= len(result_list):
                    result_list.append([])
                result = False
                if calc_exec(line["responses"][i], line["label"]):
                    success_num[i] += 1
                    result = True
                result_list[i].append(result)
                total_num[i] += 1
            if calc_exec(line["responses"][-1], line["label"]):
                final_success_num += 1
                final_result_list.append(True)
            else:
                final_result_list.append(False)
            final_total_num += 1
    else:
        for line in agentverse_output:
            for i in range(len(line["responses"])):
                if i >= len(total_num):
                    total_num.append(0)
                if i >= len(success_num):
                    success_num.append(0)
                if i >= len(result_list):
                    result_list.append([])
                result = False
                if calc_diff(line["responses"][i], line["label"]):
                    success_num[i] += 1
                    result = True
                result_list[i].append(result)
                total_num[i] += 1
            if calc_diff(line["responses"][-1], line["label"]):
                final_success_num += 1
                final_result_list.append(True)
            else:
                final_result_list.append(False)
            final_total_num += 1
    final_result_per_turn = []
    final_result = {
        "total_num": final_total_num,
        "success_num": final_success_num,
        "success_rate": float(final_success_num)/float(final_total_num),
        "result_list": final_result_list
    }
    total_len = max(len(total_num), len(success_num), len(result_list))
    for i in range(total_len):
        final_result_per_turn.append(
            {
                "cur_turn": i+1,
                "total_num": total_num[i],
                "success_num": success_num[i],
                "success_rate": float(success_num[i])/float(total_num[i]),
                "result_list": result_list[i]
            }     
        )
    final_result_output = {
        "result_per_turn": final_result_per_turn,
        "final_result": final_result
    }
    with open(f"{task_path}/{task}/final_result.json", "w") as fout:
        fout.write(json.dumps(final_result_output))

if __name__ == "__main__":
    main()
