import os
import re
import signal
import subprocess
import time
import tqdm
import numpy as np
from openai import OpenAI
from analysis_utils import get_code, getFilesFromType, get_text_embedding, get_code_embedding, get_cosine_similarity, remove_comments, get_task2cate

client = OpenAI(
    api_key='sk-xxxxx',
    base_url="https://yeysai.com/v1/",
)

task2cate = get_task2cate()

def get_category(directory):
    # assert os.path.isdir(directory)
    # prompt_filepath = getFilesFromType(directory, ".txt")[0]
    # task = open(prompt_filepath, "r").read().strip()

    # if task in task2cate.keys():
    #     return task2cate[task]
    return "NotFound"

def get_completeness(directory):
    assert os.path.isdir(directory)
    vn = get_code(directory)
    lines = vn.split("\n")

    lines = [line for line in lines if
             "password" not in line.lower() and "passenger" not in line.lower() and "passed" not in line.lower() and "passes" not in line.lower()]
    lines = [line for line in lines if "pass" in line.lower() or "todo" in line.lower()]
    if len(lines) > 0:
        return 0.0
    return 1.0

def get_executability(directory):
    assert os.path.isdir(directory)
    def findFile(directory, target):
        main_py_path = None
        for subroot, _, filenames in os.walk(directory):
            for filename in filenames:
                if target in filename:
                    main_py_path = os.path.join(subroot, filename)
        return main_py_path

    def exist_bugs(directory):
        assert os.path.isdir(directory)
        success_info = "The software run successfully without errors."
        try:
            command = "cd \"{}\"; ls -l; python3 main.py;".format(directory)
            process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            time.sleep(1)  # to 3 once stuck

            error_type = ""
            return_code = process.returncode
            # Check if the software is still running
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            if return_code == 0:
                return False, success_info, error_type
            else:
                error_output = process.stderr.read().decode('utf-8')
                try:
                    error_pattern = r'\w+Error:'
                    error_matches = re.findall(error_pattern, error_output)
                    error_type = error_matches[0].replace(":", "")
                except:
                    pass
                if error_output:
                    if "Traceback".lower() in error_output.lower():
                        errs = error_output.replace(directory + "/", "")
                        return True, errs, error_type
                else:
                    return False, success_info, error_type

        except subprocess.CalledProcessError as e:
            return True, f"Error: {e}", "subprocess.CalledProcessError"
        except Exception as ex:
            return True, f"An error occurred: {ex}", "OtherException"

        return False, success_info, error_type

    main_py_path = findFile(directory, ".py")
    pass_flag, error_type = True, ""
    if main_py_path is not None:
        main_py_path = os.path.dirname(main_py_path)
        bug_flag, info, error_type = exist_bugs(main_py_path)
        pass_flag = not bug_flag
    else:
        pass_flag, error_type = False, "NoMain"

    if error_type == "":
        error_type = info.replace("\n", "\\n")

    if pass_flag:
        return  1.0
    return 0.0

def get_consistency(directory):
    assert os.path.isdir(directory)
    files = getFilesFromType(directory, ".txt")
    if len(files) == 0:
        print()
    filepath = files[0]
    task = open(filepath).read().strip()
    codes = get_code(directory)
    codes = remove_comments(codes)

    text_embedding = get_text_embedding(task)
    code_embedding = get_code_embedding(codes)
    task_code_alignment = get_cosine_similarity(text_embedding, code_embedding)

    return task_code_alignment

def main(warehouse_root):
    def write_string(string):
        writer.write(string)
        print(string, end="")

    directories = []
    for directory in os.listdir(warehouse_root):
        if "NewFeature" not in directory:
            directories.append(os.path.join(warehouse_root, directory))
    directories = sorted(directories)
    directories = [directory for directory in directories if os.path.isdir(directory)]
    print("len(directories):", len(directories))

    # suffix = os.path.basename(warehouse_root)
    suffix = "__".join(warehouse_root.split("Desktop")[-3:]).replace("/", "__").replace("-", "_")
    tsv_file = __file__.replace(".py", ".{}.tsv".format(suffix))
    print("tsv_file:", tsv_file)
    content = ""
    if os.path.exists(tsv_file):
        content = open(tsv_file, "r").read()

    counter = 0
    completeness_list, executability_list, consistency_list = [], [], []
    with open(tsv_file, "a", encoding="utf-8") as writer:
        for i, directory in enumerate(directories):
            directory_basename = os.path.basename(directory)

            if directory_basename in content:
                print(directory_basename, "cached.")
                counter += 1
                continue

            category = "None" # get_category(directory)

            completeness = get_completeness(directory)
            executability = get_executability(directory)
            consistency = get_consistency(directory)

            completeness_list.append(completeness)
            executability_list.append(executability)
            consistency_list.append(consistency)

            string = "{}\t{}\t{}\t{}\t{}\n".format(directory_basename, category, completeness, executability, consistency)
            write_string(string)

            counter += 1

        if len(completeness_list) > 0:
            completeness_list = np.array(completeness_list)
            executability_list = np.array(executability_list)
            consistency = np.array(consistency)
            string = "{}\t\t{:.8f}\t{:.8f}\t{:.8f}\n".format("AVG", np.average(completeness_list), np.average(executability_list), np.average(consistency_list))
            write_string(string)

import sys
path_to_warehouse = sys.argv[1]
main(warehouse_root = path_to_warehouse)
